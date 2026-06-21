#!/usr/bin/env python3
"""
SPSA tuner for the Ekagine chess engine.

Tunes the UCI-exposed `Tunable` search constants (see src/search.rs `TUNABLES`)
by simultaneous-perturbation self-play (theta+ vs theta-) under cutechess-cli.

Design goals (per the task):
  * Resumable + interruptible: the running estimate `theta` is persisted
    atomically after EVERY iteration. Killing the process (or `touch run/STOP`)
    at any time keeps all completed iterations. An in-flight, unfinished match
    is simply discarded -- never a partial/corrupt update.
  * Keeps the best validated params: every `checkpoint_interval` iterations we
    play a short match of the current rounded theta vs the engine defaults and,
    if it looks better, snapshot it to run/best.json.

Subcommands:
  tune     run the SPSA loop (default; resumes from run/state.json if present)
  sprt     play tuned-vs-base with a GSPRT stopping rule (the "conclusive" test)
  apply    rewrite src/search.rs Tunable defaults from a theta source
  status   print the current state

Everything is config-driven (tuning/config.json, auto-created on first run from
the engine's own UCI option list, so it stays in sync with src/search.rs).
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(HERE, "run")
CONFIG_PATH = os.path.join(HERE, "config.json")
STATE_PATH = os.path.join(RUN, "state.json")
BEST_PATH = os.path.join(RUN, "best.json")
PARAMS_TXT = os.path.join(RUN, "current_params.txt")
LOG_PATH = os.path.join(RUN, "log.txt")
STOP_FILE = os.path.join(RUN, "STOP")

DEFAULT_CONFIG = {
    "binary": os.path.join(ROOT, "target/aarch64-apple-darwin/release/chess_engine"),
    "cutechess": "/opt/cutechess/build/cutechess-cli",
    "book": "/Users/ekazuki/Downloads/8moves_v3.pgn",
    "engine_arg": "--uci",
    # game settings
    "st": 0.2,                 # seconds per move (fixed movetime; >=0.15 for pruning params)
    "timemargin": 400,         # ms tolerance over st; MUST be > 0 or contention -> time forfeits
    "concurrency": 8,
    "batch_games": 8,          # games per SPSA iteration (rounds = batch/2, with -repeat)
    "book_plies": 8,
    # adjudication (throughput; search-margin tuning is unaffected by score adjudication)
    "resign_movecount": 4,
    "resign_score": 700,
    "draw_movenumber": 40,
    "draw_movecount": 8,
    "draw_score": 10,
    # SPSA schedule
    "alpha": 0.602,
    "gamma": 0.101,
    "A": 200.0,
    "a0": 0.15,                # base learning rate (units of C_i per clean iteration)
    "c0": 1.0,                 # base perturbation (units of C_i at k=1)
    "max_iters": 0,            # 0 = run until stopped
    # checkpoint (current theta vs defaults) to track real progress + keep best
    "checkpoint_interval": 50,
    "checkpoint_games": 60,
    "avg_burn_in": 30,         # iters to skip before accumulating the Polyak average
    # final SPRT
    "sprt_elo0": 0.0,
    "sprt_elo1": 4.0,
    "sprt_alpha": 0.05,
    "sprt_beta": 0.05,
    "sprt_max_games": 4000,
    "sprt_batch": 16,
    "sprt_st": 0.5,            # validate at a slower TC than tuning to confirm transfer
}

STOP = False


def _handle_signal(signum, frame):
    global STOP
    STOP = True
    log(f"received signal {signum}: will stop cleanly after the current match")


# ── small utils ─────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def clampi(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


# ── engine introspection ────────────────────────────────────────────────────

OPT_RE = re.compile(
    r"option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)"
)


def discover_params(binary, engine_arg):
    """Query the engine's UCI options and return the tunable spin params
    (everything except Threads)."""
    proc = subprocess.run(
        [binary, engine_arg],
        input="uci\nquit\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    params = []
    for m in OPT_RE.finditer(proc.stdout):
        name, default, mn, mx = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if name == "Threads":
            continue
        params.append({"name": name, "default": default, "min": mn, "max": mx})
    if not params:
        raise SystemExit("no tunable spin options found from `uci` output")
    return params


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        # backfill any new defaults
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    cfg = dict(DEFAULT_CONFIG)
    params = discover_params(cfg["binary"], cfg["engine_arg"])
    for p in params:
        span = p["max"] - p["min"]
        p["C"] = max(1, round(span / 12.0))  # perturbation scale in param units
    cfg["params"] = params
    atomic_write_json(CONFIG_PATH, cfg)
    log(f"wrote default config with {len(params)} params -> {CONFIG_PATH}")
    return cfg


# ── cutechess match runner ──────────────────────────────────────────────────

SCORE_RE = re.compile(r"Score of .+ vs .+: (\d+) - (\d+) - (\d+)")


def run_match(cfg, opts_a, opts_b, name_a, name_b, games, st):
    """Play `games` games (rounds=games/2, -repeat) of A vs B. Returns
    (wins_a, losses_a, draws) or None if interrupted."""
    rounds = max(1, games // 2)
    bin_ = cfg["binary"]
    arg = cfg["engine_arg"]

    def eng(name, opts):
        spec = ["-engine", f"cmd={bin_}", f"name={name}", f"arg={arg}", "proto=uci"]
        for k, v in opts.items():
            spec.append(f"option.{k}={v}")
        return spec

    cmd = [cfg["cutechess"]]
    cmd += eng(name_a, opts_a)
    cmd += eng(name_b, opts_b)
    cmd += [
        # timemargin is essential: with cutechess' default margin of 0, the fixed
        # per-move budget is a HARD deadline and any scheduling overshoot under
        # concurrency forfeits the game on time -> the whole signal becomes
        # time-forfeit noise. A generous margin keeps games decided by play.
        "-each", f"st={st}", f"timemargin={cfg['timemargin']}",
        "-rounds", str(rounds), "-games", "2", "-repeat",
        "-openings", f"file={cfg['book']}", "format=pgn", "order=random",
        f"plies={cfg['book_plies']}",
        "-resign", f"movecount={cfg['resign_movecount']}",
        f"score={cfg['resign_score']}", "twosided=true",
        "-draw", f"movenumber={cfg['draw_movenumber']}",
        f"movecount={cfg['draw_movecount']}", f"score={cfg['draw_score']}",
        "-concurrency", str(cfg["concurrency"]),
        "-recover",
        "-ratinginterval", "0",
    ]

    env = dict(os.environ, LC_ALL="C")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, start_new_session=True,
    )
    last = None
    interrupted = False
    for line in proc.stdout:
        m = SCORE_RE.search(line)
        if m:
            last = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if STOP or os.path.exists(STOP_FILE):
            interrupted = True
            break
    if interrupted and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        return None
    proc.wait()
    return last


# ── SPSA core ───────────────────────────────────────────────────────────────

def init_state(cfg):
    return {
        "k": 0,
        "theta": {p["name"]: float(p["default"]) for p in cfg["params"]},
        "theta_sum": {p["name"]: 0.0 for p in cfg["params"]},
        "sum_count": 0,
        "games_played": 0,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_state(cfg):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            st = json.load(f)
        # make sure any newly-added params / fields are present
        st.setdefault("theta_sum", {p["name"]: 0.0 for p in cfg["params"]})
        st.setdefault("sum_count", 0)
        for p in cfg["params"]:
            st["theta"].setdefault(p["name"], float(p["default"]))
            st["theta_sum"].setdefault(p["name"], 0.0)
        return st
    return init_state(cfg)


def rounded_theta(cfg, theta):
    out = {}
    for p in cfg["params"]:
        out[p["name"]] = clampi(theta[p["name"]], p["min"], p["max"])
    return out


def write_params_txt(cfg, theta):
    lines = ["# current SPSA estimate (rounded, clamped)"]
    rt = rounded_theta(cfg, theta)
    for p in cfg["params"]:
        n = p["name"]
        lines.append(f"{n} = {rt[n]}   (default {p['default']}, range {p['min']}..{p['max']})")
    with open(PARAMS_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")


# deterministic per-iteration +-1 perturbation without external RNG state
def rademacher(k, idx):
    h = (k * 2654435761 + idx * 40503 + 12345) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0x5bd1e995) & 0xFFFFFFFF
    h ^= (h >> 15)
    return 1 if (h & 1) else -1


def tune(cfg):
    state = load_state(cfg)
    params = cfg["params"]
    cmap = {p["name"]: p for p in params}
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    log(f"=== SPSA tune start (k={state['k']}, {len(params)} params, "
        f"batch={cfg['batch_games']} st={cfg['st']} cc={cfg['concurrency']}) ===")
    log("params: " + ", ".join(f"{p['name']}[C={p['C']}]" for p in params))

    best = None
    if os.path.exists(BEST_PATH):
        with open(BEST_PATH) as f:
            best = json.load(f)

    while True:
        if STOP or os.path.exists(STOP_FILE):
            log("stop requested -> saving and exiting")
            break
        if cfg["max_iters"] and state["k"] >= cfg["max_iters"]:
            log(f"reached max_iters={cfg['max_iters']}")
            break

        k = state["k"] + 1
        c_k = cfg["c0"] / (k ** cfg["gamma"])
        a_k = cfg["a0"] * ((cfg["A"] + 1.0) / (cfg["A"] + k)) ** cfg["alpha"]

        opts_plus, opts_minus, deltas = {}, {}, {}
        for idx, p in enumerate(params):
            n = p["name"]
            delta = rademacher(k, idx)
            pert = max(1, round(p["C"] * c_k))
            deltas[n] = delta
            opts_plus[n] = clampi(state["theta"][n] + pert * delta, p["min"], p["max"])
            opts_minus[n] = clampi(state["theta"][n] - pert * delta, p["min"], p["max"])

        res = run_match(cfg, opts_plus, opts_minus, "plus", "minus",
                        cfg["batch_games"], cfg["st"])
        if res is None:
            log("match interrupted -> discarding this iteration (no theta change)")
            break
        w, l, d = res
        n_games = w + l + d
        if n_games == 0:
            log("warning: 0 games returned; skipping update")
            continue
        s = (w + 0.5 * d) / n_games        # score of theta+ in [0,1]
        signal_val = 2.0 * s - 1.0          # in [-1, 1]

        for p in params:
            n = p["name"]
            step = a_k * p["C"] * signal_val * deltas[n]
            state["theta"][n] = max(p["min"], min(p["max"], state["theta"][n] + step))

        state["k"] = k
        state["games_played"] += n_games
        # Polyak (iterate) averaging after burn-in: the tail-average of the SPSA
        # walk is a much lower-variance estimate than any single iterate.
        if k > cfg["avg_burn_in"]:
            for p in params:
                state["theta_sum"][p["name"]] += state["theta"][p["name"]]
            state["sum_count"] += 1
        atomic_write_json(STATE_PATH, state)
        write_params_txt(cfg, state["theta"])

        log(f"iter {k:5d}  +{w} -{l} ={d}  s={s:.3f}  a_k={a_k:.4f} c_k={c_k:.4f}  "
            f"games_total={state['games_played']}")

        # periodic checkpoint vs defaults
        if cfg["checkpoint_interval"] and k % cfg["checkpoint_interval"] == 0:
            best = checkpoint(cfg, state, best)
            if STOP or os.path.exists(STOP_FILE):
                log("stop requested during checkpoint -> exiting")
                break

    atomic_write_json(STATE_PATH, state)
    write_params_txt(cfg, state["theta"])
    log(f"=== tune stopped at k={state['k']}, {state['games_played']} games. "
        f"state -> {STATE_PATH}, params -> {PARAMS_TXT} ===")


def checkpoint(cfg, state, best):
    rt = rounded_theta(cfg, state["theta"])
    base = {p["name"]: p["default"] for p in cfg["params"]}
    log(f"checkpoint @k={state['k']}: current vs defaults, {cfg['checkpoint_games']} games...")
    res = run_match(cfg, rt, base, "tuned", "base",
                    cfg["checkpoint_games"], cfg["st"])
    if res is None:
        log("checkpoint interrupted")
        return best
    w, l, d = res
    n = w + l + d
    if n == 0:
        return best
    score = (w + 0.5 * d) / n
    elo, margin = elo_with_ci(w, l, d)
    log(f"checkpoint result: tuned +{w} -{l} ={d}  score={score:.3f}  "
        f"elo={elo:+.1f} +-{margin:.1f}")
    cur = {"k": state["k"], "theta": rt, "elo": elo, "margin": margin,
           "score": score, "w": w, "l": l, "d": d}
    if best is None or elo > best.get("elo", -1e9):
        atomic_write_json(BEST_PATH, cur)
        log(f"new best @k={state['k']} elo={elo:+.1f} -> {BEST_PATH}")
        return cur
    return best


# ── elo / SPRT ──────────────────────────────────────────────────────────────

def elo_to_score(e):
    return 1.0 / (1.0 + 10.0 ** (-e / 400.0))


def elo_with_ci(w, l, d):
    n = w + l + d
    if n == 0:
        return 0.0, 0.0
    score = (w + 0.5 * d) / n
    if score <= 0:
        return -800.0, 0.0
    if score >= 1:
        return 800.0, 0.0
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * score ** 2) / n
    std = math.sqrt(var / n) if var > 0 else 1e-9

    def s2e(s):
        s = min(1 - 1e-9, max(1e-9, s))
        return -400.0 * math.log10(1 / s - 1)

    elo = s2e(score)
    lo = s2e(max(1e-9, score - 1.96 * std))
    hi = s2e(min(1 - 1e-9, score + 1.96 * std))
    return elo, (hi - lo) / 2.0


def gsprt_llr(w, l, d, elo0, elo1):
    n = w + l + d
    if n == 0:
        return 0.0
    score = (w + 0.5 * d) / n
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * score ** 2) / (n * n)
    if var <= 0:
        var = 1e-9
    t0, t1 = elo_to_score(elo0), elo_to_score(elo1)
    return (t1 - t0) * (2 * score - t0 - t1) / (2 * var)


def sprt(cfg, source):
    theta = resolve_theta(cfg, source)
    rt = rounded_theta(cfg, theta)
    base = {p["name"]: p["default"] for p in cfg["params"]}
    elo0, elo1 = cfg["sprt_elo0"], cfg["sprt_elo1"]
    a, b = cfg["sprt_alpha"], cfg["sprt_beta"]
    upper = math.log((1 - b) / a)
    lower = math.log(b / (1 - a))
    log(f"=== SPRT tuned vs base  H0:{elo0} H1:{elo1} elo, bounds [{lower:.2f},{upper:.2f}] "
        f"st={cfg['sprt_st']} ===")
    log("tuned params: " + ", ".join(f"{k}={v}" for k, v in rt.items()))

    W = L = D = 0
    while W + L + D < cfg["sprt_max_games"]:
        if STOP or os.path.exists(STOP_FILE):
            log("stop requested -> ending SPRT")
            break
        res = run_match(cfg, rt, base, "tuned", "base",
                        cfg["sprt_batch"], cfg["sprt_st"])
        if res is None:
            break
        w, l, d = res
        W += w; L += l; D += d
        llr = gsprt_llr(W, L, D, elo0, elo1)
        elo, margin = elo_with_ci(W, L, D)
        log(f"SPRT  +{W} -{L} ={D}  ({W+L+D} games)  elo={elo:+.1f} +-{margin:.1f}  "
            f"LLR={llr:.2f} [{lower:.2f},{upper:.2f}]")
        verdict = None
        if llr >= upper:
            verdict = "H1 ACCEPTED: tuned is better (conclusive gain)"
        elif llr <= lower:
            verdict = "H0 ACCEPTED: no gain over defaults"
        if verdict:
            log("=== " + verdict + " ===")
            atomic_write_json(os.path.join(RUN, "sprt_result.json"),
                              {"W": W, "L": L, "D": D, "elo": elo, "margin": margin,
                               "llr": llr, "verdict": verdict, "theta": rt})
            return verdict
    elo, margin = elo_with_ci(W, L, D)
    log(f"=== SPRT inconclusive at cap: elo={elo:+.1f} +-{margin:.1f} "
        f"(+{W} -{L} ={D}) ===")
    atomic_write_json(os.path.join(RUN, "sprt_result.json"),
                      {"W": W, "L": L, "D": D, "elo": elo, "margin": margin,
                       "verdict": "inconclusive", "theta": rt})
    return "inconclusive"


# ── apply / status ──────────────────────────────────────────────────────────

def resolve_theta(cfg, source):
    if source == "best" and os.path.exists(BEST_PATH):
        with open(BEST_PATH) as f:
            return {k: float(v) for k, v in json.load(f)["theta"].items()}
    if source == "avg" and os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            st = json.load(f)
        if st.get("sum_count", 0) > 0:
            return {n: st["theta_sum"][n] / st["sum_count"] for n in st["theta_sum"]}
        return st["theta"]
    if source == "current" and os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)["theta"]
    if os.path.isfile(source):
        with open(source) as f:
            j = json.load(f)
            t = j.get("theta", j)
            return {k: float(v) for k, v in t.items()}
    # default: current if present else best
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)["theta"]
    raise SystemExit(f"could not resolve theta source '{source}'")


def apply_to_source(cfg, source):
    theta = resolve_theta(cfg, source)
    rt = rounded_theta(cfg, theta)
    path = os.path.join(ROOT, "src/search.rs")
    with open(path) as f:
        src = f.read()
    changed = []
    for p in cfg["params"]:
        n = p["name"]
        # Tunable::new("Name", DEFAULT, min, max)
        pat = re.compile(
            r'(Tunable::new\(\s*"' + re.escape(n) + r'"\s*,\s*)(-?\d+)(\s*,)')
        new_val = rt[n]

        def repl(m, nv=new_val):
            return m.group(1) + str(nv) + m.group(3)

        src, n_sub = pat.subn(repl, src)
        if n_sub:
            changed.append((n, rt[n]))
    with open(path, "w") as f:
        f.write(src)
    log(f"applied {len(changed)} params to {path}:")
    for n, v in changed:
        log(f"  {n} -> {v}")


def status(cfg):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            st = json.load(f)
        print(f"iterations: {st['k']}   games: {st['games_played']}   "
              f"started: {st.get('started')}")
        print("current (rounded):")
        rt = rounded_theta(cfg, st["theta"])
        avg = None
        if st.get("sum_count", 0) > 0:
            avg = {n: st["theta_sum"][n] / st["sum_count"] for n in st["theta_sum"]}
        hdr = "  param                  current   avg(Polyak)  default"
        print(hdr if avg else "current (rounded):")
        for p in cfg["params"]:
            n = p["name"]
            cur = rt[n]
            if avg is not None:
                av = clampi(avg[n], p["min"], p["max"])
                flag = "  <--" if av != p["default"] else ""
                print(f"  {n:22s} {cur:7d}   {av:9d}   {p['default']:7d}{flag}")
            else:
                flag = "" if cur == p["default"] else "  <-- changed"
                print(f"  {n:22s} {cur:7d}  (default {p['default']}){flag}")
        if avg:
            print(f"  (Polyak avg over {st['sum_count']} post-burn-in iters)")
    else:
        print("no state yet")
    if os.path.exists(BEST_PATH):
        with open(BEST_PATH) as f:
            b = json.load(f)
        print(f"\nbest checkpoint @k={b['k']}: elo={b['elo']:+.1f} +-{b['margin']:.1f}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RUN, exist_ok=True)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="tune",
                    choices=["tune", "sprt", "apply", "status"])
    ap.add_argument("--source", default="best",
                    help="theta source for sprt/apply: best|current|<json path>")
    args = ap.parse_args()

    cfg = load_config()

    if args.cmd == "tune":
        tune(cfg)
    elif args.cmd == "sprt":
        sprt(cfg, args.source)
    elif args.cmd == "apply":
        apply_to_source(cfg, args.source)
    elif args.cmd == "status":
        status(cfg)


if __name__ == "__main__":
    main()
