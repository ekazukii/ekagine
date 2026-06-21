# SPSA tuning of the Ekagine search constants

Tunes the UCI `Tunable` params (`src/search.rs` → `TUNABLES`) by
simultaneous-perturbation self-play under cutechess-cli. All driving logic lives
in `spsa.py`; config in `config.json`; runtime state in `run/`.

## How it works
Each iteration perturbs **all** params at once by ±`C_i·c_k`, builds two option
sets (`theta+`, `theta-`), plays a small self-play batch between them, and nudges
the running estimate `theta` along the measured gradient. `theta` is written to
`run/state.json` **after every iteration** (atomic rename), so progress is never
lost. Every `checkpoint_interval` iterations it plays the current rounded `theta`
vs the engine defaults and snapshots the better one to `run/best.json`.

## Run / monitor
```bash
python3 tuning/spsa.py tune        # starts, or resumes from run/state.json
python3 tuning/spsa.py status      # current estimate + best checkpoint
tail -f tuning/run/log.txt         # live progress
```

## Interrupt (keeps all progress)
Any of these stops cleanly after the current match; completed iterations are kept
in `run/state.json`, and re-running `tune` resumes exactly where it left off:
```bash
touch tuning/run/STOP              # graceful: checked before each match
# or  kill <pid>  (SIGINT/SIGTERM handled the same way)
```
An interrupted, unfinished match is discarded — never a partial update.

## Get the result
```bash
python3 tuning/spsa.py sprt --source best     # validate best-so-far vs defaults (GSPRT)
python3 tuning/spsa.py apply --source best    # write tuned defaults into src/search.rs
# --source current  uses the latest raw estimate; --source <path.json> a snapshot
```

`run/` artifacts: `state.json` (resume point), `best.json` (best validated set),
`current_params.txt` (human-readable), `sprt_result.json` (final verdict),
`log.txt`.
