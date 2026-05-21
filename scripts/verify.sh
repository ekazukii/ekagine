#!/usr/bin/env bash
# Single acceptance gate for the chess-crate-removal migration.
# Exits 0 iff:
#   1. cargo build --release succeeds
#   2. cargo test --release (full suite) passes
#   3. perft NPS (max of 3) >= 95% of baseline AND PERFT_TOTAL_NODES matches
#      the hardcoded contract sum
#   4. UCI smoke test: `position startpos moves e2e4 e7e5; go depth 4`
#      produces a valid `bestmove <uci>` line
#   5. bestmove agreement >= 90% on 19 fixed positions at depth 8 vs baseline,
#      preceded by a sanity probe that the bestmove format is UCI long-algebraic
#   6. no `use chess` imports remain in src/
#   7. chess crate absent from the resolved dependency tree
#
# Order: quality gates first, completion gates last. This way running verify
# pre-migration shows the quality bar is reachable; only completion gates fail.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POSITIONS_FILE="$ROOT/tests/agreement_positions.fen"
BASELINE_FILE="$ROOT/tests/agreement_baseline.txt"
NPS_FILE="$ROOT/tests/perft_nps_baseline.txt"
DEPTH_FILE="$ROOT/tests/agreement_depth.txt"

# Contract values — hardcoded on purpose. PERFT_TOTAL_NODES is the sum of the
# 5 canonical perft counts in tests/perft.rs; if the test is rewritten to use
# a different movegen, this total MUST remain identical or the gate rejects.
EXPECTED_PERFT_TOTAL_NODES=205229004
NPS_THRESHOLD_PCT=95
AGREEMENT_THRESHOLD_PCT=90
UCI_BESTMOVE_RE='^bestmove [a-h][1-8][a-h][1-8][qrbn]?( |$)'
UCI_MOVE_RE='^[a-h][1-8][a-h][1-8][qrbn]?$'

for f in "$POSITIONS_FILE" "$BASELINE_FILE" "$NPS_FILE" "$DEPTH_FILE"; do
    if [ ! -f "$f" ]; then
        echo "FAIL: required baseline file missing: $f" >&2
        echo "  Run scripts/gen_baseline.sh on the pre-migration commit first." >&2
        exit 1
    fi
done

DEPTH=$(cat "$DEPTH_FILE")
BASELINE_NPS=$(cat "$NPS_FILE")

TIMEOUT_BIN=""
for cmd in timeout gtimeout; do
    if command -v "$cmd" >/dev/null 2>&1; then TIMEOUT_BIN="$cmd"; break; fi
done

# ----- [1/7] build -----
echo "==> [1/7] cargo build --release"
cargo build --release --quiet

BIN=""
for candidate in \
    "$ROOT/target/release/chess_engine" \
    "$ROOT/target/aarch64-apple-darwin/release/chess_engine" \
    "$ROOT/target/x86_64-apple-darwin/release/chess_engine"
do
    if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
done
if [ -z "$BIN" ]; then
    echo "FAIL: release binary not found under target/*/release/chess_engine" >&2
    exit 1
fi

# ----- [2/7] full test suite -----
echo "==> [2/7] cargo test --release (full suite)"
if ! cargo test --release --quiet > /tmp/verify_cargo_test.log 2>&1; then
    echo "FAIL: cargo test failed" >&2
    tail -100 /tmp/verify_cargo_test.log >&2
    exit 1
fi

# ----- [3/7] perft NPS + total nodes contract -----
echo "==> [3/7] perft NPS (max of 3 runs, threshold ${NPS_THRESHOLD_PCT}%) + total nodes contract"
# Warm-up runs (discarded) to stabilize wallclock measurements — CPU thermal/clock
# state matters a lot for short perft on laptops. 3 warmups ≈ 1.5s, gets the CPU
# to sustained clock.
for i in 1 2 3; do
    cargo test --release --test perft -- --nocapture > /dev/null 2>&1 || true
done
MAX_NPS=0
for i in 1 2 3; do
    OUT=$(cargo test --release --test perft -- --nocapture 2>&1) || {
        echo "FAIL: perft run $i failed" >&2
        echo "$OUT" | tail -40 >&2
        exit 1
    }
    NPS=$(echo "$OUT" | grep "^PERFT_NPS:" | awk '{print $2}')
    TOTAL=$(echo "$OUT" | grep "^PERFT_TOTAL_NODES:" | awk '{print $2}')
    if [ -z "$NPS" ] || [ -z "$TOTAL" ]; then
        echo "FAIL: perft run $i missing PERFT_NPS or PERFT_TOTAL_NODES" >&2
        echo "$OUT" | tail -40 >&2
        exit 1
    fi
    if [ "$TOTAL" != "$EXPECTED_PERFT_TOTAL_NODES" ]; then
        echo "FAIL: PERFT_TOTAL_NODES mismatch run $i: expected $EXPECTED_PERFT_TOTAL_NODES, got $TOTAL" >&2
        echo "  This means the perft test is no longer producing the canonical sum." >&2
        echo "  The 5 (FEN, depth, expected_count) tuples must each pass exactly." >&2
        exit 1
    fi
    if [ "$NPS" -gt "$MAX_NPS" ]; then MAX_NPS=$NPS; fi
done
echo "  max NPS over 3 runs: $MAX_NPS  (baseline $BASELINE_NPS)"
NPS_OK=$(awk -v cur="$MAX_NPS" -v base="$BASELINE_NPS" -v pct="$NPS_THRESHOLD_PCT" \
    'BEGIN { print (cur * 100 >= base * pct) ? "1" : "0" }')
if [ "$NPS_OK" != "1" ]; then
    MIN=$(awk -v base="$BASELINE_NPS" -v pct="$NPS_THRESHOLD_PCT" \
        'BEGIN { printf "%.0f", base * pct / 100 }')
    echo "FAIL: perft NPS $MAX_NPS < ${NPS_THRESHOLD_PCT}% of baseline ($MIN required)" >&2
    exit 1
fi

# ----- [4/7] UCI smoke test -----
echo "==> [4/7] UCI smoke test (position startpos moves e2e4 e7e5; go depth 4)"
UCI_INPUT='uci
position startpos moves e2e4 e7e5
go depth 4
quit
'
if [ -n "$TIMEOUT_BIN" ]; then
    UCI_OUT=$(printf '%s' "$UCI_INPUT" | $TIMEOUT_BIN 30 "$BIN" --uci 2>&1 || true)
else
    UCI_OUT=$(printf '%s' "$UCI_INPUT" | "$BIN" --uci 2>&1 || true)
fi
if ! echo "$UCI_OUT" | grep -qE "$UCI_BESTMOVE_RE"; then
    echo "FAIL: UCI smoke test did not produce a valid bestmove line" >&2
    echo "  expected pattern: $UCI_BESTMOVE_RE" >&2
    echo "  --- engine output (first 30 lines) ---" >&2
    echo "$UCI_OUT" | head -30 >&2
    exit 1
fi

# ----- [5/7] bestmove agreement -----
echo "==> [5/7] bestmove agreement at depth $DEPTH (threshold ${AGREEMENT_THRESHOLD_PCT}%)"

# Sanity probe: if the bestmove output format diverged, fail fast with a clear
# message instead of producing 0/N agreement.
FIRST_FEN=$(grep -v '^[[:space:]]*$' "$POSITIONS_FILE" | grep -v '^#' | head -n1)
FIRST_OUT=$("$BIN" --best "$FIRST_FEN" "$DEPTH" 2>&1 || true)
FIRST_MOVE=$(echo "$FIRST_OUT" | grep -E "^bestmove " | awk '{print $2}' | head -n1)
if ! echo "$FIRST_MOVE" | grep -qE "$UCI_MOVE_RE"; then
    echo "FAIL: bestmove format unexpected — got '$FIRST_MOVE'" >&2
    echo "  expected UCI long-algebraic: e2e4 (normal) or e7e8q (promotion, lowercase suffix in {q,r,b,n})" >&2
    echo "  ChessMove Display must produce this format; anything else breaks the agreement check." >&2
    echo "  --- output for first FEN ($FIRST_FEN) ---" >&2
    echo "$FIRST_OUT" | head -10 >&2
    exit 1
fi

TOTAL=0
MATCH=0
MISMATCHES=""
while IFS='|' read -r FEN EXPECTED_MOVE || [ -n "$FEN" ]; do
    [ -z "$FEN" ] && continue
    case "$FEN" in \#*) continue ;; esac
    TOTAL=$((TOTAL + 1))
    OUT=$("$BIN" --best "$FEN" "$DEPTH" 2>&1 || true)
    ACTUAL_MOVE=$(echo "$OUT" | grep -E "^bestmove " | awk '{print $2}' | head -n1)
    if [ -z "$ACTUAL_MOVE" ]; then
        MISMATCHES="${MISMATCHES}    NO_BESTMOVE | $FEN (expected $EXPECTED_MOVE)\n"
        continue
    fi
    if [ "$ACTUAL_MOVE" = "$EXPECTED_MOVE" ]; then
        MATCH=$((MATCH + 1))
    else
        MISMATCHES="${MISMATCHES}    $ACTUAL_MOVE != $EXPECTED_MOVE | $FEN\n"
    fi
done < "$BASELINE_FILE"

PCT=$(awk -v m="$MATCH" -v t="$TOTAL" 'BEGIN { if (t == 0) print 0; else printf "%.1f", m * 100 / t }')
echo "  agreement = $MATCH/$TOTAL ($PCT%)"
AGREE_OK=$(awk -v m="$MATCH" -v t="$TOTAL" -v pct="$AGREEMENT_THRESHOLD_PCT" \
    'BEGIN { print (m * 100 >= t * pct) ? "1" : "0" }')
if [ "$AGREE_OK" != "1" ]; then
    echo "FAIL: bestmove agreement below ${AGREEMENT_THRESHOLD_PCT}%" >&2
    printf "%b" "$MISMATCHES" >&2
    exit 1
fi

# ----- [6/7] grep src/ for chess imports (completion gate, fast) -----
echo "==> [6/7] no \`use chess\` imports in src/"
if grep -rn "use chess" "$ROOT/src/" > /dev/null 2>&1; then
    echo "FAIL: \`use chess\` still present:" >&2
    grep -rn "use chess" "$ROOT/src/" >&2
    exit 1
fi

# ----- [7/7] cargo tree (definitive crate presence check) -----
echo "==> [7/7] chess crate absent from resolved dependency tree"
TREE_OUT=$(cargo tree --quiet 2>&1) || {
    echo "FAIL: cargo tree failed" >&2
    echo "$TREE_OUT" >&2
    exit 1
}
if echo "$TREE_OUT" | grep -qE '(^|[├└│ ─])chess v[0-9]'; then
    echo "FAIL: chess crate still in dependency tree:" >&2
    echo "$TREE_OUT" | grep -E '(^|[├└│ ─])chess v[0-9]' >&2
    exit 1
fi

echo ""
echo "ALL CHECKS PASSED"
