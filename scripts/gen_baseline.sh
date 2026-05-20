#!/usr/bin/env bash
# Capture migration baseline: perft NPS + bestmove per FEN.
# Run ONCE on the pre-migration commit. Resulting tests/*_baseline.txt
# files MUST be committed and treated as immutable during /goal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEPTH=8
POSITIONS_FILE="$ROOT/tests/agreement_positions.fen"
BASELINE_FILE="$ROOT/tests/agreement_baseline.txt"
NPS_FILE="$ROOT/tests/perft_nps_baseline.txt"
DEPTH_FILE="$ROOT/tests/agreement_depth.txt"

if [ ! -f "$POSITIONS_FILE" ]; then
    echo "ERROR: $POSITIONS_FILE not found" >&2
    exit 1
fi

echo "==> Building release binary..."
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
    echo "ERROR: release binary not found under target/*/release/chess_engine" >&2
    exit 1
fi
echo "  binary: $BIN"

echo "==> Warming up perft (3 discarded runs to ramp CPU clock)..."
for i in 1 2 3; do
    cargo test --release --test perft -- --nocapture > /dev/null 2>&1 || true
done

echo "==> Running perft suite 3x to capture NPS baseline (max of 3)..."
MAX_NPS=0
for i in 1 2 3; do
    PERFT_OUT="$(cargo test --release --test perft -- --nocapture 2>&1)"
    if ! echo "$PERFT_OUT" | grep -q "^PERFT_NPS:"; then
        echo "ERROR: perft test failed or did not print PERFT_NPS in run $i" >&2
        echo "$PERFT_OUT" >&2
        exit 1
    fi
    NPS=$(echo "$PERFT_OUT" | grep "^PERFT_NPS:" | awk '{print $2}')
    echo "  run $i: $NPS"
    if [ "$NPS" -gt "$MAX_NPS" ]; then MAX_NPS=$NPS; fi
done
echo "$MAX_NPS" > "$NPS_FILE"
echo "  baseline NPS (max of 3) = $MAX_NPS  (written to tests/perft_nps_baseline.txt)"

echo "==> Capturing bestmove at depth $DEPTH for each FEN..."
echo "$DEPTH" > "$DEPTH_FILE"
: > "$BASELINE_FILE"

LINE_NO=0
while IFS= read -r FEN || [ -n "$FEN" ]; do
    LINE_NO=$((LINE_NO + 1))
    [ -z "$FEN" ] && continue
    case "$FEN" in \#*) continue ;; esac

    OUT=$("$BIN" --best "$FEN" "$DEPTH" 2>&1 || true)
    BESTMOVE=$(echo "$OUT" | grep -E "^bestmove " | awk '{print $2}' | head -n1)
    if [ -z "$BESTMOVE" ]; then
        echo "ERROR: no bestmove for line $LINE_NO: $FEN" >&2
        echo "$OUT" >&2
        exit 1
    fi
    printf '%s|%s\n' "$FEN" "$BESTMOVE" >> "$BASELINE_FILE"
    printf '  [%2d] %s  <-  %s\n' "$LINE_NO" "$BESTMOVE" "$FEN"
done < "$POSITIONS_FILE"

COUNT=$(wc -l < "$BASELINE_FILE" | tr -d ' ')
echo ""
echo "==> Baseline captured successfully:"
echo "  tests/perft_nps_baseline.txt  ($MAX_NPS nodes/sec)"
echo "  tests/agreement_baseline.txt  ($COUNT positions @ depth $DEPTH)"
echo "  tests/agreement_depth.txt     ($DEPTH)"
