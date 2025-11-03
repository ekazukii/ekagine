# Ekagine Chess Engine

Ekagine is a Rust chess engine that speaks the UCI protocol and focuses on
strong tactical play powered by NNUE evaluation and a modern alpha-beta search
stack. The code base is organized for experimentation and includes the same
core techniques used by contemporary tournament engines.

If you already have a lichess account, you can challenge it online at:
<https://lichess.org/@/ekagine>.

## Features
- UCI-compliant command loop with option negotiation (`Threads`) and
  self-identification for GUI integration.
- Iterative deepening with aspiration windows, principal-variation tracking,
  and info-string logging hooks for debugging.
- Negamax alpha-beta search featuring:
  - Transposition table with aging and bound-aware probing.
  - Killer moves, history, and correction-history heuristics for quiet move
    ordering.
  - Late move reductions, late move pruning, static exchange pruning, and
    futility variants (regular/reverse) to trim the tree.
  - Null-move pruning with adaptive depth reduction and check/passed-pawn
    extensions.
- Correction history system that captures evaluation deltas to suppress
  repeatedly over-valued quiet moves.
- NNUE incremental evaluation pipeline (driven by `net.bin`) with stack-based
  state push/pop so the network is coherently updated through the search.
- Incremental move generator with phased ordering (TT move, MVV-LVA captures,
  killer quiets, high-scoring quiets, remaining captures/quiets).
- Time-plan manager tuned for practical time controls (movetime, wtime/btime,
  increments, infinite, ponder) that tracks previous think time.
- Testing harness with unit coverage for NNUE deltas and move ordering helpers.

## Requirements
- Rust toolchain (stable `rustc`/`cargo`). Install from <https://rustup.rs/> if
  needed.
- The bundled `net.bin` holds the pretrained NNUE weights used during search.
  Replace it with another network of the same architecture if you want to
  experiment with different training data.

## Building
```bash
cargo build --release
```

Artifacts will appear under `target/release/`. The engine binary is named
`chess_engine` by default.

## Running
To run the engine in a UCI shell:
```bash
cargo run --release
```

Then enter standard UCI commands such as `uci`, `isready`, `position`, and
`go`. Connect the binary to any UCI-compatible GUI (e.g. CuteChess, Arena,
Banksia) to play games.

## Testing
```bash
cargo test
```

This executes the included unit tests covering NNUE incremental updates and
search helpers. Add your own tests as you expand the engine.

Continuous testing is handled on a custom SPRT infrastructure at
<https://ssprt.ekazuki.fr> (read-only access). Each pull request triggers a
GitHub Action that dispatches games to that cluster automatically.
