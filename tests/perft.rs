// Integration test for the engine's own movegen, independent of the main
// binary. We include the `engine_core` module directly via `#[path]` so the
// test doesn't depend on the crate having a lib target.

#[path = "../src/engine_core/mod.rs"]
mod engine_core;

use crate::engine_core::{
    count_legal_moves, ensure_init, gen_pseudo_legal, Board, ChessMove, PinInfo,
};
use smallvec::SmallVec;
use std::str::FromStr;
use std::time::Instant;

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if depth == 1 {
        return count_legal_moves(board);
    }
    let pin_info = PinInfo::for_board(board);
    let mut moves: SmallVec<[ChessMove; 64]> = SmallVec::new();
    gen_pseudo_legal(board, &mut moves);
    let mut count = 0u64;
    for mv in &moves {
        if !pin_info.move_is_legal(board, *mv) {
            continue;
        }
        let new_board = board.make_move_new(*mv);
        count += perft(&new_board, depth - 1);
    }
    count
}

const POSITIONS: &[(&str, u32, u64)] = &[
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        5,
        4_865_609,
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        5,
        193_690_690,
    ),
    (
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        5,
        674_624,
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4,
        2_103_487,
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        4,
        3_894_594,
    ),
];

fn perft_full(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let pin_info = PinInfo::for_board(board);
    let mut moves: SmallVec<[ChessMove; 64]> = SmallVec::new();
    gen_pseudo_legal(board, &mut moves);
    let mut count = 0u64;
    for mv in &moves {
        if !pin_info.move_is_legal(board, *mv) {
            continue;
        }
        let new_board = board.make_move_new(*mv);
        count += perft_full(&new_board, depth - 1);
    }
    count
}

/// Startpos perft(6) with `count_legal_moves` shortcut at depth 1
/// (this is the same algorithm `perft_suite` uses).
#[test]
#[ignore = "long-running perft depth 6 benchmark"]
fn perft_depth6_startpos_shortcut() {
    ensure_init();
    let _ = Board::default();

    let board = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        .expect("valid FEN");
    let start = Instant::now();
    let nodes = perft(&board, 6);
    let elapsed = start.elapsed().as_secs_f64();
    let nps_m = nodes as f64 / elapsed / 1e6;
    println!(
        "perft(6) shortcut: {} nodes in {:.3}s ({:.1} Mnps)",
        nodes, elapsed, nps_m
    );
    assert_eq!(nodes, 119_060_324);
}

/// Same position but with full move enumeration at every depth, so the
/// per-move `make_move_new` + `pin_info.move_is_legal` cost is exercised
/// at the leaves. Closer to what the actual search pays per move.
#[test]
#[ignore = "long-running perft depth 6 benchmark"]
fn perft_depth6_startpos_full() {
    ensure_init();
    let _ = Board::default();

    let board = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        .expect("valid FEN");
    let start = Instant::now();
    let nodes = perft_full(&board, 6);
    let elapsed = start.elapsed().as_secs_f64();
    let nps_m = nodes as f64 / elapsed / 1e6;
    println!(
        "perft(6) full enum: {} nodes in {:.3}s ({:.1} Mnps)",
        nodes, elapsed, nps_m
    );
    assert_eq!(nodes, 119_060_324);
}

#[test]
fn perft_suite() {
    // Force lazy table init (magic bitboards, etc.) BEFORE starting the timer
    // so the one-time init cost does not contaminate the NPS measurement.
    ensure_init();
    let _ = Board::default();

    let mut total_nodes: u64 = 0;
    let start = Instant::now();
    let mut mismatches: Vec<String> = Vec::new();
    for (fen, depth, expected) in POSITIONS {
        let board = Board::from_str(fen).expect("valid FEN");
        let nodes = perft(&board, *depth);
        if nodes != *expected {
            mismatches.push(format!(
                "  FEN `{}` d{} expected={} got={}",
                fen, depth, expected, nodes
            ));
        }
        total_nodes += nodes;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let nps = (total_nodes as f64 / elapsed) as u64;
    println!("PERFT_NPS: {}", nps);
    println!("PERFT_TOTAL_NODES: {}", total_nodes);
    println!("PERFT_ELAPSED_SECS: {:.3}", elapsed);
    if !mismatches.is_empty() {
        panic!("perft mismatches:\n{}", mismatches.join("\n"));
    }
}
