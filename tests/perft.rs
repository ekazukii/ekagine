use chess::{Board, MoveGen};
use std::str::FromStr;
use std::time::Instant;

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = MoveGen::new_legal(board);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut count = 0u64;
    for mv in moves {
        let new_board = board.make_move_new(mv);
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

#[test]
fn perft_suite() {
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
