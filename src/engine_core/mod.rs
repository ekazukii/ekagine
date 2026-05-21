//! Self-contained chess primitives + pseudo-legal move generator.
//!
//! Replaces the previous `chess`-crate wrapper. `MoveGen::new_legal` returns
//! *pseudo-legal* moves — callers must filter illegal moves via
//! `Board::is_position_legal()` after `make_move_new`, or use
//! `PinInfo::move_is_legal` for a pre-make legality test.

#![allow(dead_code, unused_imports)]

mod attacks;
mod board;
mod movegen;
mod types;
mod zobrist;

pub use attacks::{
    ensure_init, get_adjacent_files, get_bishop_moves, get_file, get_king_moves, get_knight_moves,
    get_rook_moves,
};
pub use board::Board;
pub use movegen::{
    any_legal_move, count_legal_moves, for_each_capture_pseudo_legal, for_each_pseudo_legal,
    for_each_quiet_pseudo_legal, gen_pseudo_legal, MoveGen, PinInfo,
};
pub use types::{BitBoard, BoardStatus, ChessMove, Color, File, Piece, Rank, Square, EMPTY};
