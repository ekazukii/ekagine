//! Syzygy tablebase probing (endgame perfection).
//!
//! The heavy lifting (WDL/DTZ decompression + symmetric combinatorial indexing)
//! is delegated to the vendored public-domain Fathom prober
//! (`deps/fathom/tbprobe.c`), wired via a thin FFI in this module once the C
//! source is present. What lives HERE — and what we own and validate — is:
//!   * the engine `Board` -> Fathom-argument conversion,
//!   * the safe Rust wrapper around the probe,
//!   * the WDL-agreement test oracle that checks the integration (and will also
//!     validate a future pure-Rust port against the same expected results).
//!
//! NOTE: until `deps/fathom/` + `build.rs` are added, only the conversion and
//! the oracle data below are compiled; the FFI/probe is wired in a follow-up.

use crate::engine_core::{Board, Color, Piece};

/// A position encoded the way Fathom expects it: per-colour and per-piece
/// bitboards plus the fifty-move clock, castling mask, en-passant square and the
/// side to move.
#[allow(dead_code)]
pub struct SyzygyArgs {
    pub white: u64,
    pub black: u64,
    pub kings: u64,
    pub queens: u64,
    pub rooks: u64,
    pub bishops: u64,
    pub knights: u64,
    pub pawns: u64,
    pub rule50: u32,
    pub castling: u8,
    pub ep: u32,
    pub white_to_move: bool,
}

/// Number of men on the board. Syzygy is probed only when this is `<=` the
/// largest loaded table (5 for the 3-4-5 set already in `tables/`).
#[allow(dead_code)]
#[inline]
pub fn man_count(board: &Board) -> u32 {
    board.combined().popcnt() as u32
}

/// Convert the engine board to Fathom's probe arguments. Pure data extraction —
/// no probing, no allocation; safe to call on the hot path. The caller still
/// gates probing on `man_count <= TB_LARGEST`, no castling rights, etc.
#[allow(dead_code)]
pub fn position_args(board: &Board) -> SyzygyArgs {
    SyzygyArgs {
        white: board.color_combined(Color::White).0,
        black: board.color_combined(Color::Black).0,
        kings: board.pieces(Piece::King).0,
        queens: board.pieces(Piece::Queen).0,
        rooks: board.pieces(Piece::Rook).0,
        bishops: board.pieces(Piece::Bishop).0,
        knights: board.pieces(Piece::Knight).0,
        pawns: board.pieces(Piece::Pawn).0,
        rule50: board.halfmove_clock() as u32,
        castling: board.castling_rights(),
        // Fathom uses 0 for "no en-passant" (a1 is never a legal ep target).
        ep: board.en_passant().map_or(0, |sq| sq.to_index() as u32),
        white_to_move: board.side_to_move() == Color::White,
    }
}

/// WDL validation oracle: `(FEN, expected WDL from the side-to-move's view)`.
/// `+2` = win, `0` = draw, `-2` = loss (cursed-win / blessed-loss would be `±1`,
/// but these positions are clean). Exercised by the agreement test once the
/// probe is wired, and reused as the ground truth for any future Rust port.
#[allow(dead_code)]
pub const WDL_ORACLE: &[(&str, i32)] = &[
    // KQ vs K — win.
    ("8/8/8/8/8/3k4/8/3KQ3 w - - 0 1", 2),
    // KR vs K — win.
    ("8/8/8/8/8/3k4/8/3KR3 w - - 0 1", 2),
    // KBN vs K — win (the hard mate).
    ("8/8/8/8/8/3k4/8/2BNK3 w - - 0 1", 2),
    // KNN vs K — draw (two knights cannot force mate).
    ("8/8/8/8/3k4/8/8/1N1NK3 w - - 0 1", 0),
    // KB vs K — draw (insufficient material).
    ("8/8/8/8/8/3k4/8/3KB3 w - - 0 1", 0),
    // K vs K — draw.
    ("8/8/8/8/8/3k4/8/3K4 w - - 0 1", 0),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn args_startpos() {
        let b = Board::default();
        let a = position_args(&b);
        assert_eq!(man_count(&b), 32);
        assert!(a.white_to_move);
        assert_eq!(a.ep, 0);
        assert_eq!(a.rule50, 0);
        // Colours partition the occupancy; piece bitboards partition it too.
        assert_eq!(a.white | a.black, b.combined().0);
        assert_eq!(a.white & a.black, 0);
        assert_eq!(
            a.kings | a.queens | a.rooks | a.bishops | a.knights | a.pawns,
            b.combined().0
        );
        assert_eq!(a.kings.count_ones(), 2);
        assert_eq!(a.pawns.count_ones(), 16);
        assert_ne!(a.castling, 0); // start position has all castling rights
    }

    #[test]
    fn oracle_fens_parse_and_are_small() {
        // The oracle FENs must be legal and within the 5-man tables we have.
        for (fen, _wdl) in WDL_ORACLE {
            let b = Board::from_str(fen).unwrap_or_else(|e| panic!("bad oracle FEN {fen}: {e}"));
            assert!(
                man_count(&b) <= 5,
                "oracle FEN {fen} has {} men (>5)",
                man_count(&b)
            );
        }
    }
}
