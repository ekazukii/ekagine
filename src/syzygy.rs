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
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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

// ─── FFI to the vendored Fathom prober (compiled by build.rs) ───────────────
extern "C" {
    fn ek_tb_init(path: *const c_char) -> c_int;
    #[allow(dead_code)]
    fn ek_tb_free();
    fn ek_tb_largest() -> c_uint;
    #[allow(clippy::too_many_arguments)]
    fn ek_tb_probe_wdl(
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        rule50: c_uint,
        castling: c_uint,
        ep: c_uint,
        turn: c_int,
    ) -> c_uint;
}

static TB_ENABLED: AtomicBool = AtomicBool::new(false);
static TB_LARGEST: AtomicU32 = AtomicU32::new(0);

/// Win/Draw/Loss from the side-to-move's perspective. Cursed-win / blessed-loss
/// are the fifty-move-rule edge cases (a win / loss that the rule draws).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Wdl {
    Loss,
    BlessedLoss,
    Draw,
    CursedWin,
    Win,
}

fn wdl_from_raw(r: c_uint) -> Option<Wdl> {
    match r {
        0 => Some(Wdl::Loss),
        1 => Some(Wdl::BlessedLoss),
        2 => Some(Wdl::Draw),
        3 => Some(Wdl::CursedWin),
        4 => Some(Wdl::Win),
        _ => None, // TB_RESULT_FAILED
    }
}

/// Load Syzygy tablebases from `path`. Returns true and enables probing if any
/// tables were found. Call once (single-threaded) before searching.
#[allow(dead_code)]
pub fn init(path: &str) -> bool {
    let cpath = match CString::new(path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let ok = unsafe { ek_tb_init(cpath.as_ptr()) != 0 };
    let largest = unsafe { ek_tb_largest() };
    let enabled = ok && largest > 0;
    TB_LARGEST.store(largest, Ordering::Relaxed);
    TB_ENABLED.store(enabled, Ordering::Relaxed);
    enabled
}

#[allow(dead_code)]
#[inline]
pub fn enabled() -> bool {
    TB_ENABLED.load(Ordering::Relaxed)
}

#[allow(dead_code)]
#[inline]
pub fn largest() -> u32 {
    TB_LARGEST.load(Ordering::Relaxed)
}

/// WDL probe for use in search. `None` if probing is disabled, the position has
/// too many men / castling rights / a non-zero fifty-move clock (Fathom rejects
/// the latter two), or the probe otherwise fails. Thread-safe.
#[allow(dead_code)]
pub fn probe_wdl(board: &Board) -> Option<Wdl> {
    if !enabled() || man_count(board) > largest() {
        return None;
    }
    let a = position_args(board);
    let r = unsafe {
        ek_tb_probe_wdl(
            a.white,
            a.black,
            a.kings,
            a.queens,
            a.rooks,
            a.bishops,
            a.knights,
            a.pawns,
            a.rule50,
            a.castling as c_uint,
            a.ep,
            a.white_to_move as c_int,
        )
    };
    wdl_from_raw(r)
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

    fn wdl_oracle_value(w: Wdl) -> i32 {
        match w {
            Wdl::Win => 2,
            Wdl::CursedWin => 1,
            Wdl::Draw => 0,
            Wdl::BlessedLoss => -1,
            Wdl::Loss => -2,
        }
    }

    /// Correctness gate: probe the oracle FENs against loaded tablebases and
    /// assert the WDL matches. Skips cleanly if no tablebases are present;
    /// individual positions whose table is missing are skipped, but a wrong WDL
    /// fails, and we require at least a few positions to actually validate (so a
    /// broken prober that returns only failures can't pass silently).
    #[test]
    fn wdl_agreement() {
        if !init("tables") {
            eprintln!("syzygy: no tablebases under tables/, skipping WDL agreement");
            return;
        }
        let mut checked = 0;
        for (fen, expected) in WDL_ORACLE {
            let b = Board::from_str(fen).expect("oracle FEN parses");
            match probe_wdl(&b) {
                Some(got) => {
                    assert_eq!(
                        wdl_oracle_value(got),
                        *expected,
                        "WDL mismatch for {fen}: got {got:?} = {}, expected {expected}",
                        wdl_oracle_value(got)
                    );
                    checked += 1;
                }
                None => eprintln!("syzygy: probe failed for {fen} (table missing?), skipping"),
            }
        }
        assert!(
            checked >= 3,
            "only {checked} oracle positions validated — prober likely broken"
        );
    }
}
