use std::path::PathBuf;

use chess::{Board, ChessMove, Color, File, Piece, Rank, Square};
use shakmaty::fen::Fen;
use shakmaty::CastlingMode;
use shakmaty::Chess as SPosition;
use shakmaty::Move as SMove;
use shakmaty::Role;
use shakmaty::Square as SSquare;
use shakmaty_syzygy::{Tablebase as Syzygy, Wdl};

type SyzygyTB = Syzygy<SPosition>;

/// Syzygy WDL codes: 2=win, 1=win (cursed), 0=draw, -1=loss (blessed), -2=loss.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TBWdl {
    Loss = -2,
    BlessedLoss = -1,
    Draw = 0,
    CursedWin = 1,
    Win = 2,
}

impl TBWdl {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -2 => Some(TBWdl::Loss),
            -1 => Some(TBWdl::BlessedLoss),
            0 => Some(TBWdl::Draw),
            1 => Some(TBWdl::CursedWin),
            2 => Some(TBWdl::Win),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TBProbe {
    pub wdl: TBWdl,
    /// Distance-to-zeroing move in plies (Syzygy DTZ). Optional because WDL-only tables
    /// exist and because this stub implementation does not compute DTZ yet.
    pub dtz: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct TBRootProbe {
    pub wdl: TBWdl,
    pub dtz: Option<i32>,
    pub best_move: Option<ChessMove>,
}

/// Minimal, dependency-less placeholder for Syzygy probing.
///
/// The loader keeps configuration and chosen search limits, but currently does not parse the
/// table files. The probe methods return `None` until the on-disk table parsing code is
/// implemented. The surrounding engine plumbing is wired so that a future drop-in of the actual
/// Syzygy reader can work without further changes in the search code.
#[derive(Debug)]
pub struct Tablebase {
    paths: Vec<PathBuf>,
    probe_limit: usize,
    use_rule50: bool,
    inner: Option<SyzygyTB>,
}

impl Tablebase {
    pub fn new(paths: Vec<PathBuf>, probe_limit: usize, use_rule50: bool) -> Self {
        let mut tb = Syzygy::new();
        for p in &paths {
            let _ = tb.add_directory(p);
        }

        Self {
            paths,
            probe_limit,
            use_rule50,
            inner: Some(tb),
        }
    }

    pub fn probe_limit(&self) -> usize {
        self.probe_limit
    }

    pub fn use_rule50(&self) -> bool {
        self.use_rule50
    }

    pub fn has_paths(&self) -> bool {
        !self.paths.is_empty()
    }

    pub fn path_count(&self) -> usize {
        self.paths.len()
    }

    /// Probe WDL/DTZ for an interior node. Returns `None` if no table is available or probing is
    /// not supported yet.
    pub fn probe_wdl(&self, board: &Board, _halfmove_clock: u32) -> Option<TBProbe> {
        let tb = self.inner.as_ref()?;
        let pos = board_to_syzygy(board)?;
        let wdl = tb.probe_wdl(&pos).ok()?;
        Some(TBProbe {
            wdl: map_wdl(wdl),
            dtz: None,
        })
    }

    /// Probe WDL/DTZ and best move at the root. Returns `None` if no table is available or
    /// probing is not supported yet.
    pub fn probe_root(&self, board: &Board, _halfmove_clock: u32) -> Option<TBRootProbe> {
        let tb = self.inner.as_ref()?;
        let pos = board_to_syzygy(board)?;

        let wdl = tb.probe_wdl(&pos).ok()?;

        let best = tb
            .best_move(&pos)
            .ok()
            .and_then(|opt| opt)
            .and_then(|(m, _)| syzygy_move_to_chess_move(board, m));

        Some(TBRootProbe {
            wdl: map_wdl(wdl),
            dtz: None,
            best_move: best,
        })
    }
}

/// Map Syzygy WDL to an engine-centric score, relative to the side to move.
pub fn wdl_to_score(wdl: TBWdl, mover: Color, ply_from_root: i32, mate_value: i32) -> i32 {
    let base = match wdl {
        TBWdl::Win => mate_value - ply_from_root,
        TBWdl::CursedWin => 10_000, // Favorable but not forced mate under 50-move rule
        TBWdl::Draw => 0,
        TBWdl::BlessedLoss => -10_000,
        TBWdl::Loss => -mate_value + ply_from_root,
    };
    // Scores are already from the perspective of the side to move.
    let _ = mover; // reserved for potential future color-based mapping
    base
}

fn map_wdl(wdl: shakmaty_syzygy::AmbiguousWdl) -> TBWdl {
    use shakmaty_syzygy::AmbiguousWdl::*;
    match wdl {
        Loss | MaybeLoss => TBWdl::Loss,
        BlessedLoss => TBWdl::BlessedLoss,
        Draw => TBWdl::Draw,
        CursedWin | MaybeWin => TBWdl::CursedWin,
        Win => TBWdl::Win,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_kbbk_returns_win() {
        // KBB vs K should be a win for the side with bishops.
        let tb = Tablebase::new(vec![PathBuf::from("tables")], 7, true);
        let board: Board = "k7/8/8/8/8/8/KBB5/8 w - - 0 1"
            .parse()
            .expect("valid FEN");

        let res = tb.probe_root(&board, 0).expect("tb root probe");
        assert_eq!(res.wdl, TBWdl::Win);
        assert!(res.best_move.is_some());
    }
}

fn board_to_syzygy(board: &Board) -> Option<SPosition> {
    let fen_str = board.to_string();
    let fen: Fen = fen_str.parse().ok()?;
    fen.into_position(CastlingMode::Standard).ok()
}

fn syzygy_move_to_chess_move(board: &Board, mv: SMove) -> Option<ChessMove> {
    let (from_raw, to_raw) = match (mv.from(), mv.to()) {
        (Some(f), t) => (f, t),
        _ => return None,
    };

    let from_sq = shakmaty_square_to_chess(from_raw)?;
    let to_sq = shakmaty_square_to_chess(to_raw)?;
    let promotion = mv.promotion().map(shakmaty_role_to_piece);
    let candidate = ChessMove::new(from_sq, to_sq, promotion);
    if board.legal(candidate) {
        Some(candidate)
    } else {
        None
    }
}

fn shakmaty_square_to_chess(sq: SSquare) -> Option<Square> {
    let file = File::from_index(sq.file().into());
    let rank = Rank::from_index(sq.rank().into());
    Square::make_square(rank, file).into()
}

fn shakmaty_role_to_piece(role: Role) -> Piece {
    match role {
        Role::Queen => Piece::Queen,
        Role::Rook => Piece::Rook,
        Role::Bishop => Piece::Bishop,
        Role::Knight => Piece::Knight,
        Role::Pawn => Piece::Pawn,
        Role::King => Piece::Queen, // should not happen for promotions
    }
}
