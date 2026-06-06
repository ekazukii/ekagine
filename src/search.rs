use crate::engine_core::{
    for_each_capture_pseudo_legal, Board, ChessMove, Color, MoveGen, Piece, PinInfo, Square,
};
use crate::movegen::{
    is_en_passant_capture, see_for_sort, static_exchange_eval, IncrementalMoveGen,
};
use crate::nnue::NNUEState;
use crate::tables::{
    CaptureHistoryTable, ContHist, CorrHist, CountermoveTable, HistoryTable, KillerTable,
};
use crate::time::{TimeManagerHandle, TimePlan, TimeScaleFactors};
use crate::{
    board_do_move, board_pop, send_message, RepetitionTable, StopFlag, TTFlag, TranspositionTable,
    NEG_INFINITY, POS_INFINITY,
};
use lazy_static::lazy_static;
use smallvec::SmallVec;
use std::cmp;
use std::io::Stdout;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{io, thread};

enum SearchScore {
    CANCELLED,
    EVAL(i32),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStats {
    pub nodes: u64,
    pub qnodes: u64,
    pub tt_hits: u64,
    pub tt_cutoff_exact: u64,
    pub tt_cutoff_lower: u64,
    pub tt_cutoff_upper: u64,
    pub beta_cutoffs: u64,
    pub beta_cutoffs_quiescence: u64,
    pub null_move_prunes: u64,
    pub futility_prunes: u64,
    pub reverse_futility_prunes: u64,
    pub razoring_prunes: u64,
    pub late_move_prunes: u64,
    pub lmr_researches: u64,
    pub incremental_move_gen_inits: u64,
    pub incremental_move_gen_capture_lists: u64,
    pub singular_extensions: u64,
    pub singular_checks: u64,
    pub effective_branching_factor: f64,
    pub depth: u64,
}

impl SearchStats {
    fn diff(&self, other: &SearchStats) -> SearchStats {
        SearchStats {
            nodes: self.nodes.saturating_sub(other.nodes),
            qnodes: self.qnodes.saturating_sub(other.qnodes),
            tt_hits: self.tt_hits.saturating_sub(other.tt_hits),
            tt_cutoff_exact: self.tt_cutoff_exact.saturating_sub(other.tt_cutoff_exact),
            tt_cutoff_lower: self.tt_cutoff_lower.saturating_sub(other.tt_cutoff_lower),
            tt_cutoff_upper: self.tt_cutoff_upper.saturating_sub(other.tt_cutoff_upper),
            beta_cutoffs: self.beta_cutoffs.saturating_sub(other.beta_cutoffs),
            beta_cutoffs_quiescence: self
                .beta_cutoffs_quiescence
                .saturating_sub(other.beta_cutoffs_quiescence),
            null_move_prunes: self.null_move_prunes.saturating_sub(other.null_move_prunes),
            futility_prunes: self.futility_prunes.saturating_sub(other.futility_prunes),
            reverse_futility_prunes: self
                .reverse_futility_prunes
                .saturating_sub(other.reverse_futility_prunes),
            razoring_prunes: self.razoring_prunes.saturating_sub(other.razoring_prunes),
            late_move_prunes: self.late_move_prunes.saturating_sub(other.late_move_prunes),
            lmr_researches: self.lmr_researches.saturating_sub(other.lmr_researches),
            incremental_move_gen_inits: self
                .incremental_move_gen_inits
                .saturating_sub(other.incremental_move_gen_inits),
            incremental_move_gen_capture_lists: self
                .incremental_move_gen_capture_lists
                .saturating_sub(other.incremental_move_gen_capture_lists),
            singular_extensions: self
                .singular_extensions
                .saturating_sub(other.singular_extensions),
            singular_checks: self.singular_checks.saturating_sub(other.singular_checks),
            effective_branching_factor: 0.0,
            depth: self.depth,
        }
    }

    fn update_effective_branching_factor(&mut self, nodes: u64, qnodes: u64, depth: usize) {
        if depth == 0 {
            self.effective_branching_factor = 0.0;
            return;
        }

        let total_nodes = nodes.saturating_add(qnodes);
        if total_nodes == 0 {
            self.effective_branching_factor = 0.0;
            return;
        }

        let depth_f = depth as f64;
        let total_nodes_f = total_nodes as f64;
        self.effective_branching_factor = total_nodes_f.powf(1.0 / depth_f);
    }

    fn record_depth(&mut self, ply_from_root: i32) {
        if ply_from_root >= 0 {
            let ply = ply_from_root as u64;
            if ply > self.depth {
                self.depth = ply;
            }
        }
    }

    pub(crate) fn format_as_info(&self) -> String {
        format!(
            "nodes={} qnodes={} tt_hits={} tt_exact={} tt_lower={} tt_upper={} beta_cut={} qbeta_cut={} null_prune={} futility_prune={} rfutility_prune={} late_move_prune={} lmr_retry={} img_init={} img_capgen={} sing_ext={} sing_check={} ebf={:.2} depth={}",
            self.nodes,
            self.qnodes,
            self.tt_hits,
            self.tt_cutoff_exact,
            self.tt_cutoff_lower,
            self.tt_cutoff_upper,
            self.beta_cutoffs,
            self.beta_cutoffs_quiescence,
            self.null_move_prunes,
            self.futility_prunes,
            self.reverse_futility_prunes,
            self.late_move_prunes,
            self.lmr_researches,
            self.incremental_move_gen_inits,
            self.incremental_move_gen_capture_lists,
            self.singular_extensions,
            self.singular_checks,
            self.effective_branching_factor,
            self.depth,
        )
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchOutcome {
    pub best_move: Option<ChessMove>,
    pub score: i32,
    #[allow(dead_code)]
    pub stats: SearchStats,
}

impl SearchOutcome {
    fn new(best_move: Option<ChessMove>, score: i32, stats: SearchStats) -> Self {
        SearchOutcome {
            best_move,
            score,
            stats,
        }
    }
}

// ----------------------------------------------------------------------------
// Mate distance scoring helpers
// ----------------------------------------------------------------------------
// Hard ply ceiling. Real searches never approach this; it exists purely to
// stop pathological extension/check lines from overflowing the fixed-size
// NNUE accumulator stack (and the search stack). Comfortably below both caps.
pub const MAX_SEARCH_PLY: i32 = 120;

// A large value to represent mate; must be >> any centipawn eval
pub const MATE_VALUE: i32 = 1_000_000;
// Threshold to detect mate-encoded scores
pub const MATE_THRESHOLD: i32 = 999_000;

#[inline]
fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE_THRESHOLD
}

#[inline]
fn mate_in_plies(ply: i32) -> i32 {
    MATE_VALUE - ply
}

#[inline]
fn mated_in_plies(ply: i32) -> i32 {
    -MATE_VALUE + ply
}

// The TT packs the score into an i16, so the search-scale mate values (~1e6)
// would be clamped to 32767 and lose their mate-ness on reload. Map mate
// scores into a compact i16-safe band (TT_MATE) on store and back on load,
// keeping the distance node-relative so transpositions reload correctly.
const TT_MATE: i32 = 32_000;
const TT_MATE_THRESHOLD: i32 = TT_MATE - 2 * MAX_SEARCH_PLY;

#[inline]
fn tt_score_on_store(score: i32, ply_from_root: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        // mate for the side to move: store distance-from-this-node in the
        // compact band so it survives the i16 packing.
        TT_MATE - (MATE_VALUE - score) + ply_from_root
    } else if score <= -MATE_THRESHOLD {
        -TT_MATE + (MATE_VALUE + score) - ply_from_root
    } else {
        score
    }
}

#[inline]
fn tt_score_on_load(score: i32, ply_from_root: i32) -> i32 {
    if score >= TT_MATE_THRESHOLD {
        MATE_VALUE - (TT_MATE - score) - ply_from_root
    } else if score <= -TT_MATE_THRESHOLD {
        -MATE_VALUE + (TT_MATE + score) + ply_from_root
    } else {
        score
    }
}

/// Format a score for UCI output ("cp X" or "mate N").
/// `side` is the side to move in the root position.
pub fn uci_score_string(score: i32, side: Color) -> String {
    let color = if side == Color::White { 1 } else { -1 };
    if is_mate_score(score) {
        let ply_to_mate = MATE_VALUE - score.abs();
        // Convert plies to moves (ceil(ply/2))
        let moves = (ply_to_mate + 1) / 2;
        let signed_moves = if score * color > 0 { moves } else { -moves };
        format!("mate {}", signed_moves)
    } else {
        format!("cp {}", score * color)
    }
}

#[inline(always)]
fn should_stop(flag: &StopFlag) -> bool {
    flag.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Quiescence search (captures‑only) — negamax style
// ---------------------------------------------------------------------------
/// Quiesce search, this function perform an negamax w/ alpha beta pruning but only analysing the
/// capture and checks to deeper in the tree while still being a performant search. This search
/// is meant to be run after the classic search to avoid evaluating when pieces are hanging
fn quiesce_negamax_it(
    ctx: &mut ThreadContext,
    board: &Board,
    mut alpha: i32,
    beta: i32,
    ply_from_root: i32,
) -> i32 {
    if ctx.repetition.is_in_threefold_scenario(board) {
        return 0;
    }

    ctx.stats.record_depth(ply_from_root);
    ctx.stats.qnodes += 1;

    if ply_from_root >= MAX_SEARCH_PLY {
        return ctx.nnue.evaluate(board);
    }

    let zob = board.get_hash();
    let in_check = board.checkers().0 != 0;

    // Quiescence TT lookup
    let hit = ctx.tt.probe(zob);
    if let Some(tt_hit) = hit {
        let tt_score = tt_score_on_load(tt_hit.score, ply_from_root);
        match tt_hit.flag {
            TTFlag::Exact => {
                ctx.stats.tt_cutoff_exact += 1;
                return tt_score;
            }
            TTFlag::Lower => {
                if tt_score >= beta {
                    ctx.stats.tt_cutoff_lower += 1;
                    ctx.stats.beta_cutoffs_quiescence += 1;
                    return tt_score;
                }
            }
            TTFlag::Upper => {
                if tt_score <= alpha {
                    ctx.stats.tt_cutoff_upper += 1;
                    return tt_score;
                }
            }
        }
    }

    if in_check {
        let pin_info = PinInfo::for_board(board);
        let evasions = MoveGen::new_legal(board);
        let alpha_orig = alpha;
        let mut best_move_opt: Option<ChessMove> = None;
        let mut any_legal = false;

        for mv in evasions {
            if !pin_info.move_is_legal(board, mv) {
                continue;
            }
            let new_board = board.make_move_new(mv);
            any_legal = true;
            ctx.repetition.push(
                new_board.get_hash(),
                crate::resets_halfmove_clock(board, mv),
            );
            ctx.nnue.push();
            ctx.nnue.apply_move(board, mv);
            let score = -quiesce_negamax_it(ctx, &new_board, -beta, -alpha, ply_from_root + 1);
            ctx.repetition.pop();
            ctx.nnue.pop();

            if score > alpha {
                alpha = score;
                best_move_opt = Some(mv);
            }

            if score >= beta {
                ctx.stats.beta_cutoffs_quiescence += 1;
                break;
            }
        }

        if !any_legal {
            return mated_in_plies(ply_from_root);
        }

        let bound = if alpha >= beta {
            TTFlag::Lower
        } else if alpha > alpha_orig {
            TTFlag::Exact
        } else {
            TTFlag::Upper
        };

        ctx.tt.store(
            zob,
            0,
            tt_score_on_store(alpha, ply_from_root),
            bound,
            best_move_opt,
            None,
        );

        return alpha;
    }

    let static_eval = if let Some(ref tt_hit) = hit {
        Some(tt_hit.eval.unwrap_or_else(|| ctx.nnue.evaluate(board)))
    } else {
        Some(ctx.nnue.evaluate(board))
    };

    // This path is only reached when not in check (the in-check case returns
    // above). Apply the correction-history adjustment to the stand-pat used for
    // pruning/cutoffs while keeping the raw eval in the TT (`static_eval`).
    let raw_stand = static_eval.unwrap_or(NEG_INFINITY);
    let stand_pat = if is_mate_score(raw_stand) {
        raw_stand
    } else {
        (raw_stand
            + ctx
                .corrhist
                .correction(board.side_to_move(), pawn_structure_key(board)))
        .clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1)
    };

    let alpha_orig = alpha;
    let mut best_value = stand_pat;
    let mut best_move_opt: Option<ChessMove> = None;

    if stand_pat >= beta {
        ctx.stats.beta_cutoffs_quiescence += 1;
        return stand_pat;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let futility = stand_pat + QUIESCE_FUTILITY_MARGIN.get();
    let pin_info = PinInfo::for_board(board);
    // Fast path: in non-check positions with no pinned pieces, only king
    // captures and EP captures need the per-move legality test.
    let fast_path = pin_info.num_checkers == 0 && pin_info.pinned == 0;
    let ep_sq = board.en_passant();
    for (mv, _val) in get_captures(board) {
        if !in_check && futility <= alpha && see_for_sort(board, mv) < 0 {
            continue;
        }

        let dest_piece = board.piece_on(mv.get_dest());
        let is_en_passant = dest_piece.is_none() && is_en_passant_capture(board, mv);
        if dest_piece.is_none() && !is_en_passant {
            continue;
        }
        let needs_legal_check = !fast_path
            || (mv.get_source().to_index() as u8) == pin_info.king_sq
            || Some(mv.get_dest()) == ep_sq;
        if needs_legal_check && !pin_info.move_is_legal(board, mv) {
            continue;
        }
        let new_board = board.make_move_new(mv);
        ctx.repetition.push(
            new_board.get_hash(),
            crate::resets_halfmove_clock(board, mv),
        );
        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);
        let score = -quiesce_negamax_it(ctx, &new_board, -beta, -alpha, ply_from_root + 1);
        ctx.repetition.pop();
        ctx.nnue.pop();

        if score > alpha {
            alpha = score;
            best_move_opt = Some(mv);
        }

        if score > best_value {
            best_value = score;
        }

        if score >= beta {
            ctx.stats.beta_cutoffs_quiescence += 1;
            alpha = score;
            break;
        }
    }
    let bound = if alpha >= beta {
        TTFlag::Lower
    } else if alpha > alpha_orig {
        TTFlag::Exact
    } else {
        TTFlag::Upper
    };

    ctx.tt.store(
        zob,
        0,
        tt_score_on_store(alpha, ply_from_root),
        bound,
        best_move_opt,
        static_eval,
    );
    alpha
}

#[inline]
fn has_non_pawn_material(board: &Board, side: Color) -> bool {
    let bb = board.color_combined(side);
    let non_pawns = (*board.pieces(Piece::Knight)
        | *board.pieces(Piece::Bishop)
        | *board.pieces(Piece::Rook)
        | *board.pieces(Piece::Queen))
        & bb;
    non_pawns.0 != 0
}

/// Cheap pawn-structure hash (white & black pawn bitboards mixed), used to key
/// the correction-history table. Not collision-free, but the table tolerates
/// collisions gracefully.
#[inline]
fn pawn_structure_key(board: &Board) -> usize {
    let wp = (*board.pieces(Piece::Pawn) & *board.color_combined(Color::White)).0;
    let bp = (*board.pieces(Piece::Pawn) & *board.color_combined(Color::Black)).0;
    let mut h = wp.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ bp.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h as usize
}

/// Create a new board representing a null move using Board::null_move(),
/// then update the repetition table for that new position.
fn board_do_null_move(board: &Board, repetition_table: &mut RepetitionTable) -> Option<Board> {
    // Use the chess crate's built-in null move generator
    let new_board = board.null_move()?;
    repetition_table.push(new_board.get_hash(), true);
    Some(new_board)
}

const MVV_LVA_TABLE: [[u8; 6]; 6] = [
    // Victim = Pawn
    [15, 14, 13, 12, 11, 10],
    // Victim = Knight
    [25, 24, 23, 22, 21, 20],
    // Victim = Bishop
    [35, 34, 33, 32, 31, 30],
    // Victim = Rook
    [45, 44, 43, 42, 41, 40],
    // Victim = Queen
    [55, 54, 53, 52, 51, 50],
    // Victim = King (we never actually capture kings in legal move generation)
    [0, 0, 0, 0, 0, 0],
];

// ─── SPSA-tunable search parameters ─────────────────────────────────────────
// Runtime-adjustable via UCI `setoption`, so an external tuner (SPSA) can
// optimise the eval-scale-sensitive margins without rebuilding the engine.
pub struct Tunable {
    pub name: &'static str,
    atom: std::sync::atomic::AtomicI32,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

impl Tunable {
    pub const fn new(name: &'static str, default: i32, min: i32, max: i32) -> Self {
        Self {
            name,
            atom: std::sync::atomic::AtomicI32::new(default),
            default,
            min,
            max,
        }
    }

    #[inline(always)]
    pub fn get(&self) -> i32 {
        self.atom.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set(&self, value: i32) {
        self.atom.store(
            value.clamp(self.min, self.max),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// Suppress root-search info logging (used by bulk rescoring).
pub static QUIET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Optional node limit for the current search (0 = unlimited), set by
/// `go nodes N` and by fixed-node workloads (datagen). Checked alongside the
/// periodic stop test.
pub static GO_MAX_NODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static ASPIRATION_START_WINDOW: Tunable = Tunable::new("AspirationWindow", 25, 8, 80);
pub static QUIESCE_FUTILITY_MARGIN: Tunable = Tunable::new("QFutilityMargin", 200, 60, 450);
pub static FUTILITY_MARGIN_BASE: Tunable = Tunable::new("FutilityBase", 100, 30, 260);
pub static FUTILITY_MARGIN_PER_DEPTH: Tunable = Tunable::new("FutilityPerDepth", 75, 25, 180);
pub static RAZORING_MARGIN: Tunable = Tunable::new("RazoringMargin", 300, 100, 650);
pub static PROBCUT_MARGIN: Tunable = Tunable::new("ProbcutMargin", 180, 60, 400);
pub static SINGULAR_BETA_MARGIN_MULTIPLIER: Tunable = Tunable::new("SingularBetaMult", 2, 1, 6);
pub static HIST_LMR_DIVISOR: Tunable = Tunable::new("HistLmrDivisor", 16384, 4096, 32768);

pub static TUNABLES: [&Tunable; 8] = [
    &ASPIRATION_START_WINDOW,
    &QUIESCE_FUTILITY_MARGIN,
    &FUTILITY_MARGIN_BASE,
    &FUTILITY_MARGIN_PER_DEPTH,
    &RAZORING_MARGIN,
    &PROBCUT_MARGIN,
    &SINGULAR_BETA_MARGIN_MULTIPLIER,
    &HIST_LMR_DIVISOR,
];

/// Set a tunable by UCI option name (case-insensitive). Returns false if unknown.
pub fn set_tunable(name: &str, value: i32) -> bool {
    for t in TUNABLES {
        if t.name.eq_ignore_ascii_case(name) {
            t.set(value);
            return true;
        }
    }
    false
}

const FUTILITY_PRUNE_MAX_DEPTH: i16 = 6;
const REVERSE_FUTILITY_PRUNE_MAX_DEPTH: i16 = 4;

const RAZORING_DEPTH: i16 = 1;

const CHECK_EXTENSION_DEPTH_LIMIT: i16 = 2;
const PASSED_PAWN_EXTENSION_DEPTH_LIMIT: i16 = 6;
const SINGULAR_EXTENSION_MIN_DEPTH: i16 = 8;
const SINGULAR_EXTENSION_DEPTH_LIMIT: i16 = 12;
const PROBCUT_MIN_DEPTH: i16 = 5;
const PROBCUT_REDUCTION: i16 = 4;
const SEE_PRUNE_MAX_DEPTH: i16 = 6;
const LATE_MOVE_PRUNING_MAX_DEPTH: i16 = 5;
const LATE_MOVE_PRUNING_BASE: usize = 4;
const LATE_MOVE_PRUNING_SCALE: usize = 2;

#[inline]
fn futility_margin(depth: i16, is_improving: bool) -> i32 {
    let d = depth as i32;
    let (fb, fpd) = (FUTILITY_MARGIN_BASE.get(), FUTILITY_MARGIN_PER_DEPTH.get());
    let base = fb + fpd * d.saturating_sub(1);
    // Tighter margin (more pruning) when the position is not improving.
    if is_improving {
        base
    } else {
        (base - fpd / 2).max(fb / 2)
    }
}

#[inline]
fn reverse_futility_margin(depth: i16) -> i32 {
    let depth = depth as i32;
    FUTILITY_MARGIN_PER_DEPTH.get() * depth
}

#[inline]
fn late_move_pruning_threshold(depth: i16, is_improving: bool) -> usize {
    let depth = depth.max(0) as usize;
    let base = LATE_MOVE_PRUNING_BASE + depth * LATE_MOVE_PRUNING_SCALE;
    // Prune later when improving, earlier when not.
    if is_improving {
        base + depth
    } else {
        base
    }
}

// Late Move Reduction (LMR) table and constants
const LMR_MAX_DEPTH: usize = 64;
const LMR_MAX_MOVES: usize = 64;
const LMR_BASE: f64 = 0.75;
const LMR_DIVISOR: f64 = 2.25;

lazy_static! {
    static ref LMR_TABLE: [[i16; LMR_MAX_MOVES]; LMR_MAX_DEPTH] = compute_lmr_table();
}

fn compute_lmr_table() -> [[i16; LMR_MAX_MOVES]; LMR_MAX_DEPTH] {
    let mut table = [[0i16; LMR_MAX_MOVES]; LMR_MAX_DEPTH];
    for depth in 1..LMR_MAX_DEPTH {
        for move_idx in 1..LMR_MAX_MOVES {
            let reduction = LMR_BASE + (depth as f64).ln() * (move_idx as f64).ln() / LMR_DIVISOR;
            table[depth][move_idx] = reduction.round() as i16;
        }
    }
    table
}

#[inline]
fn lmr_reduction(depth: i16, move_idx: usize, is_pv_node: bool, hist: i32) -> i16 {
    let d = (depth as usize).min(LMR_MAX_DEPTH - 1);
    let m = move_idx.min(LMR_MAX_MOVES - 1);

    let mut reduction = LMR_TABLE[d][m];

    // Reduce less in PV nodes.
    if is_pv_node {
        reduction = reduction.saturating_sub(1);
    }
    // Reduce less for moves with good history, more for bad history.
    let hist_adj = (hist / HIST_LMR_DIVISOR.get()).clamp(-2, 2) as i16;
    reduction -= hist_adj;

    // Ensure minimum reduction of 1 when LMR applies
    reduction.max(1)
}

#[inline]
fn is_passed_pawn(board: &Board, square: Square, color: Color) -> bool {
    let enemy_color = match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    let enemy_pawns = *board.pieces(Piece::Pawn) & board.color_combined(enemy_color);
    let file_idx = square.get_file().to_index() as i32;
    let rank_idx = square.get_rank().to_index() as i32;

    let mut iter = enemy_pawns;
    for enemy_sq in &mut iter {
        let df = enemy_sq.get_file().to_index() as i32 - file_idx;
        if df.abs() <= 1 {
            let enemy_rank = enemy_sq.get_rank().to_index() as i32;
            match color {
                Color::White if enemy_rank > rank_idx => return false,
                Color::Black if enemy_rank < rank_idx => return false,
                _ => {}
            }
        }
    }

    true
}

#[inline]
fn is_passed_pawn_push(parent: &Board, child: &Board, mv: ChessMove) -> bool {
    if parent.piece_on(mv.get_source()) != Some(Piece::Pawn) {
        return false;
    }

    let color_to_move = parent.side_to_move();
    let dest = mv.get_dest();

    if child.piece_on(dest) != Some(Piece::Pawn) {
        return false;
    }

    let target_rank = match color_to_move {
        Color::White => 6,
        Color::Black => 1,
    };
    if dest.get_rank().to_index() != target_rank {
        return false;
    }

    is_passed_pawn(child, dest, color_to_move)
}

fn get_captures(board: &Board) -> SmallVec<[(ChessMove, u8); 64]> {
    // Generate only capture-type moves (incl. en-passant + capture-promotions)
    // via the staged generator, then MVV-LVA-sort.
    let mut scored_moves: SmallVec<[(ChessMove, u8); 64]> = SmallVec::new();

    for_each_capture_pseudo_legal(board, |mv| {
        let score = if let Some(victim_piece) = board.piece_on(mv.get_dest()) {
            if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
                let victim_idx = victim_piece.to_index();
                let attacker_idx = attacker_piece.to_index();
                MVV_LVA_TABLE[victim_idx][attacker_idx]
            } else {
                0
            }
        } else if is_en_passant_capture(board, mv) {
            25 // arbitrary EP value; tune as needed
        } else {
            0
        };
        scored_moves.push((mv, score));
    });

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves
}

// ---------------------------------------------------------------------------
// Singular Extension helpers
// ---------------------------------------------------------------------------

#[inline]
fn should_check_singular_extension(
    depth: i16,
    in_check: bool,
    tt_move: Option<ChessMove>,
    tt_depth: i16,
    tt_flag: TTFlag,
    exclude_move: Option<ChessMove>,
) -> bool {
    // Must have sufficient depth
    if depth < SINGULAR_EXTENSION_MIN_DEPTH {
        return false;
    }

    // Don't check at depths where we won't apply the extension anyway
    if depth > SINGULAR_EXTENSION_DEPTH_LIMIT {
        return false;
    }

    // Singular checks now run at non-PV nodes too (the bulk of the tree); the
    // verification search is correct since the exclude_move TT cutoff/store
    // bugs were fixed.

    // Don't do singular extensions while in check (too expensive)
    if in_check {
        return false;
    }

    // Must have a TT move to test
    if tt_move.is_none() {
        return false;
    }

    // TT entry must be deep enough (at least depth-3)
    if tt_depth < depth - 3 {
        return false;
    }

    // TT must indicate this move is good (Lower or Exact bound)
    // Upper bound means the move failed low, not a good candidate
    if tt_flag == TTFlag::Upper {
        return false;
    }

    // Don't do singular extensions in the verification search itself!
    if exclude_move.is_some() {
        return false;
    }

    true
}

fn is_singular(
    ctx: &mut ThreadContext,
    board: &Board,
    candidate_move: ChessMove,
    depth: i16,
    beta: i32,
    ply_from_root: i32,
) -> bool {
    // Reduced beta: if other moves can't reach this, candidate is singular
    let s_beta_margin = SINGULAR_BETA_MARGIN_MULTIPLIER.get() * depth as i32;
    let s_beta = beta - s_beta_margin;

    // Reduced depth for verification search (typically depth/2 - 1)
    let s_depth = (depth / 2) - 1;
    if s_depth <= 0 {
        return false;
    }

    // Search all moves EXCEPT the candidate
    let result = negamax_it(
        ctx,
        board,
        s_depth,
        s_beta - 1, // alpha
        s_beta,     // beta (null window)
        ply_from_root,
        false,                // Not a PV node
        Some(candidate_move), // Exclude this move
    );

    match result {
        SearchScore::CANCELLED => false,
        SearchScore::EVAL(score) => {
            // If no other move reached s_beta, the candidate is singular
            score < s_beta
        }
    }
}

// ---------------------------------------------------------------------------
// Core Negamax with α/β pruning + PV maintenance — renamed `negamax_it`.
// ---------------------------------------------------------------------------
/// This is the core of the move search, it perform a negamax styled search with alpha/beta pruning
/// Usage of quiesce search at end of search and tt/pv cutoff
fn negamax_it(
    ctx: &mut ThreadContext,
    board: &Board,
    depth: i16,
    mut alpha: i32,
    beta: i32,
    ply_from_root: i32,
    is_pv_node: bool,
    exclude_move: Option<ChessMove>,
) -> SearchScore {
    if ctx.stats.nodes & 1023 == 0 {
        let limit = GO_MAX_NODES.load(std::sync::atomic::Ordering::Relaxed);
        if limit != 0 && ctx.stats.nodes >= limit {
            ctx.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if ctx.stats.nodes & 2047 == 0 && should_stop(ctx.stop) {
        if !QUIET.load(std::sync::atomic::Ordering::Relaxed) {
            let mut out = io::stdout();
            let info_line = "info string break stop".to_string();
            send_message(&mut out, &info_line);
        }
        return SearchScore::CANCELLED;
    }

    ctx.stats.record_depth(ply_from_root);
    ctx.stats.nodes += 1;

    if ctx.repetition.is_in_threefold_scenario(board) {
        return SearchScore::EVAL(0);
    }

    if ply_from_root >= MAX_SEARCH_PLY {
        return SearchScore::EVAL(ctx.nnue.evaluate(board));
    }

    if depth <= 0 {
        return SearchScore::EVAL(quiesce_negamax_it(ctx, board, alpha, beta, ply_from_root));
    }

    let depth_remaining = depth;
    let zob = board.get_hash();

    let tt_hit = ctx.tt.probe(zob);
    if let Some(entry) = tt_hit {
        ctx.stats.tt_hits += 1;
        // Don't take a TT cutoff during a singular verification search: the
        // entry was stored for the full position (including the excluded move),
        // so cutting on it would defeat the exclusion.
        if exclude_move.is_none() && entry.depth >= depth_remaining {
            let tt_score = tt_score_on_load(entry.score, ply_from_root);
            match entry.flag {
                TTFlag::Exact => {
                    ctx.stats.tt_cutoff_exact += 1;
                    return SearchScore::EVAL(tt_score);
                }
                TTFlag::Lower => {
                    if tt_score >= beta {
                        ctx.stats.tt_cutoff_lower += 1;
                        ctx.stats.beta_cutoffs += 1;
                        return SearchScore::EVAL(tt_score);
                    }
                }
                TTFlag::Upper => {
                    if tt_score <= alpha {
                        ctx.stats.tt_cutoff_upper += 1;
                        return SearchScore::EVAL(tt_score);
                    }
                }
            }
        }
    }

    let raw_eval = if let Some(hit) = tt_hit {
        if let Some(tt_eval) = hit.eval {
            tt_eval
        } else {
            ctx.nnue.evaluate(board)
        }
    } else {
        ctx.nnue.evaluate(board)
    };
    let in_check = board.checkers().0 != 0;
    // Correction history: nudge the raw static eval toward observed search
    // results for this pawn structure. Stored raw in the TT; corrected here for
    // pruning/improving decisions. Disabled in check / near mate.
    let corr_key = pawn_structure_key(board);
    let eval = if in_check || is_mate_score(raw_eval) {
        raw_eval
    } else {
        (raw_eval + ctx.corrhist.correction(board.side_to_move(), corr_key))
            .clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1)
    };
    ctx.search_stack.set(ply_from_root as usize, eval);
    let is_improving = ctx.search_stack.is_improving(ply_from_root as usize, eval);

    // Razoring: at depth 1, if eval is far below alpha, verify position is bad via qsearch
    if depth_remaining == RAZORING_DEPTH
        && !is_pv_node
        && !in_check
        && alpha > NEG_INFINITY + 1000
        && beta < POS_INFINITY - 1000
        && !is_mate_score(alpha)
        && !is_mate_score(beta)
        && eval + RAZORING_MARGIN.get() < alpha
    {
        let razor_score = quiesce_negamax_it(ctx, board, alpha, beta, ply_from_root);
        if razor_score <= alpha {
            ctx.stats.razoring_prunes += 1;
            return SearchScore::EVAL(razor_score);
        }
    }

    if !is_pv_node
        && !in_check
        && depth_remaining >= 2
        && eval >= beta
        && !is_mate_score(eval)
        && has_non_pawn_material(board, board.side_to_move())
    {
        // Reduction scales with depth and how far eval exceeds beta (safe:
        // C guarantees eval >= beta here), plus the improving bonus.
        let r: i16 = 2
            + depth_remaining / 4
            + ((eval - beta) / 256).clamp(0, 2) as i16
            + if is_improving { 1 } else { 0 };
        if let Some(nboard) = board_do_null_move(board, ctx.repetition) {
            // A null move changes no pieces, so the accumulator is identical to
            // the parent's. Reuse the parent's accumulator slot directly instead
            // of pushing a full copy: the null child reads `current_acc` (the
            // unchanged parent accumulator) and its own children push/apply from
            // there as usual. Saves a 4KB accumulator copy per null move.
            // The child of a null move has no "previous move"; clear this ply's
            // slot so continuation-history / countermove lookups in the null
            // subtree don't read a stale sibling move.
            ctx.search_stack.set_null(ply_from_root as usize);
            let result = negamax_it(
                ctx,
                &nboard,
                (depth_remaining - 1 - r).max(0),
                -beta,
                -beta + 1,
                ply_from_root + 1,
                false,
                None,
            );
            board_pop(&nboard, ctx.repetition);
            match result {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(v) => {
                    let value = -v;
                    if value >= beta {
                        ctx.stats.null_move_prunes += 1;
                        ctx.stats.beta_cutoffs += 1;
                        // Don't return an unproven mate score from a null search.
                        let value = if is_mate_score(value) { beta } else { value };
                        return SearchScore::EVAL(value);
                    }
                }
            }
        }
    }

    // ProbCut: at non-PV nodes, if a good capture beats a raised beta even in a
    // shallow verification search, the position is very likely a fail-high — so
    // prune. Same make/unmake protocol as the main move loop.
    if !is_pv_node
        && !in_check
        && exclude_move.is_none()
        && depth_remaining >= PROBCUT_MIN_DEPTH
        && !is_mate_score(beta)
        && eval >= beta
    {
        let pc_beta = beta.saturating_add(PROBCUT_MARGIN.get());
        let pin_info = PinInfo::for_board(board);
        let fast_path = pin_info.num_checkers == 0 && pin_info.pinned == 0;
        let ep_sq = board.en_passant();
        let see_threshold = pc_beta - eval;
        for (mv, _) in get_captures(board) {
            let dest_piece = board.piece_on(mv.get_dest());
            let is_ep = dest_piece.is_none() && is_en_passant_capture(board, mv);
            if dest_piece.is_none() && !is_ep {
                continue;
            }
            let needs_legal = !fast_path
                || (mv.get_source().to_index() as u8) == pin_info.king_sq
                || Some(mv.get_dest()) == ep_sq;
            if needs_legal && !pin_info.move_is_legal(board, mv) {
                continue;
            }
            if see_for_sort(board, mv) < see_threshold {
                continue;
            }

            let new_board = board_do_move(board, mv, ctx.repetition);
            ctx.nnue.push();
            ctx.nnue.apply_move(board, mv);
            ctx.search_stack
                .set_move(ply_from_root as usize, mv, board.piece_on(mv.get_source()));
            let result = negamax_it(
                ctx,
                &new_board,
                depth_remaining - PROBCUT_REDUCTION,
                -pc_beta,
                -pc_beta + 1,
                ply_from_root + 1,
                false,
                None,
            );
            ctx.nnue.pop();
            ctx.repetition.pop();
            match result {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(v) => {
                    let score = -v;
                    if score >= pc_beta {
                        ctx.tt.store(
                            zob,
                            depth_remaining - PROBCUT_REDUCTION + 1,
                            tt_score_on_store(score, ply_from_root),
                            TTFlag::Lower,
                            Some(mv),
                            Some(raw_eval),
                        );
                        return SearchScore::EVAL(score);
                    }
                }
            }
        }
    }

    let mut best_value = NEG_INFINITY;
    let mut best_move_opt: Option<ChessMove> = None;
    let alpha_orig = alpha;
    let mover = board.side_to_move();

    if depth_remaining <= REVERSE_FUTILITY_PRUNE_MAX_DEPTH
        && !in_check
        && !is_pv_node
        && alpha != NEG_INFINITY
        && beta != POS_INFINITY
        && !is_mate_score(beta)
        && !is_mate_score(eval)
    {
        let mut margin = reverse_futility_margin(depth_remaining);
        if is_improving {
            margin /= 2;
        }

        if eval.saturating_sub(margin) >= beta {
            ctx.stats.reverse_futility_prunes += 1;
            return SearchScore::EVAL(beta);
        }
    }

    // Singular extension check
    let tt_move = tt_hit.and_then(|e| e.best_move);
    let mut singular_extension: i16 = 0;
    if let Some(entry) = tt_hit {
        if should_check_singular_extension(
            depth_remaining,
            in_check,
            tt_move,
            entry.depth,
            entry.flag,
            exclude_move,
        ) {
            ctx.stats.singular_checks += 1;
            if let Some(candidate_move) = tt_move {
                if is_singular(
                    ctx,
                    board,
                    candidate_move,
                    depth_remaining,
                    beta,
                    ply_from_root,
                ) {
                    singular_extension = 1;
                    ctx.stats.singular_extensions += 1;
                }
            }
        }
    }

    ctx.stats.incremental_move_gen_inits += 1;
    let ply_index = ply_from_root.max(0) as usize;
    let killer_moves = ctx.killers.killers_for(ply_index);
    let countermove = ctx
        .search_stack
        .get_prev_move(ply_index)
        .and_then(|prev_move| ctx.countermoves.get(prev_move, !mover));
    let cont_ctx1 = ctx.search_stack.cont_context(ply_index, 1);
    let cont_ctx2 = ctx.search_stack.cont_context(ply_index, 2);
    let mut incremental_move_gen = IncrementalMoveGen::new(
        board,
        ctx.tt,
        killer_moves,
        countermove,
        cont_ctx1,
        cont_ctx2,
    );

    if incremental_move_gen.is_over() {
        if in_check {
            return SearchScore::EVAL(mated_in_plies(ply_from_root));
        }
        return SearchScore::EVAL(0);
    }

    let can_fp = !in_check && depth_remaining <= FUTILITY_PRUNE_MAX_DEPTH;
    let fp_val = if can_fp {
        eval + futility_margin(depth_remaining, is_improving)
    } else {
        POS_INFINITY
    };

    let mut tried_quiets: SmallVec<[ChessMove; 32]> = SmallVec::new();
    let mut tried_captures: SmallVec<[(Piece, Square, Piece, ChessMove); 16]> = SmallVec::new();

    let mut move_idx: usize = 0;
    let mut saw_legal_yield = false;
    while let Some(mv) =
        incremental_move_gen.next(&*ctx.history, &*ctx.cap_hist, &*ctx.cont1, &*ctx.cont2)
    {
        saw_legal_yield = true;
        if incremental_move_gen.take_capture_generation_event() {
            ctx.stats.incremental_move_gen_capture_lists += 1;
        }

        // Skip excluded move in verification search for singular extensions
        if exclude_move.is_some() && Some(mv) == exclude_move {
            continue;
        }

        let is_capture =
            board.piece_on(mv.get_dest()).is_some() || is_en_passant_capture(board, mv);
        let is_quiet = !is_capture;
        let is_first_move = move_idx == 0;
        let is_promotion = mv.get_promotion().is_some();

        if is_capture
            && depth_remaining <= SEE_PRUNE_MAX_DEPTH
            && !is_pv_node
            && !in_check
            && !is_first_move
            && static_exchange_eval(board, mv) < 0
        {
            move_idx += 1;
            continue;
        }

        if can_fp
            && !is_capture
            && !is_first_move
            && alpha != NEG_INFINITY
            && beta != POS_INFINITY
            && !is_mate_score(alpha)
            && fp_val <= alpha
        {
            ctx.stats.futility_prunes += 1;
            move_idx += 1;
            continue;
        }

        let new_board = board_do_move(board, mv, ctx.repetition);

        let gives_check = new_board.checkers().0 != 0;
        let child_is_pv_node = is_pv_node && is_first_move;

        let mut extension: i16 = 0;

        // Check if this move gets singular extension
        if Some(mv) == tt_move
            && singular_extension > 0
            && depth_remaining <= SINGULAR_EXTENSION_DEPTH_LIMIT
        {
            extension = cmp::max(extension, singular_extension);
        }

        if gives_check && depth_remaining <= CHECK_EXTENSION_DEPTH_LIMIT && depth_remaining > 0 {
            extension = cmp::max(extension, 1);
        } else if depth_remaining <= PASSED_PAWN_EXTENSION_DEPTH_LIMIT
            && depth_remaining > 0
            && is_passed_pawn_push(board, &new_board, mv)
        {
            extension = cmp::max(extension, 1);
        }

        if !is_pv_node
            && !is_first_move
            && extension == 0
            && !in_check
            && !gives_check
            && !is_capture
            && !is_promotion
            && depth_remaining <= LATE_MOVE_PRUNING_MAX_DEPTH
        {
            let lmp_limit = late_move_pruning_threshold(depth_remaining, is_improving);
            if move_idx >= lmp_limit {
                ctx.stats.late_move_prunes += 1;
                ctx.repetition.pop();
                move_idx += 1;
                continue;
            }
        }

        let apply_lmr =
            depth_remaining >= 3 && move_idx >= 3 && !is_first_move && !is_capture && !gives_check;
        let reduction: i16 = if apply_lmr {
            // History signal for this quiet (butterfly + continuation history).
            let mut hist = ctx.history.score(mover, mv);
            if let Some(p) = board.piece_on(mv.get_source()) {
                let to = mv.get_dest();
                if let Some((pp, pt)) = cont_ctx1 {
                    hist += ctx.cont1.score(pp, pt, p, to);
                }
                if let Some((pp, pt)) = cont_ctx2 {
                    hist += ctx.cont2.score(pp, pt, p, to);
                }
            }
            lmr_reduction(depth_remaining, move_idx, is_pv_node, hist)
        } else {
            0
        };

        let mut use_pvs = !is_first_move;
        let null_window_beta_candidate = alpha.saturating_add(1).min(beta);
        if null_window_beta_candidate <= alpha {
            use_pvs = false;
        }

        let mut depth_after_move = depth_remaining - 1;
        if depth_after_move < 0 {
            depth_after_move = 0;
        }
        let depth_after_move_with_ext = (depth_after_move + extension).max(0);
        let mut reduced_depth = depth_after_move - reduction;
        reduced_depth += extension;
        if reduced_depth < 0 {
            reduced_depth = 0;
        }

        // Track tried moves for batch malus on beta cutoff
        if is_quiet {
            tried_quiets.push(mv);
        } else if is_capture {
            let cap_piece = board.piece_on(mv.get_source());
            let cap_victim = board.piece_on(mv.get_dest()).or_else(|| {
                if is_en_passant_capture(board, mv) {
                    Some(Piece::Pawn)
                } else {
                    None
                }
            });
            if let (Some(piece), Some(victim)) = (cap_piece, cap_victim) {
                tried_captures.push((piece, mv.get_dest(), victim, mv));
            }
        }

        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);
        ctx.search_stack
            .set_move(ply_index, mv, board.piece_on(mv.get_source()));

        let mut value_opt: Option<i32> = None;

        if use_pvs {
            let null_window_beta = null_window_beta_candidate;
            match negamax_it(
                ctx,
                &new_board,
                reduced_depth,
                -null_window_beta,
                -alpha,
                ply_from_root + 1,
                false,
                None,
            ) {
                SearchScore::CANCELLED => {}
                SearchScore::EVAL(v) => {
                    let mut current_value = -v;
                    if current_value > alpha && current_value < beta {
                        if reduction > 0 {
                            ctx.stats.lmr_researches += 1;
                        }

                        match negamax_it(
                            ctx,
                            &new_board,
                            depth_after_move_with_ext,
                            -beta,
                            -alpha,
                            ply_from_root + 1,
                            is_pv_node,
                            None,
                        ) {
                            SearchScore::CANCELLED => {}
                            SearchScore::EVAL(v2) => {
                                current_value = -v2;
                                value_opt = Some(current_value);
                            }
                        }
                    } else {
                        value_opt = Some(current_value);
                    }
                }
            }
        } else {
            match negamax_it(
                ctx,
                &new_board,
                reduced_depth,
                -beta,
                -alpha,
                ply_from_root + 1,
                child_is_pv_node,
                None,
            ) {
                SearchScore::CANCELLED => {}
                SearchScore::EVAL(v) => {
                    value_opt = Some(-v);
                }
            }
        }

        ctx.nnue.pop();

        if value_opt.is_none() {
            ctx.repetition.pop();
            return SearchScore::CANCELLED;
        }

        let value = value_opt.unwrap();

        let was_new_best = value > best_value;
        if was_new_best {
            best_value = value;
            best_move_opt = Some(mv);
        }

        if value >= beta {
            if is_quiet {
                ctx.history.reward(mover, mv, depth_remaining);
                if let Some(p) = board.piece_on(mv.get_source()) {
                    let to = mv.get_dest();
                    if let Some((pp, pt)) = cont_ctx1 {
                        ctx.cont1.reward(pp, pt, p, to, depth_remaining);
                    }
                    if let Some((pp, pt)) = cont_ctx2 {
                        ctx.cont2.reward(pp, pt, p, to, depth_remaining);
                    }
                }
                // Batch penalize all previously tried quiet moves
                for &tried_mv in &tried_quiets {
                    if tried_mv != mv {
                        ctx.history.penalize(mover, tried_mv, depth_remaining);
                        if let Some(p) = board.piece_on(tried_mv.get_source()) {
                            let to = tried_mv.get_dest();
                            if let Some((pp, pt)) = cont_ctx1 {
                                ctx.cont1.penalize(pp, pt, p, to, depth_remaining);
                            }
                            if let Some((pp, pt)) = cont_ctx2 {
                                ctx.cont2.penalize(pp, pt, p, to, depth_remaining);
                            }
                        }
                    }
                }
                // Update countermove: if opponent's last move exists, record this as the best response
                if let Some(prev_move) = ctx.search_stack.get_prev_move(ply_index) {
                    let prev_color = !mover;
                    ctx.countermoves.record(prev_move, prev_color, mv);
                }
            } else if is_capture {
                // Reward capture history on capture beta cutoff
                let cap_piece = board.piece_on(mv.get_source());
                let cap_victim = board.piece_on(mv.get_dest()).or_else(|| {
                    if is_en_passant_capture(board, mv) {
                        Some(Piece::Pawn)
                    } else {
                        None
                    }
                });
                if let (Some(piece), Some(victim)) = (cap_piece, cap_victim) {
                    ctx.cap_hist
                        .reward(mover, piece, mv.get_dest(), victim, depth_remaining);
                }
                // Batch penalize all previously tried captures
                for &(piece, dest, victim, tried_mv) in &tried_captures {
                    if tried_mv != mv {
                        ctx.cap_hist
                            .penalize(mover, piece, dest, victim, depth_remaining);
                    }
                }
            }
            if !is_capture && !gives_check {
                ctx.killers.record(ply_index, mv);
            }
            ctx.stats.beta_cutoffs += 1;
            ctx.repetition.pop();
            break;
        }

        if was_new_best && is_quiet {
            ctx.history.reward_soft(mover, mv, depth_remaining);
            if let Some(p) = board.piece_on(mv.get_source()) {
                let to = mv.get_dest();
                if let Some((pp, pt)) = cont_ctx1 {
                    ctx.cont1.reward_soft(pp, pt, p, to, depth_remaining);
                }
                if let Some((pp, pt)) = cont_ctx2 {
                    ctx.cont2.reward_soft(pp, pt, p, to, depth_remaining);
                }
            }
        }

        if value > alpha {
            alpha = value;
        }

        ctx.repetition.pop();
        move_idx += 1;
    }

    if !saw_legal_yield {
        // No legal moves yielded → mate or stalemate.
        if in_check {
            return SearchScore::EVAL(mated_in_plies(ply_from_root));
        }
        return SearchScore::EVAL(0);
    }

    if best_move_opt.is_none() {
        // In a singular verification search all remaining moves may be pruned/
        // excluded — that's a legitimate "no move beat s_beta" result, not a
        // cancellation. Returning the (low) best_value tells is_singular the
        // candidate is singular.
        if exclude_move.is_some() {
            return SearchScore::EVAL(best_value);
        }
        return SearchScore::CANCELLED;
    }

    let bound = if best_value <= alpha_orig {
        TTFlag::Upper
    } else if best_value >= beta {
        TTFlag::Lower
    } else {
        TTFlag::Exact
    };

    // Don't write the TT or correction history from a singular verification
    // search — its best_value is computed over the position minus one move and
    // would pollute probes/ordering/eval for the real position.
    if exclude_move.is_none() {
        let stored = tt_score_on_store(best_value, ply_from_root);
        ctx.tt.store(
            zob,
            depth_remaining,
            stored,
            bound,
            best_move_opt,
            Some(raw_eval),
        );

        // Correction-history update: only when the static eval is meaningful
        // (not in check, not a mate score) and the search bound is consistent
        // with the direction of the static-vs-search error.
        if !in_check && !is_mate_score(best_value) && !is_mate_score(raw_eval) {
            let update_ok = match bound {
                TTFlag::Exact => true,
                TTFlag::Lower => best_value > eval,
                TTFlag::Upper => best_value < eval,
            };
            if update_ok {
                ctx.corrhist.update(
                    board.side_to_move(),
                    corr_key,
                    best_value - eval,
                    depth_remaining,
                );
            }
        }
    }

    SearchScore::EVAL(best_value)
}

fn log_root_move_start(
    out: &mut Stdout,
    depth: usize,
    mv: &ChessMove,
    current_best: Option<(ChessMove, i32)>,
) {
    let info_line = if let Some((best_move, best_score)) = current_best {
        format!(
            "info string [{}] evaluating move : {}, curr best is {} ({})",
            depth, mv, best_move, best_score
        )
    } else {
        format!("info string [{}] evaluating move : {}", depth, mv)
    };
    send_message(out, &info_line);
}

fn log_root_best_update(
    out: &mut Stdout,
    depth: usize,
    mv: &ChessMove,
    new_score: i32,
    previous_best: Option<(ChessMove, i32)>,
) {
    let info_line = if let Some((prev_move, prev_score)) = previous_best {
        format!(
            "info string [{}] replacing best move {} with {}, {} > {}",
            depth, prev_move, mv, new_score, prev_score
        )
    } else {
        format!(
            "info string [{}] setting initial best move to {}, score {}",
            depth, mv, new_score
        )
    };
    send_message(out, &info_line);
}

struct RootSearchResult {
    best_move: Option<ChessMove>,
    score: i32,
    fail_low: bool,
    fail_high: bool,
    aborted: bool,
    best_move_nodes: u64,
    total_nodes: u64,
    first_move_fail_high: bool,
}

fn collect_principal_variation(
    board: &Board,
    tt: &TranspositionTable,
    max_depth: usize,
) -> Vec<ChessMove> {
    let mut pv = Vec::new();
    let mut current = *board;

    for _ in 0..max_depth {
        let entry = match tt.probe(current.get_hash()) {
            Some(e) => e,
            None => break,
        };

        let mv = match entry.best_move {
            Some(mv) => mv,
            None => break,
        };

        if !current.legal(mv) {
            break;
        }

        pv.push(mv);
        current = current.make_move_new(mv);
    }

    pv
}

/// Execute one root search at a fixed depth using the provided aspiration window.
/// Returns the best move found, the score, and flags indicating fail-high/low outcomes.
fn root_search_with_window(
    board: &Board,
    max_depth: usize,
    transpo_table: &TranspositionTable,
    repetition_table: &mut RepetitionTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    nnue_state: &mut NNUEState,
    history_table: &mut HistoryTable,
    countermove_table: &mut CountermoveTable,
    cap_hist_table: &mut CaptureHistoryTable,
    cont1_table: &mut ContHist,
    cont2_table: &mut ContHist,
    corrhist_table: &mut CorrHist,
    killer_table: &mut KillerTable,
    search_stack: &mut SearchStack,
    alpha_start: i32,
    beta_start: i32,
) -> RootSearchResult {
    stats.record_depth(0);
    let mut alpha = alpha_start;
    let mut beta = beta_start;

    if alpha >= beta {
        alpha = NEG_INFINITY;
        beta = POS_INFINITY;
    }

    let alpha_orig = alpha_start;
    let beta_orig = beta_start;

    let mut best_value = NEG_INFINITY;
    let mut best_move = None;
    let mut aborted = false;

    // Node tracking for time management
    let mut best_move_nodes = 0u64;
    let mut total_nodes = 0u64;
    let mut first_move_fail_high = false;
    let mut move_count = 0usize;

    let mut ctx = ThreadContext {
        tt: transpo_table,
        stop,
        stats,
        repetition: repetition_table,
        killers: killer_table,
        history: history_table,
        countermoves: countermove_table,
        cap_hist: cap_hist_table,
        cont1: cont1_table,
        cont2: cont2_table,
        corrhist: corrhist_table,
        nnue: nnue_state,
        search_stack,
    };
    let mut out = io::stdout();

    let killer_moves = ctx.killers.killers_for(0);
    let mut incremental_move_gen =
        IncrementalMoveGen::new(board, ctx.tt, killer_moves, None, None, None);
    let root_mover = board.side_to_move();

    while let Some(mv) =
        incremental_move_gen.next(&*ctx.history, &*ctx.cap_hist, &*ctx.cont1, &*ctx.cont2)
    {
        if should_stop(ctx.stop) {
            aborted = true;
            break;
        }

        move_count += 1;
        let nodes_before = ctx.stats.nodes;

        let current_best = best_move.map(|bm| (bm, best_value));
        let is_capture =
            board.piece_on(mv.get_dest()).is_some() || is_en_passant_capture(board, mv);
        let is_quiet = !is_capture;
        let new_board = board_do_move(board, mv, ctx.repetition);
        let is_pv_node = best_move.is_none();
        let mut child_depth = max_depth.saturating_sub(1);

        child_depth = child_depth.min(i16::MAX as usize);
        let child_depth_i16 = child_depth as i16;
        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);
        ctx.search_stack
            .set_move(0, mv, board.piece_on(mv.get_source()));

        let result = negamax_it(
            &mut ctx,
            &new_board,
            child_depth_i16,
            -beta,
            -alpha,
            1,
            is_pv_node,
            None,
        );

        ctx.nnue.pop();

        let nodes_spent = ctx.stats.nodes - nodes_before;
        total_nodes += nodes_spent;

        match result {
            SearchScore::CANCELLED => {
                ctx.repetition.pop();
                aborted = true;
                break;
            }
            SearchScore::EVAL(v) => {
                let value = -v;
                let was_new_best = value > best_value;

                if value > best_value {
                    if !QUIET.load(std::sync::atomic::Ordering::Relaxed) {
                        log_root_best_update(&mut out, max_depth, &mv, value, current_best);
                    }
                    best_value = value;
                    best_move = Some(mv);

                    // Transfer node accounting: previous best moves become non-best
                    best_move_nodes = nodes_spent;
                }

                if value > alpha {
                    alpha = value;
                }

                if is_quiet {
                    if value >= beta {
                        ctx.history.reward(root_mover, mv, child_depth_i16);
                    } else if was_new_best {
                        ctx.history.reward_soft(root_mover, mv, child_depth_i16);
                    }
                }

                if alpha >= beta {
                    // First move caused fail-high
                    if move_count == 1 {
                        first_move_fail_high = true;
                    }
                    ctx.repetition.pop();
                    break;
                }

                ctx.repetition.pop();
            }
        }
    }

    let fail_low = !aborted
        && best_move.is_some()
        && alpha_orig != NEG_INFINITY
        && alpha_orig < beta_orig
        && best_value <= alpha_orig;
    let fail_high = !aborted
        && best_move.is_some()
        && beta_orig != POS_INFINITY
        && alpha_orig < beta_orig
        && best_value >= beta_orig;

    if !aborted {
        if let Some(best_mv) = best_move {
            let bound = if fail_low {
                TTFlag::Upper
            } else if fail_high {
                TTFlag::Lower
            } else {
                TTFlag::Exact
            };

            let depth_store = max_depth.min(i16::MAX as usize) as i16;
            let stored_score = tt_score_on_store(best_value, 0);
            ctx.tt.store(
                board.get_hash(),
                depth_store,
                stored_score,
                bound,
                Some(best_mv),
                None,
            );
        }
    }

    RootSearchResult {
        best_move,
        score: best_value,
        fail_low,
        fail_high,
        aborted,
        best_move_nodes,
        total_nodes,
        first_move_fail_high,
    }
}

fn aspiration_root_search(
    board: &Board,
    max_depth: usize,
    transpo_table: &TranspositionTable,
    repetition_table: &mut RepetitionTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    nnue_state: &mut NNUEState,
    prev_score: Option<i32>,
    history_table: &mut HistoryTable,
    countermove_table: &mut CountermoveTable,
    cap_hist_table: &mut CaptureHistoryTable,
    cont1_table: &mut ContHist,
    cont2_table: &mut ContHist,
    corrhist_table: &mut CorrHist,
    killer_table: &mut KillerTable,
    search_stack: &mut SearchStack,
) -> RootSearchResult {
    if prev_score.is_none() {
        return root_search_with_window(
            board,
            max_depth,
            transpo_table,
            repetition_table,
            stop,
            stats,
            nnue_state,
            history_table,
            countermove_table,
            cap_hist_table,
            cont1_table,
            cont2_table,
            corrhist_table,
            killer_table,
            search_stack,
            NEG_INFINITY,
            POS_INFINITY,
        );
    }

    let mut delta = ASPIRATION_START_WINDOW.get();
    let guess = prev_score.unwrap();

    loop {
        let alpha = guess.saturating_sub(delta).max(NEG_INFINITY);
        let beta = guess.saturating_add(delta).min(POS_INFINITY);

        let result = root_search_with_window(
            board,
            max_depth,
            transpo_table,
            repetition_table,
            stop,
            stats,
            nnue_state,
            history_table,
            countermove_table,
            cap_hist_table,
            cont1_table,
            cont2_table,
            corrhist_table,
            killer_table,
            search_stack,
            alpha,
            beta,
        );

        if result.aborted {
            return result;
        }

        if result.fail_low && alpha > NEG_INFINITY {
            delta = delta.saturating_mul(2);
            continue;
        }

        if result.fail_high && beta < POS_INFINITY {
            delta = delta.saturating_mul(2);
            continue;
        }

        return result;
    }
}

struct IterationReport {
    depth: usize,
    result: RootSearchResult,
    #[allow(dead_code)]
    best_move: Option<ChessMove>,
    best_score: i32,
    stats_delta: SearchStats,
    principal_variation: Vec<ChessMove>,
}

struct ThreadContext<'a> {
    tt: &'a TranspositionTable,
    stop: &'a StopFlag,
    stats: &'a mut SearchStats,
    repetition: &'a mut RepetitionTable,
    killers: &'a mut KillerTable,
    history: &'a mut HistoryTable,
    countermoves: &'a mut CountermoveTable,
    cap_hist: &'a mut CaptureHistoryTable,
    cont1: &'a mut ContHist,
    cont2: &'a mut ContHist,
    corrhist: &'a mut CorrHist,
    nnue: &'a mut NNUEState,
    search_stack: &'a mut SearchStack,
}

#[derive(Copy, Clone)]
struct SearchStackEntry {
    eval: i32,
    last_move: Option<ChessMove>,
    last_piece: Option<Piece>,
}

const SEARCH_STACK_SIZE: usize = 256;

struct SearchStack {
    entries: [SearchStackEntry; SEARCH_STACK_SIZE],
}

impl SearchStack {
    fn new() -> Self {
        Self {
            entries: [SearchStackEntry {
                eval: 0,
                last_move: None,
                last_piece: None,
            }; SEARCH_STACK_SIZE],
        }
    }

    fn get(&self, ply: usize) -> &SearchStackEntry {
        &self.entries[ply]
    }

    fn set(&mut self, ply: usize, eval: i32) {
        self.entries[ply].eval = eval;
    }

    fn set_move(&mut self, ply: usize, mv: ChessMove, piece: Option<Piece>) {
        self.entries[ply].last_move = Some(mv);
        self.entries[ply].last_piece = piece;
    }

    /// Mark a ply as having no move (e.g. the child of a null move), so that
    /// continuation-history / countermove lookups don't read a stale sibling.
    fn set_null(&mut self, ply: usize) {
        self.entries[ply].last_move = None;
        self.entries[ply].last_piece = None;
    }

    fn get_prev_move(&self, ply: usize) -> Option<ChessMove> {
        if ply > 0 {
            self.entries[ply - 1].last_move
        } else {
            None
        }
    }

    /// (piece, to) context of the move played `back` plies before the node at
    /// `ply` — i.e. the move recorded in slot `ply - back`. Used to index
    /// continuation history (back=1 → opponent reply, back=2 → our prior move).
    fn cont_context(&self, ply: usize, back: usize) -> Option<(Piece, Square)> {
        if ply >= back {
            let e = &self.entries[ply - back];
            match (e.last_piece, e.last_move) {
                (Some(p), Some(mv)) => Some((p, mv.get_dest())),
                _ => None,
            }
        } else {
            None
        }
    }

    fn is_improving(&self, ply: usize, new_eval: i32) -> bool {
        ply > 1 && new_eval > self.entries[ply - 2].eval
    }
}

/// Reusable per-thread search tables for bulk workloads (rescoring) — avoids
/// reallocating and zeroing ~2.3MB per `best_move` call. Move-ordering history
/// intentionally carries over between positions (ordering/reduction effect
/// only); state that influences returned scores (corrhist) or is strictly
/// per-search (killers, stack) is reset by `reset_for_search`.
pub struct ScratchTables {
    history: HistoryTable,
    countermove: CountermoveTable,
    cap_hist: CaptureHistoryTable,
    cont1: ContHist,
    cont2: ContHist,
    corrhist: CorrHist,
    killers: KillerTable,
    stack: SearchStack,
}

impl ScratchTables {
    fn with_killer_depth(depth: usize) -> Self {
        Self {
            history: HistoryTable::new(),
            countermove: CountermoveTable::new(),
            cap_hist: CaptureHistoryTable::new(),
            cont1: ContHist::new(),
            cont2: ContHist::new(),
            corrhist: CorrHist::new(),
            killers: KillerTable::new(depth),
            stack: SearchStack::new(),
        }
    }

    pub fn new() -> Self {
        Self::with_killer_depth(MAX_SEARCH_PLY as usize + 4)
    }

    fn reset_for_search(&mut self) {
        self.corrhist.clear();
        self.killers.reset_all();
        self.stack = SearchStack::new();
    }
}

/// Reusable full search state (tables + NNUE accumulator stack) for bulk
/// rescoring via `best_move(..., Some(&mut scratch))`.
pub struct SearchScratch {
    tables: ScratchTables,
    nnue: Box<NNUEState>,
}

impl SearchScratch {
    pub fn new(board: &Board) -> Self {
        Self {
            tables: ScratchTables::new(),
            nnue: NNUEState::from_board(board),
        }
    }
}

fn iterative_deepening_loop<F>(
    board: &Board,
    max_depth: usize,
    transpo_table: &TranspositionTable,
    repetition_table: &mut RepetitionTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    nnue_state: &mut NNUEState,
    time_manager: Option<&TimeManagerHandle>,
    scratch: Option<&mut ScratchTables>,
    mut on_iteration: F,
) -> (Option<ChessMove>, i32)
where
    F: FnMut(&IterationReport),
{
    let mut best_move = None;
    let mut best_score = 0;
    let mut prev_score: Option<i32> = None;
    let mut stats_prev = *stats;
    // Fresh tables unless a reusable scratch is provided (bulk rescoring).
    // Killers persist across iterative-deepening depths (was reset per depth).
    let mut fresh: Option<ScratchTables> = None;
    let tabs: &mut ScratchTables = match scratch {
        Some(s) => {
            s.reset_for_search();
            s
        }
        None => fresh.insert(ScratchTables::with_killer_depth(max_depth + 4)),
    };

    // Track best move stability for time management
    let mut best_move_changes = 0usize;
    let mut prev_best_move: Option<ChessMove> = None;

    for depth in 1..=max_depth {
        if should_stop(stop) {
            break;
        }

        let result = aspiration_root_search(
            board,
            depth,
            transpo_table,
            repetition_table,
            stop,
            stats,
            nnue_state,
            prev_score,
            &mut tabs.history,
            &mut tabs.countermove,
            &mut tabs.cap_hist,
            &mut tabs.cont1,
            &mut tabs.cont2,
            &mut tabs.corrhist,
            &mut tabs.killers,
            &mut tabs.stack,
        );

        // Extract values we need before moving result
        let result_aborted = result.aborted;
        let result_score = result.score;
        let result_best_move_nodes = result.best_move_nodes;
        let result_total_nodes = result.total_nodes;
        let result_first_move_fail_high = result.first_move_fail_high;

        // Calculate score trend BEFORE updating prev_score
        let score_trend_for_time =
            TimeScaleFactors::calculate_score_trend_factor(prev_score, result_score);

        if let Some(bm) = result.best_move {
            // Track best move changes for time management
            if let Some(prev) = prev_best_move {
                if prev != bm {
                    best_move_changes += 1;
                }
            }
            prev_best_move = Some(bm);

            best_move = Some(bm);
            best_score = result.score;
            prev_score = Some(result.score);
        }

        let iter_stats = stats.diff(&stats_prev);
        if !result.aborted {
            stats.update_effective_branching_factor(iter_stats.nodes, iter_stats.qnodes, depth);
        }
        stats_prev = *stats;

        let principal_variation = collect_principal_variation(board, transpo_table, max_depth);

        let report = IterationReport {
            depth,
            result,
            best_move,
            best_score,
            stats_delta: iter_stats,
            principal_variation,
        };

        on_iteration(&report);

        if report.result.aborted {
            break;
        }

        // Calculate time scaling factors for intelligent time management
        if let Some(manager) = time_manager {
            let mut factors = TimeScaleFactors::new();

            // Only apply scaling after depth 4 to avoid instability in early search
            if depth >= 4 && !result_aborted {
                factors.set_stability(TimeScaleFactors::calculate_stability_factor(
                    best_move_changes,
                    depth,
                ));
                factors.set_node_fraction(TimeScaleFactors::calculate_node_fraction_factor(
                    result_best_move_nodes,
                    result_total_nodes,
                ));
                factors.set_score_trend(score_trend_for_time);
                factors.set_fail_high_early(result_first_move_fail_high);
            }

            let time_scale = factors.compute_scale();

            if manager.soft_limit_reached_scaled(Instant::now(), time_scale) {
                stop.store(true, Ordering::Relaxed);
                break;
            }
        }
    }

    (best_move, best_score)
}

fn spawn_lazy_smp_helpers(
    board: &Board,
    max_depth: usize,
    transpo_table: &Arc<TranspositionTable>,
    repetition_table: RepetitionTable,
    threads: usize,
    stop: &StopFlag,
) -> Vec<thread::JoinHandle<()>> {
    let worker_count = threads.saturating_sub(1);
    if worker_count == 0 {
        return Vec::new();
    }

    let mut handles = Vec::with_capacity(worker_count);
    for worker_id in 1..=worker_count {
        let board_clone = *board;
        let tt_clone = Arc::clone(transpo_table);
        let stop_clone = stop.clone();
        let mut repetition_clone = repetition_table.clone();
        handles.push(thread::spawn(move || {
            // Light stagger to reduce identical search fronts
            if worker_id > 0 {
                let delay_ms = (worker_id as u64) * 5;
                thread::sleep(Duration::from_millis(delay_ms));
            }

            let mut local_stats = SearchStats::default();
            let mut nnue_state = NNUEState::from_board(&board_clone);
            let _ = iterative_deepening_loop(
                &board_clone,
                max_depth,
                tt_clone.as_ref(),
                &mut repetition_clone,
                &stop_clone,
                &mut local_stats,
                nnue_state.as_mut(),
                None,
                None,
                |_| {},
            );
        }));
    }

    handles
}

fn join_helper_threads(handles: Vec<thread::JoinHandle<()>>) {
    for handle in handles {
        let _ = handle.join();
    }
}

/// Iterative-deepening search that can be optionally bounded by time and/or depth.
/// If `stdout_opt` is `Some(&mut Stdout)` the routine prints Stockfish-compatible
/// "info ..." lines through `send_message` during the search.
#[allow(clippy::too_many_arguments)]
pub fn best_move(
    board: &Board,
    max_depth: usize,
    transpo_table: Arc<TranspositionTable>,
    repetition_table: RepetitionTable,
    time_plan: Option<TimePlan>,
    mut stdout_opt: Option<&mut Stdout>,
    threads: usize,
    scratch: Option<&mut SearchScratch>,
) -> SearchOutcome {
    let stop = Arc::new(AtomicBool::new(false));
    let search_start = Instant::now();
    let time_manager = time_plan
        .as_ref()
        .and_then(|plan| TimeManagerHandle::new(&stop, plan, search_start));

    let mut stats = SearchStats::default();
    // Reuse the caller's scratch NNUE/table state when provided (bulk rescoring).
    let mut fresh_nnue: Option<Box<NNUEState>> = None;
    let (nnue_state, scratch_tables): (&mut NNUEState, Option<&mut ScratchTables>) = match scratch {
        Some(s) => {
            s.nnue.refresh(board);
            (s.nnue.as_mut(), Some(&mut s.tables))
        }
        None => (
            fresh_nnue.insert(NNUEState::from_board(board)).as_mut(),
            None,
        ),
    };
    let mut local_repetition = repetition_table.clone();

    // One TT generation bump per search (not per depth) so that deep entries
    // from earlier iterations are preferred over shallow ones for the whole
    // search, and only entries from previous `go` commands are aged out.
    transpo_table.bump_generation();

    let helper_handles = spawn_lazy_smp_helpers(
        board,
        max_depth,
        &transpo_table,
        repetition_table,
        threads,
        &stop,
    );

    let t0 = Instant::now();
    let (best_move, best_score) = iterative_deepening_loop(
        board,
        max_depth,
        transpo_table.as_ref(),
        &mut local_repetition,
        &stop,
        &mut stats,
        nnue_state,
        time_manager.as_ref(),
        scratch_tables,
        |report| {
            if report.result.aborted {
                return;
            }

            if let Some(out) = stdout_opt.as_mut() {
                let nodes = report.stats_delta.nodes;
                let time_ms = t0.elapsed().as_millis() as u64;
                let nps = if time_ms > 0 {
                    nodes.saturating_mul(1_000) / time_ms
                } else {
                    0
                };

                if !report.principal_variation.is_empty() {
                    let pv_line = report
                        .principal_variation
                        .iter()
                        .map(|mv| mv.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let score_str = uci_score_string(report.best_score, board.side_to_move());
                    let info_line = format!(
                        "info depth {} score {} nodes {} nps {} time {} pv {}",
                        report.depth, score_str, nodes, nps, time_ms, pv_line
                    );
                    send_message(out, &info_line);
                }
            }
        },
    );

    if let Some(out) = stdout_opt.as_mut() {
        let stats_line = format!("info string stats {}", stats.format_as_info());
        send_message(out, &stats_line);
    }

    stop.store(true, Ordering::Relaxed);
    if let Some(manager) = &time_manager {
        manager.cancel();
    }
    join_helper_threads(helper_handles);

    SearchOutcome::new(best_move, best_score, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use std::sync::Arc;

    #[test]
    fn see_favorable_capture() {
        let board = Board::from_str("k7/3p4/8/8/8/8/3R4/7K w - - 0 1").expect("valid FEN");
        let mv = ChessMove::new(Square::D2, Square::D7, None);
        assert!(static_exchange_eval(&board, mv) > 0);
    }

    #[test]
    fn tt_mate_score_roundtrip_fits_i16_and_recovers() {
        for plies in [1, 2, 7, 25, 60, 100] {
            for ply_from_root in [0, 1, 4, 10, 30] {
                for sign in [1i32, -1] {
                    let score = sign * mate_in_plies(plies);
                    let stored = tt_score_on_store(score, ply_from_root);
                    // Must fit in the i16 TT slot without clamping.
                    assert!(
                        stored >= i16::MIN as i32 && stored <= i16::MAX as i32,
                        "stored {} out of i16 range (plies={}, ply={})",
                        stored,
                        plies,
                        ply_from_root
                    );
                    // Simulate the i16 pack/unpack the TT performs.
                    let packed = (stored as i16) as i32;
                    let loaded = tt_score_on_load(packed, ply_from_root);
                    assert_eq!(
                        loaded, score,
                        "mate score not recovered (plies={}, ply={}, sign={})",
                        plies, ply_from_root, sign
                    );
                    // And it must still register as a mate score after the round trip.
                    assert!(is_mate_score(loaded));
                }
            }
        }
        // A normal eval round-trips unchanged.
        for &cp in &[0, 37, -250, 2999, -3000] {
            assert_eq!(
                tt_score_on_load((tt_score_on_store(cp, 5) as i16) as i32, 5),
                cp
            );
        }
    }

    #[test]
    fn see_unfavorable_capture() {
        let board = Board::from_str("k7/8/4p3/3p4/8/8/8/3Q3K w - - 0 1").expect("valid FEN");
        let mv = ChessMove::new(Square::D1, Square::D5, None);
        let see = static_exchange_eval(&board, mv);
        assert!(see < 0, "expected SEE < 0, got {}", see);
    }

    #[test]
    fn regression_no_phantom_mate_on_queen_sac_line() {
        let board =
            Board::from_str("r3kb1r/ppq2ppp/4p1N1/1b1p4/8/2B5/PPPN1PPP/R2Q1RK1 b kq - 0 12")
                .expect("valid FEN");
        let tt = Arc::new(TranspositionTable::new());
        let outcome = best_move(
            &board,
            2,
            Arc::clone(&tt),
            RepetitionTable::new(board.get_hash()),
            None,
            None,
            1,
            None,
        );

        assert!(
            !is_mate_score(outcome.score),
            "expected finite score, got {}",
            outcome.score
        );
        assert!(outcome.best_move.is_some(), "expected a legal best move");
    }
}
