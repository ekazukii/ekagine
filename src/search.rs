use crate::movegen::{
    is_en_passant_capture, piece_value, see_for_sort, static_exchange_eval, IncrementalMoveGen,
};
use crate::nnue::NNUEState;
use crate::tables::{CaptureHistoryTable, CountermoveTable, HistoryTable, KillerTable};
use crate::time::{TimeManagerHandle, TimePlan, TimeScaleFactors};
use crate::{
    board_do_move, board_pop, send_message, RepetitionTable, StopFlag, TTFlag, TranspositionTable,
    NEG_INFINITY, POS_INFINITY,
};
use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_rook_moves, BitBoard, Board,
    BoardStatus, ChessMove, Color, MoveGen, Piece, Square,
};
use lazy_static::lazy_static;
use smallvec::SmallVec;
use std::cmp;
use std::io::Stdout;
use std::marker::PhantomData;
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
            singular_extensions: self.singular_extensions.saturating_sub(other.singular_extensions),
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

// Adjust mate scores when storing/loading from TT to remain comparable
#[inline]
fn tt_score_on_store(score: i32, ply_from_root: i32) -> i32 {
    if is_mate_score(score) {
        if score > 0 {
            score + ply_from_root
        } else {
            score - ply_from_root
        }
    } else {
        score
    }
}

#[inline]
fn tt_score_on_load(score: i32, ply_from_root: i32) -> i32 {
    if is_mate_score(score) {
        if score > 0 {
            score - ply_from_root
        } else {
            score + ply_from_root
        }
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
    debug_assert!(!ctx.repetition.is_in_threefold_scenario(board));

    ctx.stats.record_depth(ply_from_root);
    ctx.stats.qnodes += 1;

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
                if tt_hit.score >= beta {
                    ctx.stats.tt_cutoff_lower += 1;
                    ctx.stats.beta_cutoffs_quiescence += 1;
                    return tt_score;
                }
            }
            TTFlag::Upper => {
                if tt_hit.score <= alpha {
                    ctx.stats.tt_cutoff_upper += 1;
                    return tt_score;
                }
            }
        }
    }

    let static_eval = if in_check {
        None
    } else if let Some(ref tt_hit) = hit {
        Some(
            tt_hit
                .eval
                .unwrap_or_else(|| ctx.nnue.evaluate(board.side_to_move())),
        )
    } else {
        Some(ctx.nnue.evaluate(board.side_to_move()))
    };

    let stand_pat = static_eval.unwrap_or(NEG_INFINITY);

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

    let futility = stand_pat + QUIESCE_FUTILITY_MARGIN;
    for (mv, _val) in get_captures(board) {
        if !in_check && futility <= alpha && see_for_sort(board, mv) < 0 {
            continue;
        }

        let dest_piece = board.piece_on(mv.get_dest());
        let is_en_passant = dest_piece.is_none() && is_en_passant_capture(board, mv);
        if dest_piece.is_none() && !is_en_passant {
            continue;
        }
        let new_board = board_do_move(board, mv, ctx.repetition);
        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);
        let score = -quiesce_negamax_it(ctx, &new_board, -beta, -alpha, ply_from_root + 1);
        board_pop(&new_board, ctx.repetition);
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

const ASPIRATION_START_WINDOW: i32 = 25;
const QUIESCE_FUTILITY_MARGIN: i32 = 200;

const FUTILITY_PRUNE_MAX_DEPTH: i16 = 6;
const FUTILITY_MARGIN_BASE: i32 = 100;
const FUTILITY_MARGIN_PER_DEPTH: i32 = 75;
const REVERSE_FUTILITY_PRUNE_MAX_DEPTH: i16 = 4;

const RAZORING_DEPTH: i16 = 1;
const RAZORING_MARGIN: i32 = 300;

const CHECK_EXTENSION_DEPTH_LIMIT: i16 = 2;
const PASSED_PAWN_EXTENSION_DEPTH_LIMIT: i16 = 6;
const SINGULAR_EXTENSION_MIN_DEPTH: i16 = 8;
const SINGULAR_EXTENSION_DEPTH_LIMIT: i16 = 12;
const SINGULAR_BETA_MARGIN_MULTIPLIER: i32 = 2;
const SEE_PRUNE_MAX_DEPTH: i16 = 6;
const LATE_MOVE_PRUNING_MAX_DEPTH: i16 = 5;
const LATE_MOVE_PRUNING_BASE: usize = 4;
const LATE_MOVE_PRUNING_SCALE: usize = 2;

#[inline]
fn futility_margin(depth: i16) -> i32 {
    let depth = depth as i32;
    FUTILITY_MARGIN_BASE + FUTILITY_MARGIN_PER_DEPTH * depth.saturating_sub(1)
}

#[inline]
fn reverse_futility_margin(depth: i16) -> i32 {
    let depth = depth as i32;
    0 + FUTILITY_MARGIN_PER_DEPTH * depth
}

#[inline]
fn late_move_pruning_threshold(depth: i16) -> usize {
    let depth = depth.max(0) as usize;
    LATE_MOVE_PRUNING_BASE + depth * LATE_MOVE_PRUNING_SCALE
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
            let reduction = LMR_BASE +
                (depth as f64).ln() * (move_idx as f64).ln() / LMR_DIVISOR;
            table[depth][move_idx] = reduction.round() as i16;
        }
    }
    table
}

#[inline]
fn lmr_reduction(depth: i16, move_idx: usize, is_pv_node: bool) -> i16 {
    let d = (depth as usize).min(LMR_MAX_DEPTH - 1);
    let m = move_idx.min(LMR_MAX_MOVES - 1);

    let mut reduction = LMR_TABLE[d][m];

    // Reduce less in PV nodes
    if is_pv_node {
        reduction = reduction.saturating_sub(1);
    }

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
    // 1. Build the capture mask: opponent pieces + possible en passant square
    let mut mask = *board.color_combined(!board.side_to_move());
    if let Some(ep_square) = board.en_passant() {
        mask |= BitBoard::from_square(ep_square);
    }

    // 2. Generate only capture-type moves (filtered by mask)
    let mut gen = MoveGen::new_legal(board);
    gen.set_iterator_mask(mask);

    // 3. Score each move
    let mut scored_moves: SmallVec<[(ChessMove, u8); 64]> = SmallVec::new();

    for mv in gen {
        let score = if let Some(victim_piece) = board.piece_on(mv.get_dest()) {
            // Normal capture
            if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
                let victim_idx = victim_piece.to_index() as usize;
                let attacker_idx = attacker_piece.to_index() as usize;
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
    }

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves
}

// ---------------------------------------------------------------------------
// Singular Extension helpers
// ---------------------------------------------------------------------------

#[inline]
fn should_check_singular_extension(
    depth: i16,
    is_pv_node: bool,
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

    // Only at PV nodes
    if !is_pv_node {
        return false;
    }

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
    let s_beta_margin = SINGULAR_BETA_MARGIN_MULTIPLIER * depth as i32;
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
        s_beta - 1,      // alpha
        s_beta,          // beta (null window)
        ply_from_root,
        false,           // Not a PV node
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
    if ctx.stats.nodes & 2047 == 0 && should_stop(ctx.stop) {
        let mut out = io::stdout();
        let info_line = format!("info string break stop");
        send_message(&mut out, &info_line);
        return SearchScore::CANCELLED;
    }

    ctx.stats.record_depth(ply_from_root);
    ctx.stats.nodes += 1;

    if ctx.repetition.is_in_threefold_scenario(board) {
        return SearchScore::EVAL(0);
    }

    if depth <= 0 {
        return SearchScore::EVAL(quiesce_negamax_it(ctx, board, alpha, beta, ply_from_root));
    }

    let depth_remaining = depth;
    let zob = board.get_hash();

    let tt_hit = ctx.tt.probe(zob);
    if let Some(entry) = tt_hit {
        ctx.stats.tt_hits += 1;
        if entry.depth >= depth_remaining {
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

    let eval = if let Some(hit) = tt_hit {
        if let Some(tt_eval) = hit.eval {
            tt_eval
        } else {
            ctx.nnue.evaluate(board.side_to_move())
        }
    } else {
        ctx.nnue.evaluate(board.side_to_move())
    };
    ctx.search_stack.set(ply_from_root as usize, eval);
    let is_improving = ctx.search_stack.is_improving(ply_from_root as usize, eval);

    let in_check = board.checkers().0 != 0;

    // Razoring: at depth 1, if eval is far below alpha, verify position is bad via qsearch
    if depth_remaining == RAZORING_DEPTH
        && !is_pv_node
        && !in_check
        && alpha > NEG_INFINITY + 1000
        && beta < POS_INFINITY - 1000
        && !is_mate_score(alpha)
        && !is_mate_score(beta)
        && eval + RAZORING_MARGIN < alpha
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
        && has_non_pawn_material(board, board.side_to_move())
    {
        let r: i16 = if depth_remaining >= 6 { 3 } else { 2 } + if is_improving { 1 } else { 0 };
        if let Some(nboard) = board_do_null_move(board, ctx.repetition) {
            ctx.nnue.push();
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
            ctx.nnue.pop();
            match result {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(v) => {
                    let value = -v;
                    if value >= beta {
                        ctx.stats.null_move_prunes += 1;
                        ctx.stats.beta_cutoffs += 1;
                        return SearchScore::EVAL(value);
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
            is_pv_node,
            in_check,
            tt_move,
            entry.depth,
            entry.flag,
            exclude_move,
        ) {
            ctx.stats.singular_checks += 1;
            if let Some(candidate_move) = tt_move {
                if is_singular(ctx, board, candidate_move, depth_remaining, beta, ply_from_root) {
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
    let mut incremental_move_gen =
        IncrementalMoveGen::new(board, ctx.tt, killer_moves, countermove);

    if incremental_move_gen.is_over() {
        if in_check {
            return SearchScore::EVAL(mated_in_plies(ply_from_root));
        }
        return SearchScore::EVAL(0);
    }

    let can_fp = !in_check && depth_remaining <= FUTILITY_PRUNE_MAX_DEPTH;
    let fp_val = if can_fp {
        eval + futility_margin(depth_remaining)
    } else {
        POS_INFINITY
    };

    let mut move_idx: usize = 0;
    while let Some(mv) = incremental_move_gen.next(&*ctx.history, &*ctx.cap_hist) {
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
        {
            if static_exchange_eval(board, mv) < 0 {
                move_idx += 1;
                continue;
            }
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
            && depth_remaining <= SINGULAR_EXTENSION_DEPTH_LIMIT {
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
            let lmp_limit = late_move_pruning_threshold(depth_remaining);
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
            lmr_reduction(depth_remaining, move_idx, is_pv_node)
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

        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);
        ctx.search_stack.set_move(ply_index, mv);

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
                // Update countermove: if opponent's last move exists, record this as the best response
                if let Some(prev_move) = ctx.search_stack.get_prev_move(ply_index) {
                    let prev_color = !mover;
                    ctx.countermoves.record(prev_move, prev_color, mv);
                }
            } else if is_capture {
                // Reward capture history on capture beta cutoff
                let cap_piece = board.piece_on(mv.get_source());
                let cap_victim = board.piece_on(mv.get_dest()).or_else(|| {
                    if is_en_passant_capture(board, mv) { Some(Piece::Pawn) } else { None }
                });
                if let (Some(piece), Some(victim)) = (cap_piece, cap_victim) {
                    ctx.cap_hist.reward(mover, piece, mv.get_dest(), victim, depth_remaining);
                }
            }
            if !is_capture && !gives_check {
                ctx.killers.record(ply_index, mv);
            }
            ctx.stats.beta_cutoffs += 1;
            ctx.repetition.pop();
            break;
        }

        if was_new_best {
            if is_quiet {
                ctx.history.reward_soft(mover, mv, depth_remaining);
            }
        } else if is_quiet {
            ctx.history.penalize(mover, mv, depth_remaining);
        }

        if value > alpha {
            alpha = value;
        }

        ctx.repetition.pop();
        move_idx += 1;
    }

    if best_move_opt.is_none() {
        return SearchScore::CANCELLED;
    }

    let bound = if best_value <= alpha_orig {
        TTFlag::Upper
    } else if best_value >= beta {
        TTFlag::Lower
    } else {
        TTFlag::Exact
    };
    let stored = tt_score_on_store(best_value, ply_from_root);
    ctx.tt.store(
        zob,
        depth_remaining,
        stored,
        bound,
        best_move_opt,
        Some(eval),
    );

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
    let mut current = board.clone();

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

    let mut killer_table = KillerTable::new(max_depth + 4);
    let mut ctx = ThreadContext {
        tt: transpo_table,
        stop,
        stats,
        repetition: repetition_table,
        killers: &mut killer_table,
        history: history_table,
        countermoves: countermove_table,
        cap_hist: cap_hist_table,
        nnue: nnue_state,
        search_stack,
    };
    let mut out = io::stdout();

    let killer_moves = ctx.killers.killers_for(0);
    let mut incremental_move_gen = IncrementalMoveGen::new(board, ctx.tt, killer_moves, None);
    let root_mover = board.side_to_move();

    while let Some(mv) = incremental_move_gen.next(&*ctx.history, &*ctx.cap_hist) {
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
                    log_root_best_update(&mut out, max_depth, &mv, value, current_best);
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
                    } else {
                        ctx.history.penalize(root_mover, mv, child_depth_i16);
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
            search_stack,
            NEG_INFINITY,
            POS_INFINITY,
        );
    }

    let mut delta = ASPIRATION_START_WINDOW;
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
    nnue: &'a mut NNUEState,
    search_stack: &'a mut SearchStack,
}

#[derive(Copy, Clone)]
struct SearchStackEntry {
    eval: i32,
    last_move: Option<ChessMove>,
}

struct SearchStack {
    entries: [SearchStackEntry; 64],
}

impl SearchStack {
    fn new() -> Self {
        Self {
            entries: [SearchStackEntry {
                eval: 0,
                last_move: None,
            }; 64],
        }
    }

    fn get(&self, ply: usize) -> &SearchStackEntry {
        &self.entries[ply]
    }

    fn set(&mut self, ply: usize, eval: i32) {
        self.entries[ply].eval = eval;
    }

    fn set_move(&mut self, ply: usize, mv: ChessMove) {
        self.entries[ply].last_move = Some(mv);
    }

    fn get_prev_move(&self, ply: usize) -> Option<ChessMove> {
        if ply > 0 {
            self.entries[ply - 1].last_move
        } else {
            None
        }
    }

    fn is_improving(&self, ply: usize, new_eval: i32) -> bool {
        ply > 1 && new_eval > self.entries[ply - 2].eval
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
    mut on_iteration: F,
) -> (Option<ChessMove>, i32)
where
    F: FnMut(&IterationReport),
{
    let mut best_move = None;
    let mut best_score = 0;
    let mut prev_score: Option<i32> = None;
    let mut stats_prev = *stats;
    let mut history_table = HistoryTable::new();
    let mut countermove_table = CountermoveTable::new();
    let mut cap_hist_table = CaptureHistoryTable::new();
    let mut search_stack = SearchStack::new();

    // Track best move stability for time management
    let mut best_move_changes = 0usize;
    let mut prev_best_move: Option<ChessMove> = None;

    for depth in 1..=max_depth {
        if should_stop(stop) {
            break;
        }

        transpo_table.bump_generation();
        nnue_state.refresh(board);

        let result = aspiration_root_search(
            board,
            depth,
            transpo_table,
            repetition_table,
            stop,
            stats,
            nnue_state,
            prev_score,
            &mut history_table,
            &mut countermove_table,
            &mut cap_hist_table,
            &mut search_stack,
        );

        // Extract values we need before moving result
        let result_aborted = result.aborted;
        let result_score = result.score;
        let result_best_move_nodes = result.best_move_nodes;
        let result_total_nodes = result.total_nodes;
        let result_first_move_fail_high = result.first_move_fail_high;

        // Calculate score trend BEFORE updating prev_score
        let score_trend_for_time = TimeScaleFactors::calculate_score_trend_factor(
            prev_score,
            result_score,
        );

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
        let board_clone = board.clone();
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
) -> SearchOutcome {
    let stop = Arc::new(AtomicBool::new(false));
    let search_start = Instant::now();
    let time_manager = time_plan
        .as_ref()
        .and_then(|plan| TimeManagerHandle::new(&stop, plan, search_start));

    let mut stats = SearchStats::default();
    let mut nnue_state = NNUEState::from_board(board);
    let mut local_repetition = repetition_table.clone();

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
        nnue_state.as_mut(),
        time_manager.as_ref(),
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
                    send_message(&mut **out, &info_line);
                }
            }
        },
    );

    if let Some(out) = stdout_opt.as_mut() {
        let stats_line = format!("info string stats {}", stats.format_as_info());
        send_message(&mut **out, &stats_line);
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

    #[test]
    fn see_favorable_capture() {
        let board = Board::from_str("k7/3p4/8/8/8/8/3R4/7K w - - 0 1").expect("valid FEN");
        let mv = ChessMove::new(Square::D2, Square::D7, None);
        assert!(static_exchange_eval(&board, mv) > 0);
    }

    #[test]
    fn see_unfavorable_capture() {
        let board = Board::from_str("k7/8/4p3/3p4/8/8/8/3Q3K w - - 0 1").expect("valid FEN");
        let mv = ChessMove::new(Square::D1, Square::D5, None);
        let see = static_exchange_eval(&board, mv);
        assert!(see < 0, "expected SEE < 0, got {}", see);
    }
}
