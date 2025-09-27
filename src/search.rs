use crate::eval::eval_board;
use crate::{
    board_do_move, board_pop, send_message, PVTable, RepetitionTable, StopFlag, TTFlag,
    TranspositionTable, CACHE_COUNT, DEPTH_COUNT, EVAL_COUNT, NEG_INFINITY, POS_INFINITY,
    QUIESCE_REMAIN,
};
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen, Piece};
use std::io::Stdout;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{io, thread};

/* For reference
type TranspositionTable = HashMap<u64, i32>;
type RepetitionTable    = HashMap<u64, usize>;
type PVTable = HashMap<u64, ChessMove>;
 */

enum SearchScore {
    CANCELLED,
    //MATE(u8),
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
    pub lmr_researches: u64,
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
            lmr_researches: self.lmr_researches.saturating_sub(other.lmr_researches),
        }
    }

    fn format_as_info(&self) -> String {
        format!(
            "nodes={} qnodes={} tt_hits={} tt_exact={} tt_lower={} tt_upper={} beta_cut={} qbeta_cut={} null_prune={} futility_prune={} lmr_retry={}",
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
            self.lmr_researches,
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
    board: &Board,
    mut alpha: i32,
    beta: i32,
    remain_quiet: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    color: i32, // +1 if White to move in this node, −1 otherwise
    ply_from_root: i32,
    stats: &mut SearchStats,
) -> i32 {
    stats.qnodes += 1;
    match board.status() {
        BoardStatus::Checkmate => return mated_in_plies(ply_from_root),
        BoardStatus::Stalemate => return 0,
        BoardStatus::Ongoing => {}
    }

    if remain_quiet == 0 || is_in_threefold_scenario(board, repetition_table) {
        return color * cache_eval(board, transpo_table, repetition_table);
    }

    let stand_pat = color * cache_eval(board, transpo_table, repetition_table);
    if stand_pat >= beta {
        stats.beta_cutoffs_quiescence += 1;
        return stand_pat;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    for mv in sort_moves(board) {
        if board.piece_on(mv.get_dest()).is_none() && !is_en_passant_capture(board, mv) {
            continue;
        }
        let new_board = board_do_move(board, mv, repetition_table);
        let score = -quiesce_negamax_it(
            &new_board,
            -beta,
            -alpha,
            remain_quiet - 1,
            transpo_table,
            repetition_table,
            -color,
            ply_from_root + 1,
            stats,
        );
        board_pop(&new_board, repetition_table);

        if score >= beta {
            stats.beta_cutoffs_quiescence += 1;
            return score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Check if this position's hash has occurred more than once already.
/// For simplicity we'll make so that repeating once is the same as repeating twice.
/// This should not impact the evaluation of engine since repeating a move once or twice ends up anyway
/// in the same position
fn is_in_threefold_scenario(board: &Board, repetition_table: &RepetitionTable) -> bool {
    let zob = board.get_hash();
    repetition_table.get(&zob).cloned().unwrap_or(0) > 1
}

/// Cached evaluation: if threefold, return 0, else look up in transposition table.
/// If not found, compute via `eval_board`, insert into the table, and return.
fn cache_eval(
    board: &Board,
    transpo_table: &mut TranspositionTable,
    repetition_table: &RepetitionTable,
) -> i32 {
    if is_in_threefold_scenario(board, repetition_table) {
        return 0;
    }
    let zob = board.get_hash();
    if let Some(entry) = transpo_table.probe(zob) {
        if let Some(cached) = entry.eval {
            CACHE_COUNT.fetch_add(1, Ordering::Relaxed);
            return cached;
        }
    }
    let val = eval_board(board);
    transpo_table.store_eval(zob, val);
    val
}

#[inline]
fn has_non_pawn_material(board: &Board, side: Color) -> bool {
    let bb = board.color_combined(side);
    let non_pawns = (*board.pieces(Piece::Knight)
        | *board.pieces(Piece::Bishop)
        | *board.pieces(Piece::Rook)
        | *board.pieces(Piece::Queen))
        & bb;
    non_pawns.popcnt() > 0
}

/// Create a new board representing a null move using Board::null_move(),
/// then update the repetition table for that new position.
fn board_do_null_move(board: &Board, repetition_table: &mut RepetitionTable) -> Option<Board> {
    // Use the chess crate's built-in null move generator
    let new_board = board.null_move()?;
    let zob = new_board.get_hash();
    *repetition_table.entry(zob).or_insert(0) += 1;
    Some(new_board)
}

struct AppliedBoard<'a> {
    board: Board,
    repetition_table: *mut RepetitionTable,
    popped: bool,
    _marker: PhantomData<&'a mut RepetitionTable>,
}

impl<'a> AppliedBoard<'a> {
    fn new(parent: &Board, mv: ChessMove, repetition_table: &'a mut RepetitionTable) -> Self {
        let board = board_do_move(parent, mv, repetition_table);
        Self {
            board,
            repetition_table,
            popped: false,
            _marker: PhantomData,
        }
    }

    fn board(&self) -> &Board {
        &self.board
    }

    fn split(&mut self) -> (&Board, &mut RepetitionTable) {
        let board_ptr: *const Board = &self.board;
        let rep_ptr = self.repetition_table;
        unsafe { (&*board_ptr, &mut *rep_ptr) }
    }

    fn mark_popped(&mut self) {
        if !self.popped {
            unsafe {
                board_pop(&self.board, &mut *self.repetition_table);
            }
            self.popped = true;
        }
    }
}

impl<'a> Drop for AppliedBoard<'a> {
    fn drop(&mut self) {
        self.mark_popped();
    }
}

const CHECK_BONUS: u8 = 60;
const CHECKMATE_BONUS: u8 = 100;
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

const ASPIRATION_START_WINDOW: i32 = 50;

const FUTILITY_PRUNE_MAX_DEPTH: i16 = 3;
const FUTILITY_MARGIN_BASE: i32 = 200;
const FUTILITY_MARGIN_PER_DEPTH: i32 = 150;

const CHECK_EXTENSION: usize = 1;
const CHECK_EXTENSION_DEPTH_LIMIT: i16 = 2;

#[inline]
fn futility_margin(depth: i16) -> i32 {
    let depth = depth as i32;
    FUTILITY_MARGIN_BASE + FUTILITY_MARGIN_PER_DEPTH * depth.saturating_sub(1)
}

#[inline]
fn is_en_passant_capture(board: &Board, mv: ChessMove) -> bool {
    // En passant is a pawn move to the ep square capturing a pawn that isn't on dest.
    // chess::Board exposes the en-passant target square via `en_passant()` in recent versions.
    // We conservatively detect: moving piece is a pawn, destination equals ep square.
    if let Some(Piece::Pawn) = board.piece_on(mv.get_source()) {
        if let Some(ep_sq) = board.en_passant() {
            return mv.get_dest() == ep_sq;
        }
    }
    false
}

/// This functions orders the move over multiple criteria so that the best probable moves are ordered
/// on top of the search list.
/// This function is meant to be the base sorter used by the quiesce search and the base of the ordering
/// for the "classic search"
fn sort_moves(board: &Board) -> Vec<ChessMove> {
    let mut scored_moves: Vec<(ChessMove, u8)> = Vec::new();

    for mv in MoveGen::new_legal(board) {
        let score = if let Some(victim_piece) = board.piece_on(mv.get_dest()) {
            // It's a capture; find the attacker piece type
            if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
                let victim_idx = victim_piece.to_index() as usize;
                let attacker_idx = attacker_piece.to_index() as usize;
                MVV_LVA_TABLE[victim_idx][attacker_idx]
            } else {
                0
            }
        } else if is_en_passant_capture(board, mv) {
            25
        } else {
            0
        };

        let board_after = board.make_move_new(mv);

        let check_bonus = if board_after.checkers().popcnt() > 0 {
            // match board_after.status() {
            //    BoardStatus::Checkmate => {
            //        CHECKMATE_BONUS
            //    }
            //    _ => CHECK_BONUS,
            //}
            CHECK_BONUS
        } else {
            0
        };

        scored_moves.push((mv, score + check_bonus));

        //scored_moves.push((mv, score));
    }

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

/// This functions orders to move based on the first sort of `sort_moves` but adds specific order
/// criteria for the non-quiesce search.
/// Such as taking moves from the transposition table and principal variation
fn ordered_moves(
    board: &Board,
    pv_table: &PVTable,
    tt: &TranspositionTable,
    _depth: usize,
) -> Vec<ChessMove> {
    let mut moves = sort_moves(board);

    let zob = board.get_hash();
    // Promote TT move first if present
    if let Some(tt_mv) = tt.probe(zob).and_then(|e| e.best_move) {
        if let Some(pos) = moves.iter().position(|m| *m == tt_mv) {
            let mv = moves.remove(pos);
            moves.insert(0, mv);
        }
    }
    // Then promote PV move (if different from TT move)
    if let Some(pv_mv) = pv_table.get(&zob) {
        if let Some(pos) = moves.iter().position(|m| m == pv_mv) {
            let mv = moves.remove(pos);
            moves.insert(0, mv);
        }
    }

    moves
}

// ---------------------------------------------------------------------------
// Core Negamax with α/β pruning + PV maintenance — renamed `negamax_it`.
// ---------------------------------------------------------------------------
/// This is the core of the move search, it perform a negamax styled search with alpha/beta pruning
/// Usage of quiesce search at end of search and tt/pv cutoff
fn negamax_it(
    board: &Board,
    depth: usize,
    max_depth: usize,
    mut alpha: i32,
    beta: i32,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    color: i32,
    stop: &StopFlag,
    ply_from_root: i32,
    stats: &mut SearchStats,
) -> SearchScore {
    // If time budget expired
    if should_stop(stop) {
        let mut out = io::stdout();
        let info_line = format!("info string break stop");
        send_message(&mut out, &info_line);
        return SearchScore::CANCELLED;
    }

    stats.nodes += 1;

    // If game is over
    if is_in_threefold_scenario(board, repetition_table) {
        return SearchScore::EVAL(0);
    }
    match board.status() {
        BoardStatus::Checkmate => return SearchScore::EVAL(mated_in_plies(ply_from_root)),
        BoardStatus::Stalemate => return SearchScore::EVAL(0),
        BoardStatus::Ongoing => {}
    }

    // Start quiesce search if we reached or exceeded the depth limit
    if depth >= max_depth {
        return SearchScore::EVAL(quiesce_negamax_it(
            board,
            alpha,
            beta,
            QUIESCE_REMAIN,
            transpo_table,
            repetition_table,
            color,
            ply_from_root,
            stats,
        ));
    }

    // Probe TT for cutoff
    let rem_depth: i16 = max_depth.saturating_sub(depth) as i16;
    let zob = board.get_hash();
    if let Some(entry) = transpo_table.probe(zob) {
        stats.tt_hits += 1;
        // Only keep if entry depth is same or higher to avoid pruning from not fully search position
        if entry.depth >= rem_depth {
            let tt_score = tt_score_on_load(entry.score, ply_from_root);
            match entry.flag {
                TTFlag::Exact => {
                    stats.tt_cutoff_exact += 1;
                    return SearchScore::EVAL(tt_score);
                }
                TTFlag::Lower => {
                    if tt_score >= beta {
                        stats.tt_cutoff_lower += 1;
                        stats.beta_cutoffs += 1;
                        return SearchScore::EVAL(tt_score);
                    }
                }
                TTFlag::Upper => {
                    if tt_score <= alpha {
                        stats.tt_cutoff_upper += 1;
                        return SearchScore::EVAL(tt_score);
                    }
                }
            }
        }
    }

    // Null-move pruning: if not in check and enough depth, try a null move with reduction.
    // Guard against zugzwang: require some non-pawn material for the side to move.
    let in_check = board.checkers().popcnt() > 0;
    if !in_check && rem_depth >= 2 && has_non_pawn_material(board, board.side_to_move()) {
        let r = if rem_depth >= 6 { 3 } else { 2 } as usize;
        if let Some(nboard) = board_do_null_move(board, repetition_table) {
            // null-window search (fail-high test)
            match negamax_it(
                &nboard,
                depth + 1 + r,
                max_depth,
                -beta,
                -beta + 1,
                transpo_table,
                repetition_table,
                pv_table,
                -color,
                stop,
                ply_from_root + 1,
                stats,
            ) {
                SearchScore::CANCELLED => {
                    board_pop(&nboard, repetition_table);
                    return SearchScore::CANCELLED;
                }
                SearchScore::EVAL(v) => {
                    let value = -v;
                    board_pop(&nboard, repetition_table);
                    if value >= beta {
                        stats.null_move_prunes += 1;
                        stats.beta_cutoffs += 1;
                        return SearchScore::EVAL(value);
                    }
                }
            }
        }
    }

    let mut best_value = NEG_INFINITY;
    let mut best_move_opt: Option<ChessMove> = None;
    let alpha_orig = alpha;

    let static_eval_for_futility = if !in_check && rem_depth <= FUTILITY_PRUNE_MAX_DEPTH {
        Some(color * cache_eval(board, transpo_table, repetition_table))
    } else {
        None
    };

    // Start the main search
    let mut move_idx: usize = 0;
    for mv in ordered_moves(board, pv_table, transpo_table, depth) {
        let mut applied = AppliedBoard::new(board, mv, repetition_table);

        // Determine properties for LMR conditions
        let is_capture =
            board.piece_on(mv.get_dest()).is_some() || is_en_passant_capture(board, mv);
        let gives_check = applied.board().checkers().popcnt() > 0;
        let is_pv_move = move_idx == 0; // treat first move as PV
        let extension = if gives_check && rem_depth <= CHECK_EXTENSION_DEPTH_LIMIT && rem_depth > 0
        {
            CHECK_EXTENSION
        } else {
            0
        };

        if let Some(stand_pat) = static_eval_for_futility {
            if rem_depth <= FUTILITY_PRUNE_MAX_DEPTH
                && !is_capture
                && !gives_check
                && !is_pv_move
                && alpha != NEG_INFINITY
                && beta != POS_INFINITY
                && !is_mate_score(alpha)
            {
                let margin = futility_margin(rem_depth);
                if stand_pat + margin <= alpha {
                    stats.futility_prunes += 1;
                    move_idx += 1;
                    continue;
                }
            }
        }

        // Apply LMR for late, quiet, non-check, non-PV moves with sufficient remaining depth
        let apply_lmr =
            rem_depth >= 3 && move_idx >= 3 && !is_pv_move && !is_capture && !gives_check;
        let value = if apply_lmr {
            let r = if rem_depth >= 6 { 2 } else { 1 } as usize;
            // Reduced-depth null-window search first
            let (child_board_view, rep_table) = applied.split();
            match negamax_it(
                child_board_view,
                depth + 1 + r,
                max_depth + extension,
                -beta,
                -alpha,
                transpo_table,
                rep_table,
                pv_table,
                -color,
                stop,
                ply_from_root + 1,
                stats,
            ) {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(vr) => {
                    let vred = -vr;
                    // If reduced search raises alpha, re-search at full depth
                    if vred > alpha {
                        stats.lmr_researches += 1;
                        let (child_board_view, rep_table) = applied.split();
                        match negamax_it(
                            child_board_view,
                            depth + 1,
                            max_depth + extension,
                            -beta,
                            -alpha,
                            transpo_table,
                            rep_table,
                            pv_table,
                            -color,
                            stop,
                            ply_from_root + 1,
                            stats,
                        ) {
                            SearchScore::CANCELLED => return SearchScore::CANCELLED,
                            SearchScore::EVAL(vf) => -vf,
                        }
                    } else {
                        vred
                    }
                }
            }
        } else {
            // Normal full-depth search
            let (child_board_view, rep_table) = applied.split();
            match negamax_it(
                child_board_view,
                depth + 1,
                max_depth + extension,
                -beta,
                -alpha,
                transpo_table,
                rep_table,
                pv_table,
                -color,
                stop,
                ply_from_root + 1,
                stats,
            ) {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(v) => -v,
            }
        };

        // Update best / alpha-beta
        if value > best_value {
            best_value = value;
            best_move_opt = Some(mv);
        }
        if value >= beta {
            stats.beta_cutoffs += 1;
            break;
        }
        if value > alpha {
            alpha = value;
        }

        move_idx += 1;
    }

    if best_move_opt.is_some() {
        pv_table.insert(board.get_hash(), best_move_opt.unwrap());
    } else {
        return SearchScore::CANCELLED;
    }

    // If it would have cause an alpha beta pruning store the information in the TT table
    let bound = if best_value <= alpha_orig {
        TTFlag::Upper
    } else if best_value >= beta {
        TTFlag::Lower
    } else {
        TTFlag::Exact
    };
    let stored = tt_score_on_store(best_value, ply_from_root);
    let prev_eval = transpo_table.probe(zob).and_then(|e| e.eval);
    transpo_table.store(zob, rem_depth, stored, bound, best_move_opt, prev_eval);

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
}

/// Execute one root search at a fixed depth using the provided aspiration window.
/// Returns the best move found, the score, and flags indicating fail-high/low outcomes.
fn root_search_with_window(
    board: &Board,
    max_depth: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    alpha_start: i32,
    beta_start: i32,
) -> RootSearchResult {
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
    let color = if board.side_to_move() == Color::White {
        1
    } else {
        -1
    };

    let mut out = io::stdout();

    for mv in ordered_moves(board, pv_table, transpo_table, 0) {
        if should_stop(stop) {
            aborted = true;
            break;
        }

        let current_best = best_move.map(|bm| (bm, best_value));
        log_root_move_start(&mut out, max_depth, &mv, current_best);

        let mut applied = AppliedBoard::new(board, mv, repetition_table);
        let (child_board, rep_table) = applied.split();
        match negamax_it(
            child_board,
            1,
            max_depth,
            -beta,
            -alpha,
            transpo_table,
            rep_table,
            pv_table,
            -color,
            stop,
            1, // one ply from root after making mv
            stats,
        ) {
            SearchScore::CANCELLED => {
                aborted = true;
                break;
            }
            SearchScore::EVAL(v) => {
                let value = -v;

                if value > best_value {
                    log_root_best_update(&mut out, max_depth, &mv, value, current_best);
                    best_value = value;
                    best_move = Some(mv);
                }
                if value > alpha {
                    alpha = value;
                }
                if alpha >= beta {
                    break;
                }
            }
        }
    }

    if best_move.is_some() {
        pv_table.insert(board.get_hash(), best_move.unwrap());
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

    RootSearchResult {
        best_move,
        score: best_value,
        fail_low,
        fail_high,
        aborted,
    }
}

fn aspiration_root_search(
    board: &Board,
    max_depth: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    prev_score: Option<i32>,
) -> RootSearchResult {
    if prev_score.is_none() {
        return root_search_with_window(
            board,
            max_depth,
            transpo_table,
            repetition_table,
            pv_table,
            stop,
            stats,
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
            pv_table,
            stop,
            stats,
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
}

fn iterative_deepening_loop<F>(
    board: &Board,
    max_depth: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    mut on_iteration: F,
) -> (Option<ChessMove>, i32)
where
    F: FnMut(&IterationReport, &PVTable),
{
    let mut best_move = None;
    let mut best_score = 0;
    let mut prev_score: Option<i32> = None;
    let mut stats_prev = *stats;

    for depth in 1..=max_depth {
        if should_stop(stop) {
            break;
        }

        transpo_table.bump_generation();

        let result = aspiration_root_search(
            board,
            depth,
            transpo_table,
            repetition_table,
            pv_table,
            stop,
            stats,
            prev_score,
        );

        if let Some(bm) = result.best_move {
            best_move = Some(bm);
            best_score = result.score;
            prev_score = Some(result.score);
        }

        let iter_stats = stats.diff(&stats_prev);
        stats_prev = *stats;

        let report = IterationReport {
            depth,
            result,
            best_move,
            best_score,
            stats_delta: iter_stats,
        };

        on_iteration(&report, &*pv_table);

        if report.result.aborted {
            break;
        }
    }

    (best_move, best_score)
}

/// Compute the best move of a board from a depth limit
#[allow(clippy::too_many_arguments)]
pub fn best_move_using_iterative_deepening(
    board: &Board,
    max_depth: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
) -> SearchOutcome {
    let mut pv_table: PVTable = PVTable::default();
    let stop = Arc::new(AtomicBool::new(false));
    let mut stats = SearchStats::default();

    let (best_move, best_score) = iterative_deepening_loop(
        board,
        max_depth,
        transpo_table,
        repetition_table,
        &mut pv_table,
        &stop,
        &mut stats,
        |_, _| {},
    );

    SearchOutcome::new(best_move, best_score, stats)
}

/// Iterative-deepening search that can be stopped by a time budget.
/// If `stdout_opt` is `Some(&mut Stdout)` the routine prints Stockfish-
/// compatible "info ..." lines through your existing `send_message`.
///
/// * `board`          – starting position
/// * `time_budget`    – wall-clock budget (hard stop)
/// * `max_depth_cap`  – fail-safe depth limit (e.g. 99)
/// * `stdout_opt`     – `Some(&mut std::io::Stdout)` to enable printing /
///                      logging, or `None` to run silently.
///
/// Returns `(best_move, score_in_cp)`
pub fn best_move_interruptible(
    board: &Board,
    time_budget: Duration,
    max_depth_cap: usize,
    mut repetition_table: RepetitionTable,
    transpo_table: &mut TranspositionTable,
    mut stdout_opt: Option<&mut Stdout>,
) -> SearchOutcome {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop_clone = stop.clone();
        thread::spawn(move || {
            thread::sleep(time_budget);
            stop_clone.store(true, Ordering::Relaxed);
        });
    }

    let mut pv_table = PVTable::default();
    let mut stats = SearchStats::default();

    let t0 = Instant::now();
    EVAL_COUNT.store(0, Ordering::Relaxed);
    DEPTH_COUNT.store(0, Ordering::Relaxed);

    let (best_move, best_score) = iterative_deepening_loop(
        board,
        max_depth_cap,
        transpo_table,
        &mut repetition_table,
        &mut pv_table,
        &stop,
        &mut stats,
        |report, pv_view| {
            if report.result.aborted {
                return;
            }

            DEPTH_COUNT.store(report.depth, Ordering::Relaxed);

            if let Some(out) = stdout_opt.as_mut() {
                let nodes = report.stats_delta.nodes;
                let time_ms = t0.elapsed().as_millis() as u64;
                let nps = if time_ms > 0 {
                    nodes * 1_000 / time_ms
                } else {
                    0
                };

                if let Some(pv_mv) = pv_view.get(&board.get_hash()) {
                    let score_str = uci_score_string(report.best_score, board.side_to_move());
                    let info_line = format!(
                        "info depth {} score {} nodes {} nps {} time {} pv {}",
                        report.depth, score_str, nodes, nps, time_ms, pv_mv
                    );
                    send_message(&mut **out, &info_line);
                }
            }

            EVAL_COUNT.store(0, Ordering::Relaxed);
        },
    );

    if let Some(out) = stdout_opt.as_mut() {
        let stats_line = format!("info string stats {}", stats.format_as_info());
        send_message(&mut **out, &stats_line);
    }

    SearchOutcome::new(best_move, best_score, stats)
}
