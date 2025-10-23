use crate::movegen::IncrementalMoveGen;
use crate::nnue::NNUEState;
use crate::{
    board_do_move, board_pop, send_message, RepetitionTable, StopFlag, TTFlag, TranspositionTable,
    NEG_INFINITY, POS_INFINITY,
};
use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_rook_moves, BitBoard, Board,
    BoardStatus, ChessMove, Color, MoveGen, Piece, Square,
};
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
    pub lmr_researches: u64,
    pub incremental_move_gen_inits: u64,
    pub incremental_move_gen_capture_lists: u64,
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
            lmr_researches: self.lmr_researches.saturating_sub(other.lmr_researches),
            incremental_move_gen_inits: self
                .incremental_move_gen_inits
                .saturating_sub(other.incremental_move_gen_inits),
            incremental_move_gen_capture_lists: self
                .incremental_move_gen_capture_lists
                .saturating_sub(other.incremental_move_gen_capture_lists),
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
            "nodes={} qnodes={} tt_hits={} tt_exact={} tt_lower={} tt_upper={} beta_cut={} qbeta_cut={} null_prune={} futility_prune={} rfutility_prune={} lmr_retry={} img_init={} img_capgen={} ebf={:.2} depth={}",
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
            self.lmr_researches,
            self.incremental_move_gen_inits,
            self.incremental_move_gen_capture_lists,
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

const MAX_KILLER_MOVES: usize = 2;

#[derive(Debug, Clone)]
struct KillerTable {
    moves: Vec<[Option<ChessMove>; MAX_KILLER_MOVES]>,
}

impl KillerTable {
    fn new(initial_depth: usize) -> Self {
        let depth = initial_depth.max(1);
        KillerTable {
            moves: vec![[None; MAX_KILLER_MOVES]; depth],
        }
    }

    fn ensure_capacity(&mut self, ply: usize) {
        if ply >= self.moves.len() {
            self.moves.resize(ply + 1, [None; MAX_KILLER_MOVES]);
        }
    }

    fn killers_for(&mut self, ply: usize) -> [Option<ChessMove>; MAX_KILLER_MOVES] {
        self.ensure_capacity(ply);
        self.moves[ply]
    }

    fn record(&mut self, ply: usize, mv: ChessMove) {
        self.ensure_capacity(ply);
        let entry = &mut self.moves[ply];
        if entry.iter().any(|existing| *existing == Some(mv)) {
            if entry[1] == Some(mv) {
                entry.swap(0, 1);
            }
            return;
        }
        entry[1] = entry[0];
        entry[0] = Some(mv);
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

    // Quiescence TT lookup
    let hit = ctx.tt.probe(board.get_hash());
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

    let in_check = board.checkers().popcnt() > 0;
    let stand_pat = if in_check {
        NEG_INFINITY
    } else if let Some(ref tt_hit) = hit {
        tt_hit
            .eval
            .unwrap_or_else(|| ctx.nnue.evaluate(board.side_to_move()))
    } else {
        ctx.nnue.evaluate(board.side_to_move())
    };

    if stand_pat >= beta {
        ctx.stats.beta_cutoffs_quiescence += 1;
        return stand_pat;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    for (mv, _val) in get_captures(board) {
        let dest_piece = board.piece_on(mv.get_dest());
        let is_en_passant = dest_piece.is_none() && is_en_passant_capture(board, mv);
        if dest_piece.is_none() && !is_en_passant {
            continue;
        }
        let captured_val = dest_piece
            .map(piece_value)
            .unwrap_or_else(|| piece_value(Piece::Pawn));
        if !in_check && stand_pat + captured_val + QUIESCE_FUTILITY_MARGIN <= alpha {
            continue;
        }
        if see_for_sort(board, mv) < 0 {
            continue;
        }
        let new_board = board_do_move(board, mv, ctx.repetition);
        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);
        let score = -quiesce_negamax_it(
            ctx,
            &new_board,
            -beta,
            -alpha,
            ply_from_root + 1,
        );
        board_pop(&new_board, ctx.repetition);
        ctx.nnue.pop();

        if score >= beta {
            ctx.stats.beta_cutoffs_quiescence += 1;
            return score;
        }
        if score > alpha {
            alpha = score;
        }
    }

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
    non_pawns.popcnt() > 0
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

const ASPIRATION_START_WINDOW: i32 = 50;
const QUIESCE_FUTILITY_MARGIN: i32 = 200;

const FUTILITY_PRUNE_MAX_DEPTH: i16 = 3;
const FUTILITY_MARGIN_BASE: i32 = 200;
const FUTILITY_MARGIN_PER_DEPTH: i32 = 150;
const REVERSE_FUTILITY_PRUNE_MAX_DEPTH: i16 = 3;

const CHECK_EXTENSION_DEPTH_LIMIT: i16 = 2;
const PASSED_PAWN_EXTENSION_DEPTH_LIMIT: i16 = 4;

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

const PIECE_ORDER: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[inline]
fn opposite_color(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

#[inline]
fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 300,
        Piece::Bishop => 300,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 10_000,
    }
}

#[inline]
fn pawn_attackers_to(square: Square, pawns: BitBoard, color: Color) -> BitBoard {
    let mut attackers = BitBoard::new(0);
    match color {
        Color::White => {
            if let Some(down) = square.down() {
                if let Some(left) = down.left() {
                    let bb = BitBoard::from_square(left);
                    if (pawns & bb).0 != 0 {
                        attackers |= bb;
                    }
                }
                if let Some(right) = down.right() {
                    let bb = BitBoard::from_square(right);
                    if (pawns & bb).0 != 0 {
                        attackers |= bb;
                    }
                }
            }
        }
        Color::Black => {
            if let Some(up) = square.up() {
                if let Some(left) = up.left() {
                    let bb = BitBoard::from_square(left);
                    if (pawns & bb).0 != 0 {
                        attackers |= bb;
                    }
                }
                if let Some(right) = up.right() {
                    let bb = BitBoard::from_square(right);
                    if (pawns & bb).0 != 0 {
                        attackers |= bb;
                    }
                }
            }
        }
    }
    attackers
}

fn compute_attackers_to(
    square: Square,
    occ: BitBoard,
    piece_bb: &[[BitBoard; 6]; 2],
) -> [BitBoard; 2] {
    let mut attackers = [BitBoard::new(0); 2];

    for color in [Color::White, Color::Black] {
        let idx = color.to_index();
        let pawns = piece_bb[idx][Piece::Pawn.to_index() as usize];
        let knights = piece_bb[idx][Piece::Knight.to_index() as usize];
        let bishops = piece_bb[idx][Piece::Bishop.to_index() as usize];
        let rooks = piece_bb[idx][Piece::Rook.to_index() as usize];
        let queens = piece_bb[idx][Piece::Queen.to_index() as usize];
        let kings = piece_bb[idx][Piece::King.to_index() as usize];

        let mut attack = BitBoard::new(0);
        attack |= pawn_attackers_to(square, pawns, color);
        attack |= get_knight_moves(square) & knights;
        attack |= get_bishop_moves(square, occ) & (bishops | queens);
        attack |= get_rook_moves(square, occ) & (rooks | queens);
        attack |= get_king_moves(square) & kings;
        attackers[idx] = attack;
    }

    attackers
}

fn select_least_valuable_attacker(
    color_idx: usize,
    attack_mask: BitBoard,
    piece_bb: &[[BitBoard; 6]; 2],
) -> Option<(Piece, Square)> {
    for piece in PIECE_ORDER.iter() {
        let idx = piece.to_index() as usize;
        let candidates = piece_bb[color_idx][idx] & attack_mask;
        if candidates.0 != 0 {
            let sq = candidates.to_square();
            return Some((*piece, sq));
        }
    }
    None
}

pub fn see_for_sort(board: &Board, mv: ChessMove) -> i32 {
    let captured_val = board
        .piece_on(mv.get_dest())
        .map(piece_value)
        .unwrap_or_else(|| piece_value(Piece::Pawn));

    let current_val = match mv.get_promotion() {
        Some(prom) => piece_value(prom) - piece_value(Piece::Pawn),
        None => piece_value(board.piece_on(mv.get_source()).unwrap()),
    };

    // If first capture is already good or equal no need to go later
    if captured_val >= current_val {
        return captured_val - current_val;
    }

    // For "bad" captures we check if really bad with static exchange evaluation
    static_exchange_eval(board, mv)
}

fn static_exchange_eval(board: &Board, mv: ChessMove) -> i32 {
    let from = mv.get_source();
    let to = mv.get_dest();

    let moving_piece = match board.piece_on(from) {
        Some(p) => p,
        None => return 0,
    };
    let captured_piece = if is_en_passant_capture(board, mv) {
        Some(Piece::Pawn)
    } else {
        board.piece_on(to)
    };
    if captured_piece.is_none() {
        return 0;
    }
    let captured_piece = captured_piece.unwrap();

    let mut occ = *board.combined();
    let mut piece_bb = [[BitBoard::new(0); 6]; 2];
    for piece in PIECE_ORDER.iter() {
        let idx = piece.to_index() as usize;
        let bb = *board.pieces(*piece);
        piece_bb[Color::White.to_index()][idx] = bb & *board.color_combined(Color::White);
        piece_bb[Color::Black.to_index()][idx] = bb & *board.color_combined(Color::Black);
    }

    let us = board.side_to_move();
    let them = opposite_color(us);
    let us_idx = us.to_index();
    let them_idx = them.to_index();

    let from_bb = BitBoard::from_square(from);
    let to_bb = BitBoard::from_square(to);

    // Remove moving piece from its origin square
    piece_bb[us_idx][moving_piece.to_index() as usize] &= !from_bb;
    occ &= !from_bb;

    // Remove captured piece
    if is_en_passant_capture(board, mv) {
        if let Some(capture_sq) = to.backward(us) {
            let cap_bb = BitBoard::from_square(capture_sq);
            piece_bb[them_idx][Piece::Pawn.to_index() as usize] &= !cap_bb;
            occ &= !cap_bb;
        }
    } else {
        piece_bb[them_idx][captured_piece.to_index() as usize] &= !to_bb;
        occ &= !to_bb;
    }

    // Place moving piece on the destination square (handle promotion)
    let mut current_piece = mv.get_promotion().unwrap_or(moving_piece);
    piece_bb[us_idx][current_piece.to_index() as usize] |= to_bb;
    occ |= to_bb;

    let mut gain = [0i32; 32];
    gain[0] = piece_value(captured_piece);

    let mut attackers = compute_attackers_to(to, occ, &piece_bb);
    let mut depth = 0usize;
    let mut side_idx = them_idx;

    while depth < gain.len() - 1 {
        let attack_mask = attackers[side_idx] & occ;
        if attack_mask.0 == 0 {
            break;
        }

        depth += 1;
        let captured_value = piece_value(current_piece);

        let (att_piece, att_square) =
            match select_least_valuable_attacker(side_idx, attack_mask, &piece_bb) {
                Some(v) => v,
                None => break,
            };

        gain[depth] = captured_value - gain[depth - 1];
        if gain[depth] < 0 {
            break;
        }

        let captured_side_idx = 1 - side_idx;
        piece_bb[captured_side_idx][current_piece.to_index() as usize] &= !to_bb;
        occ &= !to_bb;

        let att_bb = BitBoard::from_square(att_square);
        piece_bb[side_idx][att_piece.to_index() as usize] &= !att_bb;
        occ &= !att_bb;

        piece_bb[side_idx][att_piece.to_index() as usize] |= to_bb;
        occ |= to_bb;

        current_piece = att_piece;
        attackers = compute_attackers_to(to, occ, &piece_bb);
        side_idx ^= 1;
    }

    while depth > 0 {
        gain[depth - 1] = -cmp::max(-gain[depth - 1], gain[depth]);
        depth -= 1;
    }

    gain[0]
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
) -> SearchScore {
    if should_stop(ctx.stop) {
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
        return SearchScore::EVAL(quiesce_negamax_it(
            ctx,
            board,
            alpha,
            beta,
            ply_from_root,
        ));
    }

    let depth_remaining = depth;
    let zob = board.get_hash();
    let mut tt_eval_hint = None;
    if let Some(entry) = ctx.tt.probe(zob) {
        ctx.stats.tt_hits += 1;
        tt_eval_hint = entry.eval;
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

    let in_check = board.checkers().popcnt() > 0;
    if !is_pv_node
        && !in_check
        && depth_remaining >= 2
        && has_non_pawn_material(board, board.side_to_move())
    {
        let r: i16 = if depth_remaining >= 6 { 3 } else { 2 };
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

    let (static_eval, static_eval_raw) =
        if !in_check && depth_remaining <= REVERSE_FUTILITY_PRUNE_MAX_DEPTH {
            let eval = ctx.nnue.evaluate(board.side_to_move());
            (Some(eval), Some(eval))
        } else {
            (None, None)
        };

    if let Some(raw) = static_eval_raw {
        tt_eval_hint = Some(raw);
    }

    if let Some(stand_pat) = static_eval {
        if depth_remaining <= REVERSE_FUTILITY_PRUNE_MAX_DEPTH
            && !is_pv_node
            && alpha != NEG_INFINITY
            && beta != POS_INFINITY
            && !is_mate_score(beta)
            && !is_mate_score(stand_pat)
            && stand_pat >= beta.saturating_add(reverse_futility_margin(depth_remaining))
        {
            ctx.stats.reverse_futility_prunes += 1;
            return SearchScore::EVAL(beta);
        }
    }

    let static_eval_for_futility = if depth_remaining <= FUTILITY_PRUNE_MAX_DEPTH {
        static_eval
    } else {
        None
    };

    ctx.stats.incremental_move_gen_inits += 1;
    let ply_index = ply_from_root.max(0) as usize;
    let killer_moves = ctx.killers.killers_for(ply_index);
    let mut incremental_move_gen = IncrementalMoveGen::new(board, ctx.tt, killer_moves);

    if incremental_move_gen.is_over() {
        if in_check {
            return SearchScore::EVAL(mated_in_plies(ply_from_root));
        }
        return SearchScore::EVAL(0);
    }

    let mut move_idx: usize = 0;
    while let Some(mv) = incremental_move_gen.next() {
        if incremental_move_gen.take_capture_generation_event() {
            ctx.stats.incremental_move_gen_capture_lists += 1;
        }
        let new_board = board_do_move(board, mv, ctx.repetition);

        let is_capture =
            board.piece_on(mv.get_dest()).is_some() || is_en_passant_capture(board, mv);
        let gives_check = new_board.checkers().popcnt() > 0;
        let is_first_move = move_idx == 0;
        let child_is_pv_node = is_pv_node && is_first_move;

        let mut extension: i16 = 0;
        if gives_check && depth_remaining <= CHECK_EXTENSION_DEPTH_LIMIT && depth_remaining > 0 {
            extension = 1;
        } else if depth_remaining <= PASSED_PAWN_EXTENSION_DEPTH_LIMIT
            && depth_remaining > 0
            && is_passed_pawn_push(board, &new_board, mv)
        {
            extension = 1;
        }

        if let Some(stand_pat) = static_eval_for_futility {
            if depth_remaining <= FUTILITY_PRUNE_MAX_DEPTH
                && !is_capture
                && !gives_check
                && !is_first_move
                && alpha != NEG_INFINITY
                && beta != POS_INFINITY
                && !is_mate_score(alpha)
            {
                let margin = futility_margin(depth_remaining);
                if stand_pat + margin <= alpha {
                    ctx.stats.futility_prunes += 1;
                    move_idx += 1;
                    ctx.repetition.pop();
                    continue;
                }
            }
        }

        let apply_lmr =
            depth_remaining >= 3 && move_idx >= 3 && !is_first_move && !is_capture && !gives_check;
        let reduction: i16 = if apply_lmr {
            if depth_remaining >= 6 {
                2
            } else {
                1
            }
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
                            child_is_pv_node,
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

        if value > best_value {
            best_value = value;
            best_move_opt = Some(mv);
        }

        if value >= beta {
            if !is_capture && !gives_check {
                ctx.killers.record(ply_index, mv);
            }
            ctx.stats.beta_cutoffs += 1;
            ctx.repetition.pop();
            break;
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
        tt_eval_hint,
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

    let mut killer_table = KillerTable::new(max_depth + 4);
    let mut ctx = ThreadContext {
        tt: transpo_table,
        stop,
        stats,
        repetition: repetition_table,
        killers: &mut killer_table,
        nnue: nnue_state,
    };
    let mut out = io::stdout();

    let killer_moves = ctx.killers.killers_for(0);
    let mut incremental_move_gen = IncrementalMoveGen::new(board, ctx.tt, killer_moves);

    while let Some(mv) = incremental_move_gen.next() {
        if should_stop(ctx.stop) {
            aborted = true;
            break;
        }

        let current_best = best_move.map(|bm| (bm, best_value));
        let new_board = board_do_move(board, mv, ctx.repetition);
        let is_pv_node = best_move.is_none();
        let mut child_depth = max_depth.saturating_sub(1);

        child_depth = child_depth.min(i16::MAX as usize);
        ctx.nnue.push();
        ctx.nnue.apply_move(board, mv);

        let result = negamax_it(
            &mut ctx,
            &new_board,
            child_depth as i16,
            -beta,
            -alpha,
            1,
            is_pv_node,
        );

        ctx.nnue.pop();

        match result {
            SearchScore::CANCELLED => {
                ctx.repetition.pop();
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
    nnue: &'a mut NNUEState,
}

fn iterative_deepening_loop<F>(
    board: &Board,
    max_depth: usize,
    transpo_table: &TranspositionTable,
    repetition_table: &mut RepetitionTable,
    stop: &StopFlag,
    stats: &mut SearchStats,
    nnue_state: &mut NNUEState,
    mut on_iteration: F,
) -> (Option<ChessMove>, i32)
where
    F: FnMut(&IterationReport),
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
        );

        if let Some(bm) = result.best_move {
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

/// Compute the best move of a board from a depth limit
#[allow(clippy::too_many_arguments)]
pub fn best_move_using_iterative_deepening(
    board: &Board,
    max_depth: usize,
    transpo_table: Arc<TranspositionTable>,
    repetition_table: RepetitionTable,
    threads: usize,
) -> SearchOutcome {
    let stop = Arc::new(AtomicBool::new(false));
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

    let (best_move, best_score) = iterative_deepening_loop(
        board,
        max_depth,
        transpo_table.as_ref(),
        &mut local_repetition,
        &stop,
        &mut stats,
        nnue_state.as_mut(),
        |_| {},
    );

    stop.store(true, Ordering::Relaxed);
    join_helper_threads(helper_handles);

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
    repetition_table: RepetitionTable,
    transpo_table: Arc<TranspositionTable>,
    mut stdout_opt: Option<&mut Stdout>,
    threads: usize,
) -> SearchOutcome {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop_clone = stop.clone();
        thread::spawn(move || {
            thread::sleep(time_budget);
            stop_clone.store(true, Ordering::Relaxed);
        });
    }

    let mut stats = SearchStats::default();
    let mut nnue_state = NNUEState::from_board(board);
    let mut local_repetition = repetition_table.clone();

    let helper_handles = spawn_lazy_smp_helpers(
        board,
        max_depth_cap,
        &transpo_table,
        repetition_table,
        threads,
        &stop,
    );

    let t0 = Instant::now();
    let (best_move, best_score) = iterative_deepening_loop(
        board,
        max_depth_cap,
        transpo_table.as_ref(),
        &mut local_repetition,
        &stop,
        &mut stats,
        nnue_state.as_mut(),
        |report| {
            if report.result.aborted {
                return;
            }

            if let Some(out) = stdout_opt.as_mut() {
                let nodes = report.stats_delta.nodes;
                let time_ms = t0.elapsed().as_millis() as u64;
                let nps = if time_ms > 0 {
                    nodes * 1_000 / time_ms
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
