use crate::eval::eval_board;
use crate::movegen::IncrementalMoveGen;
use crate::{
    board_do_move, board_pop, send_message, PVTable, RepetitionTable, StopFlag, TTFlag,
    TranspositionTable, CACHE_COUNT, DEPTH_COUNT, EVAL_COUNT, NEG_INFINITY, POS_INFINITY,
    QUIESCE_REMAIN,
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
    pub reverse_futility_prunes: u64,
    pub lmr_researches: u64,
    pub incremental_move_gen_inits: u64,
    pub incremental_move_gen_capture_lists: u64,
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
        }
    }

    fn format_as_info(&self) -> String {
        format!(
            "nodes={} qnodes={} tt_hits={} tt_exact={} tt_lower={} tt_upper={} beta_cut={} qbeta_cut={} null_prune={} futility_prune={} rfutility_prune={} lmr_retry={} img_init={} img_capgen={}",
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

    if remain_quiet == 0 || repetition_table.is_in_threefold_scenario(board) {
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

    for (mv, _val) in get_captures(board) {
        let dest_piece = board.piece_on(mv.get_dest());
        let is_en_passant = dest_piece.is_none() && is_en_passant_capture(board, mv);
        if dest_piece.is_none() && !is_en_passant {
            continue;
        }
        let captured_val = dest_piece
            .map(piece_value)
            .unwrap_or_else(|| piece_value(Piece::Pawn));
        if stand_pat + captured_val + QUIESCE_FUTILITY_MARGIN <= alpha {
            continue;
        }
        if see_for_sort(board, mv) < 0 {
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

/// Cached evaluation: if threefold, return 0, else look up in transposition table.
/// If not found, compute via `eval_board`, insert into the table, and return.
fn cache_eval(
    board: &Board,
    transpo_table: &mut TranspositionTable,
    repetition_table: &RepetitionTable,
) -> i32 {
    if repetition_table.is_in_threefold_scenario(board) {
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
    let captured_val = board.piece_on(mv.get_dest())
        .map(piece_value)
        .unwrap_or_else(|| piece_value(Piece::Pawn));

    let current_val = match mv.get_promotion() {
        Some(prom) => piece_value(prom) - piece_value(Piece::Pawn),
        None => piece_value(board.piece_on(mv.get_source()).unwrap())
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

/// This functions orders to move based on the first sort of `sort_moves` but adds specific order
/// criteria for the non-quiesce search.
/// Such as taking moves from the transposition table and principal variation
fn ordered_moves(
    board: &Board,
    pv_table: &PVTable,
    tt: &TranspositionTable,
    _depth: usize,
) -> Vec<ChessMove> {
    let mut scored_moves: SmallVec<[(ChessMove, i32); 64]> = SmallVec::new();

    let zob = board.get_hash();
    let mut tt_mv = tt.probe(zob).and_then(|e| e.best_move);
    let mut pv_mv = pv_table.get(&zob);

    for mv in MoveGen::new_legal(board) {
        let score = if pv_mv.is_some() && pv_mv == Some(&mv) {
            pv_mv = None;
            70
        } else if tt_mv.is_some() && tt_mv == Some(mv) {
            tt_mv = None;
            65
        } else if let Some(victim_piece) = board.piece_on(mv.get_dest()) {
            // It's a capture; find the attacker piece type
            if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
                let victim_idx = victim_piece.to_index() as usize;
                let attacker_idx = attacker_piece.to_index() as usize;
                MVV_LVA_TABLE[victim_idx][attacker_idx] as i32
            } else {
                0
            }
        } else if is_en_passant_capture(board, mv) {
            25
        } else {
            0
        };

        scored_moves.push((mv, score));
    }

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

// ---------------------------------------------------------------------------
// Core Negamax with α/β pruning + PV maintenance — renamed `negamax_it`.
// ---------------------------------------------------------------------------
/// This is the core of the move search, it perform a negamax styled search with alpha/beta pruning
/// Usage of quiesce search at end of search and tt/pv cutoff
fn negamax_it(
    board: &Board,
    depth: i16,
    mut alpha: i32,
    beta: i32,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    killers: &mut KillerTable,
    color: i32,
    stop: &StopFlag,
    ply_from_root: i32,
    is_pv_node: bool,
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
    if repetition_table.is_in_threefold_scenario(board) {
        return SearchScore::EVAL(0);
    }

    // Start quiesce search if we exhausted the search depth
    if depth <= 0 {
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
    let depth_remaining: i16 = depth;
    let zob = board.get_hash();
    let mut tt_eval_hint = None;
    if let Some(entry) = transpo_table.probe(zob) {
        stats.tt_hits += 1;
        tt_eval_hint = entry.eval;
        // Only keep if entry depth is same or higher to avoid pruning from not fully search position
        if entry.depth >= depth_remaining {
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
    if !is_pv_node
        && !in_check
        && depth_remaining >= 2
        && has_non_pawn_material(board, board.side_to_move())
    {
        let r: i16 = if depth_remaining >= 6 { 3 } else { 2 };
        if let Some(nboard) = board_do_null_move(board, repetition_table) {
            // null-window search (fail-high test)
            match negamax_it(
                &nboard,
                (depth_remaining - 1 - r).max(0),
                -beta,
                -beta + 1,
                transpo_table,
                repetition_table,
                pv_table,
                killers,
                -color,
                stop,
                ply_from_root + 1,
                false,
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

    let (static_eval, static_eval_raw) =
        if !in_check && depth_remaining <= REVERSE_FUTILITY_PRUNE_MAX_DEPTH {
            let raw = cache_eval(board, transpo_table, repetition_table);
            (Some(color * raw), Some(raw))
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
            stats.reverse_futility_prunes += 1;
            return SearchScore::EVAL(beta);
        }
    }

    let static_eval_for_futility = if depth_remaining <= FUTILITY_PRUNE_MAX_DEPTH {
        static_eval
    } else {
        None
    };

    stats.incremental_move_gen_inits += 1;
    let ply_index = ply_from_root.max(0) as usize;
    let killer_moves = killers.killers_for(ply_index);
    let mut incremental_move_gen =
        IncrementalMoveGen::new(board, pv_table, transpo_table, killer_moves);

    if incremental_move_gen.is_over() {
        if in_check {
            return SearchScore::EVAL(mated_in_plies(ply_from_root))
        }
        // Else in stalemate
        return SearchScore::EVAL(0);
    }

    // Start the main search
    let mut move_idx: usize = 0;
    while let Some(mv) = incremental_move_gen.next() {
        if incremental_move_gen.take_capture_generation_event() {
            stats.incremental_move_gen_capture_lists += 1;
        }
        let new_board = board_do_move(board, mv, repetition_table);

        // Determine properties for LMR conditions
        let is_capture =
            board.piece_on(mv.get_dest()).is_some() || is_en_passant_capture(board, mv);
        let gives_check = new_board.checkers().popcnt() > 0;
        let is_first_move = move_idx == 0; // treat first move as PV
        let child_is_pv_node = is_pv_node && is_first_move;

        // Search extension
        let mut extension: i16 = 0;
        if gives_check && depth_remaining <= CHECK_EXTENSION_DEPTH_LIMIT && depth_remaining > 0 {
            extension = 1;
        } else if depth_remaining <= PASSED_PAWN_EXTENSION_DEPTH_LIMIT
            && depth_remaining > 0
            //TODO: Maybe add an endgame closeness to trigger the is_passed_pawn_push call (can be computed at the root and just be passed bellow)
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
                    stats.futility_prunes += 1;
                    move_idx += 1;
                    repetition_table.pop();
                    continue;
                }
            }
        }

        // Apply LMR for late, quiet, non-check, non-PV moves with sufficient remaining depth
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

        let mut value;
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

        if use_pvs {
            let null_window_beta = null_window_beta_candidate;
            let first_result = negamax_it(
                &new_board,
                reduced_depth,
                -null_window_beta,
                -alpha,
                transpo_table,
                repetition_table,
                pv_table,
                killers,
                -color,
                stop,
                ply_from_root + 1,
                false,
                stats,
            );

            value = match first_result {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(v) => -v,
            };

            if value > alpha && value < beta {
                if reduction > 0 {
                    stats.lmr_researches += 1;
                }

                value = match negamax_it(
                    &new_board,
                    depth_after_move_with_ext,
                    -beta,
                    -alpha,
                    transpo_table,
                    repetition_table,
                    pv_table,
                    killers,
                    -color,
                    stop,
                    ply_from_root + 1,
                    child_is_pv_node,
                    stats,
                ) {
                    SearchScore::CANCELLED => return SearchScore::CANCELLED,
                    SearchScore::EVAL(v) => -v,
                };
            }
        } else {
            // Normal full-depth search (PV move)
            value = match negamax_it(
                &new_board,
                reduced_depth,
                -beta,
                -alpha,
                transpo_table,
                repetition_table,
                pv_table,
                killers,
                -color,
                stop,
                ply_from_root + 1,
                child_is_pv_node,
                stats,
            ) {
                SearchScore::CANCELLED => return SearchScore::CANCELLED,
                SearchScore::EVAL(v) => -v,
            };
        }

        // Update best / alpha-beta
        if value > best_value {
            best_value = value;
            best_move_opt = Some(mv);
            pv_table.insert(zob, mv);
        }

        if value >= beta {
            if !is_capture && !gives_check {
                killers.record(ply_index, mv);
            }
            stats.beta_cutoffs += 1;
            repetition_table.pop();
            break;
        }

        if value > alpha {
            alpha = value;
        }

        repetition_table.pop();
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
    transpo_table.store(
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

    let mut killer_table = KillerTable::new(max_depth + 4);
    let mut out = io::stdout();

    for mv in ordered_moves(board, pv_table, transpo_table, 0) {
        if should_stop(stop) {
            aborted = true;
            break;
        }

        let current_best = best_move.map(|bm| (bm, best_value));
        let new_board = board_do_move(board, mv, repetition_table);
        let is_pv_node = best_move.is_none();
        let mut child_depth = max_depth.saturating_sub(1);

        child_depth = child_depth.min(i16::MAX as usize);
        match negamax_it(
            &new_board,
            child_depth as i16,
            -beta,
            -alpha,
            transpo_table,
            repetition_table,
            pv_table,
            &mut killer_table,
            -color,
            stop,
            1, // one ply from root after making mv
            is_pv_node,
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
                    repetition_table.pop();
                    break;
                }

                repetition_table.pop();
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
