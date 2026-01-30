use crate::tables::{CaptureHistoryTable, HistoryTable};
use crate::TranspositionTable;
use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_rook_moves, BitBoard, Board, ChessMove,
    Color, MoveGen, Piece, Square, EMPTY,
};
use smallvec::SmallVec;
use std::cmp;

type MoveList = SmallVec<[ChessMove; 64]>;

const PROMOTION_HISTORY_BONUS: i32 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveGenPhase {
    TTMove,
    GoodCaptures,
    GoodQuiet,
    BadCaptures,
    BadQuiet,
    Done,
}

pub struct IncrementalMoveGen<'a> {
    board: &'a Board,
    tt_move: Option<ChessMove>,
    killer_moves: [Option<ChessMove>; 2],
    countermove: Option<ChessMove>,
    move_gen_iter: MoveGen,
    phase: MoveGenPhase,
    side_to_move: Color,
    has_legal_moves: bool,
    capture_gen_pending: bool,
    captures_generated: bool,
    quiet_generated: bool,
    good_captures: MoveList,
    bad_captures: MoveList,
    killer_quiet: MoveList,
    good_quiet: MoveList,
    bad_quiet: MoveList,
    good_capture_idx: usize,
    bad_capture_idx: usize,
    killer_quiet_idx: usize,
    good_quiet_idx: usize,
    bad_quiet_idx: usize,
}

impl<'a> IncrementalMoveGen<'a> {
    pub fn new(
        board: &'a Board,
        tt: &TranspositionTable,
        killer_moves: [Option<ChessMove>; 2],
        countermove: Option<ChessMove>,
    ) -> Self {
        let probe_iter = MoveGen::new_legal(board);
        let has_legal_moves = probe_iter.len() > 0;

        let zob = board.get_hash();
        let tt_move = tt
            .probe(zob)
            .and_then(|e| e.best_move)
            .filter(|mv| board.legal(*mv));

        Self {
            board,
            tt_move,
            killer_moves,
            countermove,
            phase: MoveGenPhase::TTMove,
            has_legal_moves,
            capture_gen_pending: false,
            captures_generated: false,
            quiet_generated: false,
            move_gen_iter: probe_iter,
            side_to_move: board.side_to_move(),
            good_captures: MoveList::new(),
            bad_captures: MoveList::new(),
            killer_quiet: MoveList::new(),
            good_quiet: MoveList::new(),
            bad_quiet: MoveList::new(),
            good_capture_idx: 0,
            bad_capture_idx: 0,
            killer_quiet_idx: 0,
            good_quiet_idx: 0,
            bad_quiet_idx: 0,
        }
    }

    fn ensure_captures(&mut self, cap_hist: &CaptureHistoryTable) {
        if self.captures_generated {
            return;
        }
        self.captures_generated = true;
        self.capture_gen_pending = true;

        let mut capture_mask = *self.board.color_combined(!self.board.side_to_move());
        if let Some(ep_sq) = self.board.en_passant() {
            capture_mask |= BitBoard::from_square(ep_sq);
        }

        self.move_gen_iter.set_iterator_mask(capture_mask);

        let mut good_scored: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();
        let mut bad_scored: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();

        for mv in self.move_gen_iter.by_ref() {
            if Some(mv) == self.tt_move {
                continue;
            }

            let see_val = see_for_sort(self.board, mv);
            let ch_bonus = if let (Some(piece), Some(captured)) = (
                self.board.piece_on(mv.get_source()),
                self.board.piece_on(mv.get_dest()),
            ) {
                cap_hist
                    .score(self.side_to_move, piece, mv.get_dest(), captured)
            } else {
                0
            };
            // SEE dominates; capture history is a tiebreaker within the same SEE band
            let mut score = see_val * 1024 + ch_bonus;
            if see_val >= 0 {
                score += 1024; // ensure good captures stay above 0
                good_scored.push((mv, score));
            } else {
                bad_scored.push((mv, score));
            }
        }

        good_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        bad_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        self.good_captures
            .extend(good_scored.into_iter().map(|(mv, _)| mv));
        self.bad_captures
            .extend(bad_scored.into_iter().map(|(mv, _)| mv));
    }

    fn ensure_quiet(&mut self, history: &HistoryTable) {
        if self.quiet_generated {
            return;
        }
        self.quiet_generated = true;

        let mut killer_hits: MoveList = MoveList::new();
        let mut good_scored: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();
        let mut bad_scored: SmallVec<[(ChessMove, i32); 64]> = SmallVec::new();
        let mut killer_seen = [false; 2];
        let mut countermove_seen = false;

        self.move_gen_iter.set_iterator_mask(!EMPTY);
        for mv in self.move_gen_iter.by_ref() {
            if Some(mv) == self.tt_move {
                continue;
            }
            if self.board.piece_on(mv.get_dest()).is_some() {
                continue;
            }

            // Check if this is the countermove
            if !countermove_seen && self.countermove == Some(mv) {
                countermove_seen = true;
                killer_hits.push(mv);
                continue;
            }

            let mut matched_killer = false;
            for (idx, killer) in self.killer_moves.iter().enumerate() {
                if killer_seen[idx] {
                    continue;
                }
                if *killer == Some(mv) {
                    killer_seen[idx] = true;
                    killer_hits.push(mv);
                    matched_killer = true;
                    break;
                }
            }
            if matched_killer {
                continue;
            }

            let mut score = history.score(self.side_to_move, mv);
            if mv.get_promotion().is_some() {
                score += PROMOTION_HISTORY_BONUS;
                good_scored.push((mv, score));
            } else {
                bad_scored.push((mv, score));
            }
        }

        self.killer_quiet = killer_hits;
        good_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        bad_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        self.good_quiet
            .extend(good_scored.into_iter().map(|(mv, _)| mv));
        self.bad_quiet
            .extend(bad_scored.into_iter().map(|(mv, _)| mv));
    }

    pub fn take_capture_generation_event(&mut self) -> bool {
        if self.capture_gen_pending {
            self.capture_gen_pending = false;
            true
        } else {
            false
        }
    }

    // Game's over if player cannot play a single move
    pub fn is_over(&self) -> bool {
        !self.has_legal_moves
    }

    pub fn next(&mut self, history: &HistoryTable, cap_hist: &CaptureHistoryTable) -> Option<ChessMove> {
        loop {
            match self.phase {
                MoveGenPhase::TTMove => {
                    self.phase = MoveGenPhase::GoodCaptures;
                    if let Some(tt_mv) = self.tt_move {
                        return Some(tt_mv);
                    }
                }
                MoveGenPhase::GoodCaptures => {
                    self.ensure_captures(cap_hist);
                    if self.good_capture_idx < self.good_captures.len() {
                        let mv = self.good_captures[self.good_capture_idx];
                        self.good_capture_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::GoodQuiet;
                }
                MoveGenPhase::GoodQuiet => {
                    self.ensure_quiet(history);
                    if self.killer_quiet_idx < self.killer_quiet.len() {
                        let mv = self.killer_quiet[self.killer_quiet_idx];
                        self.killer_quiet_idx += 1;
                        return Some(mv);
                    }
                    if self.good_quiet_idx < self.good_quiet.len() {
                        let mv = self.good_quiet[self.good_quiet_idx];
                        self.good_quiet_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::BadCaptures;
                }
                MoveGenPhase::BadCaptures => {
                    self.ensure_captures(cap_hist);
                    if self.bad_capture_idx < self.bad_captures.len() {
                        let mv = self.bad_captures[self.bad_capture_idx];
                        self.bad_capture_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::BadQuiet;
                }
                MoveGenPhase::BadQuiet => {
                    self.ensure_quiet(history);
                    if self.bad_quiet_idx < self.bad_quiet.len() {
                        let mv = self.bad_quiet[self.bad_quiet_idx];
                        self.bad_quiet_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::Done;
                }
                MoveGenPhase::Done => return None,
            }
        }
    }
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
pub fn piece_value(piece: Piece) -> i32 {
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

pub fn static_exchange_eval(board: &Board, mv: ChessMove) -> i32 {
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
        // Not a bug: if recapturing gives negative gain, the player won't recapture.
        // In chess you can always stop the exchange sequence if continuing is bad.
        // Example: Pawn takes Knight (+300), opponent can recapture but that gives them -200,
        // so they simply don't recapture. The negamax loop below handles the final eval.
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
pub fn is_en_passant_capture(board: &Board, mv: ChessMove) -> bool {
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
