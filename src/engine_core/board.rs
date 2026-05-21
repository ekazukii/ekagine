use super::attacks::{
    bishop_attacks_u64, king_attacks_u64, knight_attacks_u64, pawn_attacks_u64, rook_attacks_u64,
};
use super::types::{BitBoard, BoardStatus, ChessMove, Color, File, Piece, Rank, Square};
use super::zobrist::{key_castling, key_ep_file, key_piece, key_side};
use std::str::FromStr;

pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

const CASTLING_RIGHTS_MASK: [u8; 64] = {
    let mut a = [0xFFu8; 64];
    a[0] = !WQ;
    a[4] = !(WK | WQ);
    a[7] = !WK;
    a[56] = !BQ;
    a[60] = !(BK | BQ);
    a[63] = !BK;
    a
};

#[inline(always)]
fn encode_piece(piece: Piece, color: Color) -> u8 {
    let p = piece.to_index() as u8;
    let c = color.to_index() as u8;
    1 + c * 6 + p
}

#[inline(always)]
fn decode_piece(code: u8) -> Option<(Piece, Color)> {
    if code == 0 {
        return None;
    }
    let v = code - 1;
    let p = match v % 6 {
        0 => Piece::Pawn,
        1 => Piece::Knight,
        2 => Piece::Bishop,
        3 => Piece::Rook,
        4 => Piece::Queen,
        _ => Piece::King,
    };
    let c = if v / 6 == 0 { Color::White } else { Color::Black };
    Some((p, c))
}

#[derive(Debug, Clone, Copy)]
pub struct Board {
    pieces_bb: [BitBoard; 6],
    colors_bb: [BitBoard; 2],
    combined_bb: BitBoard,
    mailbox: [u8; 64],
    side: Color,
    castling: u8,
    en_passant: Option<Square>,
    halfmove_clock: u8,
    fullmove_number: u16,
    hash: u64,
    checkers_bb: BitBoard,
}

impl PartialEq for Board {
    fn eq(&self, other: &Board) -> bool {
        self.pieces_bb == other.pieces_bb
            && self.colors_bb == other.colors_bb
            && self.side == other.side
            && self.castling == other.castling
            && self.en_passant == other.en_passant
    }
}

impl Eq for Board {}

impl std::hash::Hash for Board {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::startpos()
    }
}

impl Board {
    pub fn startpos() -> Self {
        Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn empty() -> Self {
        Board {
            pieces_bb: [BitBoard(0); 6],
            colors_bb: [BitBoard(0); 2],
            combined_bb: BitBoard(0),
            mailbox: [0; 64],
            side: Color::White,
            castling: 0,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            checkers_bb: BitBoard(0),
        }
    }

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side
    }

    #[inline(always)]
    pub fn combined(&self) -> &BitBoard {
        &self.combined_bb
    }

    #[inline(always)]
    pub fn color_combined(&self, color: Color) -> &BitBoard {
        unsafe { self.colors_bb.get_unchecked(color.to_index()) }
    }

    #[inline(always)]
    pub fn pieces(&self, piece: Piece) -> &BitBoard {
        unsafe { self.pieces_bb.get_unchecked(piece.to_index() as usize) }
    }

    #[inline]
    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        decode_piece(self.mailbox[sq.to_index()]).map(|(p, _)| p)
    }

    #[inline]
    pub fn color_on(&self, sq: Square) -> Option<Color> {
        decode_piece(self.mailbox[sq.to_index()]).map(|(_, c)| c)
    }

    #[inline(always)]
    pub fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }

    #[inline(always)]
    pub fn checkers(&self) -> &BitBoard {
        &self.checkers_bb
    }

    #[inline(always)]
    pub fn get_hash(&self) -> u64 {
        self.hash
    }

    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.castling
    }

    pub fn status(&self) -> BoardStatus {
        let has_legal = super::movegen::any_legal_move(self);
        if has_legal {
            BoardStatus::Ongoing
        } else if self.checkers_bb.0 != 0 {
            BoardStatus::Checkmate
        } else {
            BoardStatus::Stalemate
        }
    }

    /// Returns true iff the side that just moved is not in check.
    /// Called on a board produced by `make_move_new`; the side just moved is `!self.side`.
    #[inline]
    pub fn is_position_legal(&self) -> bool {
        let just_moved = !self.side;
        let king_bb = self.pieces_bb[Piece::King.to_index() as usize].0
            & self.colors_bb[just_moved.to_index()].0;
        if king_bb == 0 {
            return true;
        }
        let king_sq = king_bb.trailing_zeros() as usize;
        !is_sq_attacked_internal(self, king_sq, self.side, self.combined_bb.0)
    }

    /// Verifies a (possibly user-supplied) move is fully legal in this position.
    /// Fast path: cheap pseudo-legality check, then king-safety via `PinInfo`.
    /// No `make_move_new` involved on the rejection path.
    pub fn legal(&self, mv: ChessMove) -> bool {
        if !self.is_pseudo_legal(mv) {
            return false;
        }
        let pin_info = super::movegen::PinInfo::for_board(self);
        pin_info.move_is_legal(self, mv)
    }

    /// Cheap pseudo-legality check for a single move (O(1) attack lookups,
    /// no full move-list regeneration). Verifies piece presence/color, target
    /// reachability and special-move shape (castling rights & path, EP target,
    /// promotion field consistency). King-safety (pin/check) is handled
    /// separately by `PinInfo::move_is_legal` — callers must combine both.
    pub fn is_pseudo_legal(&self, mv: ChessMove) -> bool {
        let from = mv.get_source();
        let to = mv.get_dest();
        let from_idx = from.to_index();
        let to_idx = to.to_index();
        if from_idx == to_idx {
            return false;
        }

        let (piece, color) = match decode_piece(self.mailbox[from_idx]) {
            Some(pc) => pc,
            None => return false,
        };
        if color != self.side {
            return false;
        }

        let us = self.side;
        let them = !us;
        let our_pieces = self.colors_bb[us.to_index()].0;
        let their_pieces = self.colors_bb[them.to_index()].0;
        let occ = self.combined_bb.0;
        let to_bit = 1u64 << to_idx;

        if (to_bit & our_pieces) != 0 {
            return false;
        }

        let promotion = mv.get_promotion();

        match piece {
            Piece::Pawn => {
                let from_rank = (from_idx / 8) as i32;
                let to_rank = (to_idx / 8) as i32;
                let last_rank = if us == Color::White { 7 } else { 0 };
                if promotion.is_some() != (to_rank == last_rank) {
                    return false;
                }

                let single_step: i32 = if us == Color::White { 8 } else { -8 };
                let single_target = from_idx as i32 + single_step;
                let double_target = from_idx as i32 + 2 * single_step;
                let start_rank = if us == Color::White { 1 } else { 6 };

                if to_idx as i32 == single_target && (to_bit & occ) == 0 {
                    return true;
                }
                if to_idx as i32 == double_target
                    && from_rank == start_rank
                    && (to_bit & occ) == 0
                    && ((1u64 << single_target) & occ) == 0
                {
                    return true;
                }
                let attacks = pawn_attacks_u64(us, from_idx);
                if (attacks & to_bit) != 0 {
                    if (to_bit & their_pieces) != 0 {
                        return true;
                    }
                    if Some(to) == self.en_passant && (to_bit & occ) == 0 {
                        return true;
                    }
                }
                false
            }
            Piece::Knight => {
                if promotion.is_some() {
                    return false;
                }
                (knight_attacks_u64(from_idx) & to_bit) != 0
            }
            Piece::Bishop => {
                if promotion.is_some() {
                    return false;
                }
                (bishop_attacks_u64(from_idx, occ) & to_bit) != 0
            }
            Piece::Rook => {
                if promotion.is_some() {
                    return false;
                }
                (rook_attacks_u64(from_idx, occ) & to_bit) != 0
            }
            Piece::Queen => {
                if promotion.is_some() {
                    return false;
                }
                ((bishop_attacks_u64(from_idx, occ) | rook_attacks_u64(from_idx, occ))
                    & to_bit)
                    != 0
            }
            Piece::King => {
                if promotion.is_some() {
                    return false;
                }
                if (king_attacks_u64(from_idx) & to_bit) != 0 {
                    return true;
                }
                // Castling: same rank, king moves exactly 2 files.
                if (from_idx / 8) == (to_idx / 8)
                    && (from_idx as i32 - to_idx as i32).abs() == 2
                {
                    return self.castling_pseudo_legal(
                        us,
                        from_idx as u8,
                        to_idx as u8,
                        occ,
                        them,
                    );
                }
                false
            }
        }
    }

    #[inline]
    fn castling_pseudo_legal(
        &self,
        us: Color,
        king_sq: u8,
        dest_sq: u8,
        occ: u64,
        them: Color,
    ) -> bool {
        if self.checkers_bb.0 != 0 {
            return false;
        }
        let castling = self.castling;
        let occ_no_king = occ ^ (1u64 << king_sq);
        match (us, king_sq, dest_sq) {
            (Color::White, 4, 6) => {
                (castling & WK) != 0
                    && (occ & ((1u64 << 5) | (1u64 << 6))) == 0
                    && !is_sq_attacked_internal(self, 5, them, occ_no_king)
                    && !is_sq_attacked_internal(self, 6, them, occ_no_king)
            }
            (Color::White, 4, 2) => {
                (castling & WQ) != 0
                    && (occ & ((1u64 << 1) | (1u64 << 2) | (1u64 << 3))) == 0
                    && !is_sq_attacked_internal(self, 3, them, occ_no_king)
                    && !is_sq_attacked_internal(self, 2, them, occ_no_king)
            }
            (Color::Black, 60, 62) => {
                (castling & BK) != 0
                    && (occ & ((1u64 << 61) | (1u64 << 62))) == 0
                    && !is_sq_attacked_internal(self, 61, them, occ_no_king)
                    && !is_sq_attacked_internal(self, 62, them, occ_no_king)
            }
            (Color::Black, 60, 58) => {
                (castling & BQ) != 0
                    && (occ & ((1u64 << 57) | (1u64 << 58) | (1u64 << 59))) == 0
                    && !is_sq_attacked_internal(self, 59, them, occ_no_king)
                    && !is_sq_attacked_internal(self, 58, them, occ_no_king)
            }
            _ => false,
        }
    }

    pub fn null_move(&self) -> Option<Board> {
        if self.checkers_bb.0 != 0 {
            return None;
        }
        let mut nb = *self;
        if let Some(ep) = nb.en_passant {
            nb.hash ^= key_ep_file(ep.0 % 8);
        }
        nb.en_passant = None;
        nb.side = !nb.side;
        nb.hash ^= key_side();
        nb.halfmove_clock = nb.halfmove_clock.saturating_add(1);
        nb.checkers_bb = compute_checkers(&nb);
        Some(nb)
    }

    pub fn make_move_new(&self, mv: ChessMove) -> Board {
        let mut nb = *self;
        nb.apply_move(mv);
        nb
    }

    #[inline]
    fn apply_move(&mut self, mv: ChessMove) {
        let from = mv.get_source();
        let to = mv.get_dest();
        let promotion = mv.get_promotion();
        let us = self.side;
        let them = !us;
        let from_idx = from.to_index();
        let to_idx = to.to_index();

        let from_code = self.mailbox[from_idx];
        let (piece, _) = decode_piece(from_code).expect("piece on source square");

        // Clear old en-passant key.
        if let Some(ep) = self.en_passant {
            self.hash ^= key_ep_file(ep.0 % 8);
        }
        let mut new_ep: Option<Square> = None;

        let is_castling = piece == Piece::King && (from_idx as i32 - to_idx as i32).abs() == 2;
        let to_code = self.mailbox[to_idx];
        let captured = decode_piece(to_code);
        let is_en_passant_capture =
            piece == Piece::Pawn && Some(to) == self.en_passant && captured.is_none();

        if piece == Piece::Pawn || captured.is_some() || is_en_passant_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        }

        // Remove captured piece (if any).
        if let Some((cap_piece, cap_color)) = captured {
            self.remove_piece(to_idx, cap_piece, cap_color);
        }
        if is_en_passant_capture {
            let captured_sq_idx = match us {
                Color::White => to_idx - 8,
                Color::Black => to_idx + 8,
            };
            self.remove_piece(captured_sq_idx, Piece::Pawn, them);
        }

        // Move (or promote) the piece.
        if let Some(promo) = promotion {
            self.remove_piece(from_idx, Piece::Pawn, us);
            self.put_piece(to_idx, promo, us);
        } else {
            self.move_piece(from_idx, to_idx, piece, us);
        }

        // Move rook for castling.
        if is_castling {
            let (rook_from, rook_to) = match (us, to_idx) {
                (Color::White, 6) => (7usize, 5usize),  // G1: H1->F1
                (Color::White, 2) => (0usize, 3usize),  // C1: A1->D1
                (Color::Black, 62) => (63usize, 61usize), // G8: H8->F8
                (Color::Black, 58) => (56usize, 59usize), // C8: A8->D8
                _ => unreachable!("invalid castling destination"),
            };
            self.move_piece(rook_from, rook_to, Piece::Rook, us);
        }

        // Double pawn push → ep square is the *target* (square passed over).
        // Only retain the EP marker (and hash its file) when an enemy pawn
        // can actually capture en-passant on the next move; otherwise two
        // positions that differ only by an inert EP marker would hash to
        // different keys and miss TT / 3-fold transpositions.
        if piece == Piece::Pawn {
            let from_rank = from_idx / 8;
            let to_rank = to_idx / 8;
            if (to_rank as i32 - from_rank as i32).abs() == 2 {
                let ep_idx = match us {
                    Color::White => from_idx + 8,
                    Color::Black => from_idx - 8,
                };
                let their_pawns = self.pieces_bb[Piece::Pawn.to_index() as usize].0
                    & self.colors_bb[them.to_index()].0;
                if (pawn_attacks_u64(us, ep_idx) & their_pawns) != 0 {
                    new_ep = Some(Square(ep_idx as u8));
                }
            }
        }

        // Update castling rights.
        let old_castling = self.castling;
        self.castling &= CASTLING_RIGHTS_MASK[from_idx];
        self.castling &= CASTLING_RIGHTS_MASK[to_idx];
        if self.castling != old_castling {
            self.hash ^= key_castling(old_castling);
            self.hash ^= key_castling(self.castling);
        }

        if let Some(ep_sq) = new_ep {
            self.hash ^= key_ep_file(ep_sq.0 % 8);
        }
        self.en_passant = new_ep;

        self.side = them;
        self.hash ^= key_side();

        if us == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }

        self.checkers_bb = compute_checkers(self);
    }

    #[inline(always)]
    fn put_piece(&mut self, sq_idx: usize, piece: Piece, color: Color) {
        let bit = 1u64 << sq_idx;
        self.pieces_bb[piece.to_index() as usize].0 |= bit;
        self.colors_bb[color.to_index()].0 |= bit;
        self.combined_bb.0 |= bit;
        self.mailbox[sq_idx] = encode_piece(piece, color);
        self.hash ^= key_piece(color, piece, Square(sq_idx as u8));
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq_idx: usize, piece: Piece, color: Color) {
        let bit = !(1u64 << sq_idx);
        self.pieces_bb[piece.to_index() as usize].0 &= bit;
        self.colors_bb[color.to_index()].0 &= bit;
        self.combined_bb.0 &= bit;
        self.mailbox[sq_idx] = 0;
        self.hash ^= key_piece(color, piece, Square(sq_idx as u8));
    }

    #[inline(always)]
    fn move_piece(&mut self, from_idx: usize, to_idx: usize, piece: Piece, color: Color) {
        let move_bb = (1u64 << from_idx) ^ (1u64 << to_idx);
        self.pieces_bb[piece.to_index() as usize].0 ^= move_bb;
        self.colors_bb[color.to_index()].0 ^= move_bb;
        self.combined_bb.0 ^= move_bb;
        self.mailbox[from_idx] = 0;
        self.mailbox[to_idx] = encode_piece(piece, color);
        self.hash ^= key_piece(color, piece, Square(from_idx as u8));
        self.hash ^= key_piece(color, piece, Square(to_idx as u8));
    }
}

#[inline]
pub(crate) fn compute_checkers(board: &Board) -> BitBoard {
    let king_bb = board.pieces_bb[Piece::King.to_index() as usize].0
        & board.colors_bb[board.side.to_index()].0;
    if king_bb == 0 {
        return BitBoard(0);
    }
    let king_sq = king_bb.trailing_zeros() as usize;
    let them = !board.side;
    let occ = board.combined_bb.0;
    let them_idx = them.to_index();
    let their_pieces = board.colors_bb[them_idx].0;

    let mut c = 0u64;
    let their_pawns = board.pieces_bb[Piece::Pawn.to_index() as usize].0 & their_pieces;
    c |= pawn_attacks_u64(board.side, king_sq) & their_pawns;
    let their_knights = board.pieces_bb[Piece::Knight.to_index() as usize].0 & their_pieces;
    c |= knight_attacks_u64(king_sq) & their_knights;
    let their_bq = (board.pieces_bb[Piece::Bishop.to_index() as usize].0
        | board.pieces_bb[Piece::Queen.to_index() as usize].0)
        & their_pieces;
    c |= bishop_attacks_u64(king_sq, occ) & their_bq;
    let their_rq = (board.pieces_bb[Piece::Rook.to_index() as usize].0
        | board.pieces_bb[Piece::Queen.to_index() as usize].0)
        & their_pieces;
    c |= rook_attacks_u64(king_sq, occ) & their_rq;
    BitBoard(c)
}

#[inline]
pub(crate) fn is_sq_attacked_internal(board: &Board, sq_idx: usize, by: Color, occ: u64) -> bool {
    let by_idx = by.to_index();
    let their_pieces = board.colors_bb[by_idx].0;
    let their_pawns = board.pieces_bb[Piece::Pawn.to_index() as usize].0 & their_pieces;
    if (pawn_attacks_u64(!by, sq_idx) & their_pawns) != 0 {
        return true;
    }
    let their_knights = board.pieces_bb[Piece::Knight.to_index() as usize].0 & their_pieces;
    if (knight_attacks_u64(sq_idx) & their_knights) != 0 {
        return true;
    }
    let their_kings = board.pieces_bb[Piece::King.to_index() as usize].0 & their_pieces;
    if (king_attacks_u64(sq_idx) & their_kings) != 0 {
        return true;
    }
    let their_bq = (board.pieces_bb[Piece::Bishop.to_index() as usize].0
        | board.pieces_bb[Piece::Queen.to_index() as usize].0)
        & their_pieces;
    if (bishop_attacks_u64(sq_idx, occ) & their_bq) != 0 {
        return true;
    }
    let their_rq = (board.pieces_bb[Piece::Rook.to_index() as usize].0
        | board.pieces_bb[Piece::Queen.to_index() as usize].0)
        & their_pieces;
    if (rook_attacks_u64(sq_idx, occ) & their_rq) != 0 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_core::Square;

    #[test]
    fn zobrist_consistent_e2e4() {
        let b1 = Board::default();
        let mv = ChessMove::new(Square::E2, Square::E4, None);
        let b2 = b1.make_move_new(mv);
        let b2_fen = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        assert_eq!(
            b2.get_hash(),
            b2_fen.get_hash(),
            "zobrist hash mismatch after e2e4"
        );
    }

    #[test]
    fn zobrist_consistent_castling() {
        let b1 = Board::from_str("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let mv = ChessMove::new(Square::E1, Square::G1, None);
        let b2 = b1.make_move_new(mv);
        let b2_fen = Board::from_str("r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1").unwrap();
        assert_eq!(
            b2.get_hash(),
            b2_fen.get_hash(),
            "zobrist hash mismatch after O-O"
        );
    }

    #[test]
    fn zobrist_consistent_ep() {
        let b1 = Board::from_str("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
            .unwrap();
        let mv = ChessMove::new(Square::E5, Square::D6, None);
        let b2 = b1.make_move_new(mv);
        let b2_fen =
            Board::from_str("rnbqkbnr/ppp1pppp/3P4/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3").unwrap();
        assert_eq!(
            b2.get_hash(),
            b2_fen.get_hash(),
            "zobrist hash mismatch after EP"
        );
    }
}

impl FromStr for Board {
    type Err = String;

    fn from_str(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN must have at least 4 fields, got '{}'", fen));
        }
        let mut board = Board::empty();

        // 1. piece placement
        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += c as i32 - '0' as i32;
                }
                _ => {
                    if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                        return Err(format!(
                            "FEN piece placement out of range: char '{}' at rank {} file {}",
                            c, rank, file
                        ));
                    }
                    let (piece, color) = match c {
                        'P' => (Piece::Pawn, Color::White),
                        'N' => (Piece::Knight, Color::White),
                        'B' => (Piece::Bishop, Color::White),
                        'R' => (Piece::Rook, Color::White),
                        'Q' => (Piece::Queen, Color::White),
                        'K' => (Piece::King, Color::White),
                        'p' => (Piece::Pawn, Color::Black),
                        'n' => (Piece::Knight, Color::Black),
                        'b' => (Piece::Bishop, Color::Black),
                        'r' => (Piece::Rook, Color::Black),
                        'q' => (Piece::Queen, Color::Black),
                        'k' => (Piece::King, Color::Black),
                        _ => return Err(format!("FEN bad piece char '{}'", c)),
                    };
                    let sq_idx = (rank * 8 + file) as usize;
                    let bit = 1u64 << sq_idx;
                    board.pieces_bb[piece.to_index() as usize].0 |= bit;
                    board.colors_bb[color.to_index()].0 |= bit;
                    board.combined_bb.0 |= bit;
                    board.mailbox[sq_idx] = encode_piece(piece, color);
                    file += 1;
                }
            }
        }

        // 2. side to move
        board.side = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(format!("FEN bad side '{}'", parts[1])),
        };

        // 3. castling
        let mut rights = 0u8;
        if parts[2] != "-" {
            for c in parts[2].chars() {
                match c {
                    'K' => rights |= WK,
                    'Q' => rights |= WQ,
                    'k' => rights |= BK,
                    'q' => rights |= BQ,
                    '-' => {}
                    _ => return Err(format!("FEN bad castling char '{}'", c)),
                }
            }
        }
        board.castling = rights;

        // 4. en passant: store the target square directly (FEN convention),
        //    but discard the marker when the side-to-move has no pawn that
        //    can actually capture en-passant — keeps Zobrist consistent.
        let raw_ep = if parts[3] == "-" {
            None
        } else {
            let bytes = parts[3].as_bytes();
            if bytes.len() != 2 {
                return Err(format!("FEN bad ep '{}'", parts[3]));
            }
            let f = bytes[0].wrapping_sub(b'a');
            let r = bytes[1].wrapping_sub(b'1');
            if f > 7 || r > 7 {
                return Err(format!("FEN ep out of range '{}'", parts[3]));
            }
            Some(Square::make_square(
                Rank::from_index(r as usize),
                File::from_index(f as usize),
            ))
        };
        board.en_passant = raw_ep.and_then(|ep_sq| {
            let mover_just_pushed = !board.side;
            let our_pawns = board.pieces_bb[Piece::Pawn.to_index() as usize].0
                & board.colors_bb[board.side.to_index()].0;
            if (pawn_attacks_u64(mover_just_pushed, ep_sq.to_index()) & our_pawns) != 0 {
                Some(ep_sq)
            } else {
                None
            }
        });

        // 5/6. clocks
        board.halfmove_clock = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        board.fullmove_number = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        // Zobrist hash
        let mut h = 0u64;
        for sq in 0..64u8 {
            if let Some((p, c)) = decode_piece(board.mailbox[sq as usize]) {
                h ^= key_piece(c, p, Square(sq));
            }
        }
        if board.side == Color::Black {
            h ^= key_side();
        }
        h ^= key_castling(board.castling);
        if let Some(ep) = board.en_passant {
            h ^= key_ep_file(ep.0 % 8);
        }
        board.hash = h;

        board.checkers_bb = compute_checkers(&board);
        Ok(board)
    }
}
