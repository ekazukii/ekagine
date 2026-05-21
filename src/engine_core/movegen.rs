use super::attacks::{
    between_u64, bishop_attacks_u64, king_attacks_u64, knight_attacks_u64, line_u64,
    pawn_attacks_u64, rook_attacks_u64,
};
use super::board::{is_sq_attacked_internal, Board, BK, BQ, WK, WQ};
use super::types::{BitBoard, ChessMove, Color, Piece, Square};
use smallvec::SmallVec;

const RANK_8: u64 = 0xFF00_0000_0000_0000;
const RANK_1: u64 = 0x0000_0000_0000_00FF;

/// Pseudo-legal move generator. Castling is generated with full legality
/// (path squares + destination must not be attacked). All other moves may
/// leave the side's king in check; callers must filter via
/// `Board::is_position_legal` after `make_move_new`, or use
/// `PinInfo::move_is_legal` for a pre-make legality test.
#[inline]
pub fn gen_pseudo_legal(board: &Board, moves: &mut SmallVec<[ChessMove; 64]>) {
    for_each_pseudo_legal(board, |mv| moves.push(mv));
}

/// Same as `gen_pseudo_legal` but with a callback — useful for hot loops
/// (perft / leaf counting) where allocating a SmallVec is wasteful.
#[inline]
pub fn for_each_pseudo_legal<F: FnMut(ChessMove)>(board: &Board, mut f: F) {
    let us = board.side_to_move();
    let them = !us;
    let our_pieces = board.color_combined(us).0;
    let their_pieces = board.color_combined(them).0;
    let occ = board.combined().0;
    let empty = !occ;

    let pawns = board.pieces(Piece::Pawn).0 & our_pieces;
    if pawns != 0 {
        match us {
            Color::White => pawn_white_cb(pawns, empty, their_pieces, board.en_passant(), &mut f),
            Color::Black => pawn_black_cb(pawns, empty, their_pieces, board.en_passant(), &mut f),
        }
    }

    let mut knights = board.pieces(Piece::Knight).0 & our_pieces;
    while knights != 0 {
        let from = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        let mut attacks = knight_attacks_u64(from as usize) & !our_pieces;
        while attacks != 0 {
            let to = attacks.trailing_zeros() as u8;
            attacks &= attacks - 1;
            f(ChessMove::new(Square(from), Square(to), None));
        }
    }

    let mut bishops = board.pieces(Piece::Bishop).0 & our_pieces;
    while bishops != 0 {
        let from = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        let mut attacks = bishop_attacks_u64(from as usize, occ) & !our_pieces;
        while attacks != 0 {
            let to = attacks.trailing_zeros() as u8;
            attacks &= attacks - 1;
            f(ChessMove::new(Square(from), Square(to), None));
        }
    }

    let mut rooks = board.pieces(Piece::Rook).0 & our_pieces;
    while rooks != 0 {
        let from = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        let mut attacks = rook_attacks_u64(from as usize, occ) & !our_pieces;
        while attacks != 0 {
            let to = attacks.trailing_zeros() as u8;
            attacks &= attacks - 1;
            f(ChessMove::new(Square(from), Square(to), None));
        }
    }

    let mut queens = board.pieces(Piece::Queen).0 & our_pieces;
    while queens != 0 {
        let from = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        let mut attacks =
            (bishop_attacks_u64(from as usize, occ) | rook_attacks_u64(from as usize, occ))
                & !our_pieces;
        while attacks != 0 {
            let to = attacks.trailing_zeros() as u8;
            attacks &= attacks - 1;
            f(ChessMove::new(Square(from), Square(to), None));
        }
    }

    let king_bb = board.pieces(Piece::King).0 & our_pieces;
    if king_bb != 0 {
        let king_sq = king_bb.trailing_zeros() as u8;
        let mut attacks = king_attacks_u64(king_sq as usize) & !our_pieces;
        while attacks != 0 {
            let to = attacks.trailing_zeros() as u8;
            attacks &= attacks - 1;
            f(ChessMove::new(Square(king_sq), Square(to), None));
        }

        let in_check = board.checkers().0 != 0;
        if !in_check {
            castling_cb(board, us, king_sq, occ, them, &mut f);
        }
    }
}

#[inline]
fn castling_cb<F: FnMut(ChessMove)>(
    board: &Board,
    us: Color,
    king_sq: u8,
    occ: u64,
    them: Color,
    f: &mut F,
) {
    let castling = board.castling_rights();
    let occ_no_king = occ ^ (1u64 << king_sq);
    match us {
        Color::White => {
            if (castling & WK) != 0 {
                let between = (1u64 << 5) | (1u64 << 6);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 5, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 6, them, occ_no_king)
                {
                    f(ChessMove::new(Square(king_sq), Square::G1, None));
                }
            }
            if (castling & WQ) != 0 {
                let between = (1u64 << 1) | (1u64 << 2) | (1u64 << 3);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 3, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 2, them, occ_no_king)
                {
                    f(ChessMove::new(Square(king_sq), Square::C1, None));
                }
            }
        }
        Color::Black => {
            if (castling & BK) != 0 {
                let between = (1u64 << 61) | (1u64 << 62);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 61, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 62, them, occ_no_king)
                {
                    f(ChessMove::new(Square(king_sq), Square::G8, None));
                }
            }
            if (castling & BQ) != 0 {
                let between = (1u64 << 57) | (1u64 << 58) | (1u64 << 59);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 59, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 58, them, occ_no_king)
                {
                    f(ChessMove::new(Square(king_sq), Square::C8, None));
                }
            }
        }
    }
}

#[inline]
fn pawn_white_cb<F: FnMut(ChessMove)>(
    pawns: u64,
    empty: u64,
    their_pieces: u64,
    ep: Option<Square>,
    f: &mut F,
) {
    let single = (pawns << 8) & empty;
    let mut bb = single & !RANK_8;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to - 8), Square(to), None));
    }
    let mut promo_push = single & RANK_8;
    while promo_push != 0 {
        let to = promo_push.trailing_zeros() as u8;
        promo_push &= promo_push - 1;
        promote_cb(Square(to - 8), Square(to), f);
    }
    let double = ((single & 0x0000_0000_00FF_0000) << 8) & empty;
    let mut bb = double;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to - 16), Square(to), None));
    }
    let ne_targets = ((pawns & 0x7F7F_7F7F_7F7F_7F7F) << 9) & their_pieces;
    let mut bb = ne_targets & !RANK_8;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to - 9), Square(to), None));
    }
    let mut bb = ne_targets & RANK_8;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        promote_cb(Square(to - 9), Square(to), f);
    }
    let nw_targets = ((pawns & 0xFEFE_FEFE_FEFE_FEFE) << 7) & their_pieces;
    let mut bb = nw_targets & !RANK_8;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to - 7), Square(to), None));
    }
    let mut bb = nw_targets & RANK_8;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        promote_cb(Square(to - 7), Square(to), f);
    }
    if let Some(ep_sq) = ep {
        let target = ep_sq.0;
        let target_bb = 1u64 << target;
        let ne_ep = ((pawns & 0x7F7F_7F7F_7F7F_7F7F) << 9) & target_bb;
        if ne_ep != 0 {
            f(ChessMove::new(Square(target - 9), ep_sq, None));
        }
        let nw_ep = ((pawns & 0xFEFE_FEFE_FEFE_FEFE) << 7) & target_bb;
        if nw_ep != 0 {
            f(ChessMove::new(Square(target - 7), ep_sq, None));
        }
    }
}

#[inline]
fn pawn_black_cb<F: FnMut(ChessMove)>(
    pawns: u64,
    empty: u64,
    their_pieces: u64,
    ep: Option<Square>,
    f: &mut F,
) {
    let single = (pawns >> 8) & empty;
    let mut bb = single & !RANK_1;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to + 8), Square(to), None));
    }
    let mut promo_push = single & RANK_1;
    while promo_push != 0 {
        let to = promo_push.trailing_zeros() as u8;
        promo_push &= promo_push - 1;
        promote_cb(Square(to + 8), Square(to), f);
    }
    let double = ((single & 0x0000_FF00_0000_0000) >> 8) & empty;
    let mut bb = double;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to + 16), Square(to), None));
    }
    let sw_targets = ((pawns & 0xFEFE_FEFE_FEFE_FEFE) >> 9) & their_pieces;
    let mut bb = sw_targets & !RANK_1;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to + 9), Square(to), None));
    }
    let mut bb = sw_targets & RANK_1;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        promote_cb(Square(to + 9), Square(to), f);
    }
    let se_targets = ((pawns & 0x7F7F_7F7F_7F7F_7F7F) >> 7) & their_pieces;
    let mut bb = se_targets & !RANK_1;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        f(ChessMove::new(Square(to + 7), Square(to), None));
    }
    let mut bb = se_targets & RANK_1;
    while bb != 0 {
        let to = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        promote_cb(Square(to + 7), Square(to), f);
    }
    if let Some(ep_sq) = ep {
        let target = ep_sq.0;
        let target_bb = 1u64 << target;
        let sw_ep = ((pawns & 0xFEFE_FEFE_FEFE_FEFE) >> 9) & target_bb;
        if sw_ep != 0 {
            f(ChessMove::new(Square(target + 9), ep_sq, None));
        }
        let se_ep = ((pawns & 0x7F7F_7F7F_7F7F_7F7F) >> 7) & target_bb;
        if se_ep != 0 {
            f(ChessMove::new(Square(target + 7), ep_sq, None));
        }
    }
}

#[inline(always)]
fn promote_cb<F: FnMut(ChessMove)>(from: Square, to: Square, f: &mut F) {
    f(ChessMove::new(from, to, Some(Piece::Queen)));
    f(ChessMove::new(from, to, Some(Piece::Rook)));
    f(ChessMove::new(from, to, Some(Piece::Bishop)));
    f(ChessMove::new(from, to, Some(Piece::Knight)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pin info — used to filter pseudo-legal moves without `make_move_new`.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct PinInfo {
    pub king_sq: u8,
    pub pinned: u64,
    pub checkers: u64,
    pub num_checkers: u32,
    pub check_evasion_mask: u64,
}

impl PinInfo {
    pub fn for_board(board: &Board) -> Self {
        let us = board.side_to_move();
        let them = !us;
        let our_pieces = board.color_combined(us).0;
        let their_pieces = board.color_combined(them).0;
        let occ = board.combined().0;
        let king_bb = board.pieces(Piece::King).0 & our_pieces;
        if king_bb == 0 {
            return PinInfo {
                king_sq: 0,
                pinned: 0,
                checkers: 0,
                num_checkers: 0,
                check_evasion_mask: !0u64,
            };
        }
        let king_sq = king_bb.trailing_zeros() as u8;
        let checkers = board.checkers().0;
        let num_checkers = checkers.count_ones();

        // Pinned: our pieces lying between king and an enemy slider, with
        // exactly one blocker on the ray.
        let mut pinned = 0u64;
        let their_orth = (board.pieces(Piece::Rook).0 | board.pieces(Piece::Queen).0) & their_pieces;
        let king_orth_xray = rook_attacks_u64(king_sq as usize, 0);
        let mut potential = king_orth_xray & their_orth;
        while potential != 0 {
            let pinner = potential.trailing_zeros() as usize;
            potential &= potential - 1;
            let between = between_u64(king_sq as usize, pinner);
            let blockers = between & occ;
            if blockers.count_ones() == 1 && (blockers & our_pieces) != 0 {
                pinned |= blockers;
            }
        }
        let their_diag = (board.pieces(Piece::Bishop).0 | board.pieces(Piece::Queen).0) & their_pieces;
        let king_diag_xray = bishop_attacks_u64(king_sq as usize, 0);
        let mut potential = king_diag_xray & their_diag;
        while potential != 0 {
            let pinner = potential.trailing_zeros() as usize;
            potential &= potential - 1;
            let between = between_u64(king_sq as usize, pinner);
            let blockers = between & occ;
            if blockers.count_ones() == 1 && (blockers & our_pieces) != 0 {
                pinned |= blockers;
            }
        }

        let check_evasion_mask = if num_checkers == 1 {
            let checker_sq = checkers.trailing_zeros() as usize;
            let cap = checkers; // capture the checker
            let block = if let Some(p) = board.piece_on(Square(checker_sq as u8)) {
                if matches!(p, Piece::Bishop | Piece::Rook | Piece::Queen) {
                    between_u64(king_sq as usize, checker_sq)
                } else {
                    0
                }
            } else {
                0
            };
            cap | block
        } else {
            !0u64
        };

        PinInfo {
            king_sq,
            pinned,
            checkers,
            num_checkers,
            check_evasion_mask,
        }
    }

    #[inline]
    pub fn move_is_legal(&self, board: &Board, mv: ChessMove) -> bool {
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        let from_bit = 1u64 << from;
        let to_bit = 1u64 << to;

        if from as u8 == self.king_sq {
            if (from as i32 - to as i32).abs() == 2 {
                return true;
            }
            let occ_no_king = board.combined().0 ^ from_bit;
            let them = !board.side_to_move();
            return !is_sq_attacked_internal(board, to, them, occ_no_king);
        }

        if self.num_checkers >= 2 {
            return false;
        }

        // En passant: pawn moving to the EP target square (empty).
        if Some(mv.get_dest()) == board.en_passant()
            && board.piece_on(mv.get_dest()).is_none()
            && board.piece_on(mv.get_source()) == Some(Piece::Pawn)
        {
            return self.ep_is_legal(board, mv);
        }

        if self.num_checkers == 1 && (self.check_evasion_mask & to_bit) == 0 {
            return false;
        }

        if (self.pinned & from_bit) != 0 {
            let line = line_u64(self.king_sq as usize, from);
            if (line & to_bit) == 0 {
                return false;
            }
        }

        true
    }

    fn ep_is_legal(&self, board: &Board, mv: ChessMove) -> bool {
        let us = board.side_to_move();
        let them = !us;
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        // Target convention: en_passant is the target; captured pawn sits
        // one rank back from the target (toward us).
        let captured_sq = match us {
            Color::White => to - 8,
            Color::Black => to + 8,
        };

        // Check pin line first: the pawn capturing diagonally must stay on a
        // pin line if it's pinned (other than the EP capture's own diagonal —
        // captured by line_bb covering destination too).
        let from_bit = 1u64 << from;
        if (self.pinned & from_bit) != 0 {
            let line = line_u64(self.king_sq as usize, from);
            if (line & (1u64 << to)) == 0 {
                return false;
            }
        }

        let our_pawn_bb = 1u64 << from;
        let captured_bb = 1u64 << captured_sq;
        let to_bb = 1u64 << to;

        // EP can resolve single check only if it captures the checker or
        // blocks. (The "block" case includes ep_sq being on the check ray.)
        if self.num_checkers == 1 {
            let captured_is_checker = (captured_bb & self.checkers) != 0;
            let ep_on_block = (to_bb & self.check_evasion_mask) != 0;
            if !captured_is_checker && !ep_on_block {
                return false;
            }
        }

        let new_occ = (board.combined().0 ^ our_pawn_bb ^ captured_bb) | to_bb;
        let their_pieces = board.color_combined(them).0 & !captured_bb;
        let king_sq = self.king_sq as usize;
        let their_bq = (board.pieces(Piece::Bishop).0 | board.pieces(Piece::Queen).0) & their_pieces;
        if (bishop_attacks_u64(king_sq, new_occ) & their_bq) != 0 {
            return false;
        }
        let their_rq = (board.pieces(Piece::Rook).0 | board.pieces(Piece::Queen).0) & their_pieces;
        if (rook_attacks_u64(king_sq, new_occ) & their_rq) != 0 {
            return false;
        }
        true
    }
}

/// Count legal moves at this position. Computes everything inline using
/// bitboard popcounts where possible, avoiding per-move enumeration.
#[inline(always)]
pub fn count_legal_moves(board: &Board) -> u64 {
    let us = board.side_to_move();
    let them = !us;
    let our_pieces = board.color_combined(us).0;
    let their_pieces = board.color_combined(them).0;
    let occ = board.combined().0;
    let king_bb = board.pieces(Piece::King).0 & our_pieces;
    if king_bb == 0 {
        return 0;
    }
    let king_sq = king_bb.trailing_zeros() as usize;
    let checkers = board.checkers().0;
    let num_checkers = checkers.count_ones();

    // King moves: precompute the set of squares attacked by the opponent
    // (with our king removed from occupancy) and exclude them in one shot,
    // instead of running `is_sq_attacked` per target.
    let mut count: u64 = 0;
    let occ_no_king = occ ^ king_bb;
    let enemy_attacks = compute_attack_set(board, them, occ_no_king);
    let king_targets = king_attacks_u64(king_sq) & !our_pieces & !enemy_attacks;
    count += king_targets.count_ones() as u64;

    // Castling
    if num_checkers == 0 {
        count += count_castling(board, us, king_sq as u8, occ, them);
    }

    if num_checkers >= 2 {
        return count;
    }

    // Compute allowed mask (check evasion).
    let allowed = if num_checkers == 1 {
        let checker_sq = checkers.trailing_zeros() as usize;
        let block = if let Some(p) = board.piece_on(Square(checker_sq as u8)) {
            if matches!(p, Piece::Bishop | Piece::Rook | Piece::Queen) {
                between_u64(king_sq, checker_sq)
            } else {
                0
            }
        } else {
            0
        };
        checkers | block
    } else {
        !0u64
    };

    // Compute pinned bitboard.
    let pinned = compute_pinned(board, king_sq, our_pieces, their_pieces, occ);

    // Knights (non-pinned only).
    let mut knights = board.pieces(Piece::Knight).0 & our_pieces & !pinned;
    while knights != 0 {
        let from = knights.trailing_zeros() as usize;
        knights &= knights - 1;
        let targets = knight_attacks_u64(from) & allowed & !our_pieces;
        count += targets.count_ones() as u64;
    }

    // Bishops (non-pinned: all moves; pinned: must stay on pin line).
    let bishops = board.pieces(Piece::Bishop).0 & our_pieces;
    let mut unpinned = bishops & !pinned;
    while unpinned != 0 {
        let from = unpinned.trailing_zeros() as usize;
        unpinned &= unpinned - 1;
        let targets = bishop_attacks_u64(from, occ) & allowed & !our_pieces;
        count += targets.count_ones() as u64;
    }
    let mut pinned_bishops = bishops & pinned;
    while pinned_bishops != 0 {
        let from = pinned_bishops.trailing_zeros() as usize;
        pinned_bishops &= pinned_bishops - 1;
        let targets = bishop_attacks_u64(from, occ) & allowed & !our_pieces & line_u64(king_sq, from);
        count += targets.count_ones() as u64;
    }

    // Rooks.
    let rooks = board.pieces(Piece::Rook).0 & our_pieces;
    let mut unpinned = rooks & !pinned;
    while unpinned != 0 {
        let from = unpinned.trailing_zeros() as usize;
        unpinned &= unpinned - 1;
        let targets = rook_attacks_u64(from, occ) & allowed & !our_pieces;
        count += targets.count_ones() as u64;
    }
    let mut pinned_rooks = rooks & pinned;
    while pinned_rooks != 0 {
        let from = pinned_rooks.trailing_zeros() as usize;
        pinned_rooks &= pinned_rooks - 1;
        let targets = rook_attacks_u64(from, occ) & allowed & !our_pieces & line_u64(king_sq, from);
        count += targets.count_ones() as u64;
    }

    // Queens.
    let queens = board.pieces(Piece::Queen).0 & our_pieces;
    let mut unpinned = queens & !pinned;
    while unpinned != 0 {
        let from = unpinned.trailing_zeros() as usize;
        unpinned &= unpinned - 1;
        let targets = (bishop_attacks_u64(from, occ) | rook_attacks_u64(from, occ))
            & allowed
            & !our_pieces;
        count += targets.count_ones() as u64;
    }
    let mut pinned_queens = queens & pinned;
    while pinned_queens != 0 {
        let from = pinned_queens.trailing_zeros() as usize;
        pinned_queens &= pinned_queens - 1;
        let targets = (bishop_attacks_u64(from, occ) | rook_attacks_u64(from, occ))
            & allowed
            & !our_pieces
            & line_u64(king_sq, from);
        count += targets.count_ones() as u64;
    }

    // Pawns.
    let pawns = board.pieces(Piece::Pawn).0 & our_pieces;
    if pawns != 0 {
        count += count_legal_pawns(board, pawns, occ, their_pieces, allowed, pinned, king_sq, us);
    }

    count
}

/// Union of all squares attacked by `by`. Sliding pieces are computed with
/// the supplied `occ` (so a caller can pass `occ ^ our_king` to capture
/// squares the king can't escape to).
#[inline(always)]
fn compute_attack_set(board: &Board, by: Color, occ: u64) -> u64 {
    let by_pieces = board.color_combined(by).0;
    let mut atk = 0u64;

    // Pawns: shift the whole pawn bitboard rather than iterating per pawn.
    let pawns = board.pieces(Piece::Pawn).0 & by_pieces;
    match by {
        Color::White => {
            atk |= (pawns & 0xFEFE_FEFE_FEFE_FEFE) << 7;
            atk |= (pawns & 0x7F7F_7F7F_7F7F_7F7F) << 9;
        }
        Color::Black => {
            atk |= (pawns & 0xFEFE_FEFE_FEFE_FEFE) >> 9;
            atk |= (pawns & 0x7F7F_7F7F_7F7F_7F7F) >> 7;
        }
    }

    let mut iter = board.pieces(Piece::Knight).0 & by_pieces;
    while iter != 0 {
        let sq = iter.trailing_zeros() as usize;
        iter &= iter - 1;
        atk |= knight_attacks_u64(sq);
    }

    let mut iter =
        (board.pieces(Piece::Bishop).0 | board.pieces(Piece::Queen).0) & by_pieces;
    while iter != 0 {
        let sq = iter.trailing_zeros() as usize;
        iter &= iter - 1;
        atk |= bishop_attacks_u64(sq, occ);
    }

    let mut iter =
        (board.pieces(Piece::Rook).0 | board.pieces(Piece::Queen).0) & by_pieces;
    while iter != 0 {
        let sq = iter.trailing_zeros() as usize;
        iter &= iter - 1;
        atk |= rook_attacks_u64(sq, occ);
    }

    let kings = board.pieces(Piece::King).0 & by_pieces;
    if kings != 0 {
        atk |= king_attacks_u64(kings.trailing_zeros() as usize);
    }

    atk
}

#[inline(always)]
fn compute_pinned(
    board: &Board,
    king_sq: usize,
    our_pieces: u64,
    their_pieces: u64,
    occ: u64,
) -> u64 {
    let mut pinned = 0u64;
    let their_orth =
        (board.pieces(Piece::Rook).0 | board.pieces(Piece::Queen).0) & their_pieces;
    let king_orth_xray = rook_attacks_u64(king_sq, 0);
    let mut potential = king_orth_xray & their_orth;
    while potential != 0 {
        let pinner = potential.trailing_zeros() as usize;
        potential &= potential - 1;
        let between = between_u64(king_sq, pinner);
        let blockers = between & occ;
        if blockers.count_ones() == 1 && (blockers & our_pieces) != 0 {
            pinned |= blockers;
        }
    }
    let their_diag =
        (board.pieces(Piece::Bishop).0 | board.pieces(Piece::Queen).0) & their_pieces;
    let king_diag_xray = bishop_attacks_u64(king_sq, 0);
    let mut potential = king_diag_xray & their_diag;
    while potential != 0 {
        let pinner = potential.trailing_zeros() as usize;
        potential &= potential - 1;
        let between = between_u64(king_sq, pinner);
        let blockers = between & occ;
        if blockers.count_ones() == 1 && (blockers & our_pieces) != 0 {
            pinned |= blockers;
        }
    }
    pinned
}

#[inline]
fn count_castling(board: &Board, us: Color, king_sq: u8, occ: u64, them: Color) -> u64 {
    let castling = board.castling_rights();
    let occ_no_king = occ ^ (1u64 << king_sq);
    let mut c = 0u64;
    match us {
        Color::White => {
            if (castling & WK) != 0 {
                let between = (1u64 << 5) | (1u64 << 6);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 5, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 6, them, occ_no_king)
                {
                    c += 1;
                }
            }
            if (castling & WQ) != 0 {
                let between = (1u64 << 1) | (1u64 << 2) | (1u64 << 3);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 3, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 2, them, occ_no_king)
                {
                    c += 1;
                }
            }
        }
        Color::Black => {
            if (castling & BK) != 0 {
                let between = (1u64 << 61) | (1u64 << 62);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 61, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 62, them, occ_no_king)
                {
                    c += 1;
                }
            }
            if (castling & BQ) != 0 {
                let between = (1u64 << 57) | (1u64 << 58) | (1u64 << 59);
                if (occ & between) == 0
                    && !is_sq_attacked_internal(board, 59, them, occ_no_king)
                    && !is_sq_attacked_internal(board, 58, them, occ_no_king)
                {
                    c += 1;
                }
            }
        }
    }
    c
}

#[inline]
fn count_legal_pawns(
    board: &Board,
    pawns: u64,
    occ: u64,
    their_pieces: u64,
    allowed: u64,
    pinned: u64,
    king_sq: usize,
    us: Color,
) -> u64 {
    let empty = !occ;
    let mut count = 0u64;
    let unpinned = pawns & !pinned;
    let pinned_pawns = pawns & pinned;

    let (push_shift, double_rank, promo_rank): (i32, u64, u64) = match us {
        Color::White => (8, 0x0000_0000_00FF_0000, RANK_8),
        Color::Black => (-8, 0x0000_FF00_0000_0000, RANK_1),
    };

    // Single pushes (unpinned).
    let push = if push_shift > 0 {
        (unpinned << 8) & empty
    } else {
        (unpinned >> 8) & empty
    };
    let push_allowed = push & allowed;
    let push_non_promo = push_allowed & !promo_rank;
    let push_promo = push_allowed & promo_rank;
    count += push_non_promo.count_ones() as u64;
    count += (push_promo.count_ones() as u64) * 4;

    // Double pushes (unpinned).
    let double_intermediate = push & double_rank;
    let double = if push_shift > 0 {
        (double_intermediate << 8) & empty
    } else {
        (double_intermediate >> 8) & empty
    };
    count += (double & allowed).count_ones() as u64;

    // Captures (unpinned).
    let (left_capture, right_capture) = match us {
        Color::White => (
            ((unpinned & 0xFEFE_FEFE_FEFE_FEFE) << 7) & their_pieces,
            ((unpinned & 0x7F7F_7F7F_7F7F_7F7F) << 9) & their_pieces,
        ),
        Color::Black => (
            ((unpinned & 0xFEFE_FEFE_FEFE_FEFE) >> 9) & their_pieces,
            ((unpinned & 0x7F7F_7F7F_7F7F_7F7F) >> 7) & their_pieces,
        ),
    };
    for cap in [left_capture, right_capture] {
        let cap_allowed = cap & allowed;
        let non_promo = cap_allowed & !promo_rank;
        let promo = cap_allowed & promo_rank;
        count += non_promo.count_ones() as u64;
        count += (promo.count_ones() as u64) * 4;
    }

    // Pinned pawns: count via line filter per pawn (rare).
    let mut iter = pinned_pawns;
    while iter != 0 {
        let from = iter.trailing_zeros() as usize;
        iter &= iter - 1;
        let from_bb = 1u64 << from;
        let line = line_u64(king_sq, from);

        // Push
        let push_sq = if push_shift > 0 {
            from + 8
        } else {
            from.wrapping_sub(8)
        };
        if push_sq < 64 {
            let push_bb = 1u64 << push_sq;
            if (push_bb & occ) == 0 && (push_bb & allowed & line) != 0 {
                if (push_bb & promo_rank) != 0 {
                    count += 4;
                } else {
                    count += 1;
                }
                // Double push (only from the pawn's starting rank).
                let start_rank_mask = match us {
                    Color::White => 0x0000_0000_0000_FF00,
                    Color::Black => 0x00FF_0000_0000_0000,
                };
                if (from_bb & start_rank_mask) != 0 {
                    let dbl_sq = if push_shift > 0 {
                        push_sq + 8
                    } else {
                        push_sq.wrapping_sub(8)
                    };
                    if dbl_sq < 64 {
                        let dbl_bb = 1u64 << dbl_sq;
                        if (dbl_bb & occ) == 0 && (dbl_bb & allowed & line) != 0 {
                            count += 1;
                        }
                    }
                }
            }
        }

        // Captures
        let attacks = pawn_attacks_u64(us, from);
        let cap_targets = attacks & their_pieces & allowed & line;
        let non_promo = cap_targets & !promo_rank;
        let promo = cap_targets & promo_rank;
        count += non_promo.count_ones() as u64;
        count += (promo.count_ones() as u64) * 4;
    }

    // En passant (rare, check legality individually).
    if let Some(ep_sq) = board.en_passant() {
        let target = ep_sq.to_index();
        let captured_sq = match us {
            Color::White => target - 8,
            Color::Black => target + 8,
        };
        let target_bb = 1u64 << target;
        let ep_attackers = match us {
            Color::White => {
                (((pawns & 0xFEFE_FEFE_FEFE_FEFE) << 7) & target_bb).wrapping_shr(7)
                    | (((pawns & 0x7F7F_7F7F_7F7F_7F7F) << 9) & target_bb).wrapping_shr(9)
            }
            Color::Black => {
                (((pawns & 0xFEFE_FEFE_FEFE_FEFE) >> 9) & target_bb).wrapping_shl(9)
                    | (((pawns & 0x7F7F_7F7F_7F7F_7F7F) >> 7) & target_bb).wrapping_shl(7)
            }
        };
        let mut it = ep_attackers;
        while it != 0 {
            let from = it.trailing_zeros() as usize;
            it &= it - 1;
            let mv = ChessMove::new(Square(from as u8), Square(target as u8), None);
            if ep_legal_inline(board, mv, king_sq, us, captured_sq, pinned) {
                count += 1;
            }
        }
    }

    count
}

#[inline]
fn ep_legal_inline(
    board: &Board,
    mv: ChessMove,
    king_sq: usize,
    us: Color,
    captured_sq: usize,
    pinned: u64,
) -> bool {
    let from = mv.get_source().to_index();
    let to = mv.get_dest().to_index();
    let from_bit = 1u64 << from;
    let to_bit = 1u64 << to;
    if (pinned & from_bit) != 0 {
        let line = line_u64(king_sq, from);
        if (line & to_bit) == 0 {
            return false;
        }
    }
    let our_pawn_bb = from_bit;
    let captured_bb = 1u64 << captured_sq;
    let new_occ = (board.combined().0 ^ our_pawn_bb ^ captured_bb) | to_bit;
    let them = !us;
    let their_pieces = board.color_combined(them).0 & !captured_bb;
    let their_bq = (board.pieces(Piece::Bishop).0 | board.pieces(Piece::Queen).0) & their_pieces;
    if (bishop_attacks_u64(king_sq, new_occ) & their_bq) != 0 {
        return false;
    }
    let their_rq = (board.pieces(Piece::Rook).0 | board.pieces(Piece::Queen).0) & their_pieces;
    if (rook_attacks_u64(king_sq, new_occ) & their_rq) != 0 {
        return false;
    }
    true
}

/// True iff there exists at least one legal move from this position.
pub fn any_legal_move(board: &Board) -> bool {
    let pin_info = PinInfo::for_board(board);
    let mut found = false;
    for_each_pseudo_legal(board, |mv| {
        if !found && pin_info.move_is_legal(board, mv) {
            found = true;
        }
    });
    found
}

// ─────────────────────────────────────────────────────────────────────────────
// MoveGen — pseudo-legal iterator with set_iterator_mask support
// ─────────────────────────────────────────────────────────────────────────────

pub struct MoveGen {
    moves: SmallVec<[ChessMove; 64]>,
    consumed: usize,
    matching_end: usize,
    mask: BitBoard,
}

impl MoveGen {
    pub fn new_legal(board: &Board) -> Self {
        let mut moves: SmallVec<[ChessMove; 64]> = SmallVec::new();
        gen_pseudo_legal(board, &mut moves);
        let total = moves.len();
        Self {
            moves,
            consumed: 0,
            matching_end: total,
            mask: BitBoard(!0u64),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.matching_end.saturating_sub(self.consumed)
    }

    pub fn set_iterator_mask(&mut self, new_mask: BitBoard) {
        // Stable partition of moves[consumed..]: matching first, then
        // non-matching, preserving the original relative order in each
        // bucket.
        let total = self.moves.len();
        let start = self.consumed;
        let mask_bits = new_mask.0;
        let mut matching: SmallVec<[ChessMove; 64]> = SmallVec::new();
        let mut other: SmallVec<[ChessMove; 64]> = SmallVec::new();
        for i in start..total {
            let mv = self.moves[i];
            let dest_bit = 1u64 << mv.get_dest().to_index();
            if (dest_bit & mask_bits) != 0 {
                matching.push(mv);
            } else {
                other.push(mv);
            }
        }
        let mat_len = matching.len();
        for (i, &mv) in matching.iter().enumerate() {
            self.moves[start + i] = mv;
        }
        for (i, &mv) in other.iter().enumerate() {
            self.moves[start + mat_len + i] = mv;
        }
        self.matching_end = start + mat_len;
        self.mask = new_mask;
    }
}

impl Iterator for MoveGen {
    type Item = ChessMove;
    #[inline]
    fn next(&mut self) -> Option<ChessMove> {
        if self.consumed < self.matching_end {
            let mv = self.moves[self.consumed];
            self.consumed += 1;
            Some(mv)
        } else {
            None
        }
    }
}
