use std::ops::Not;
use std::time::Instant;

#[repr(u8)]
#[derive(Copy, Clone)]
pub enum Color {
    White,
    Black,
}

impl Not for Color {
    type Output = Color;

    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}


#[inline]
fn lsb(b: u64) -> u32 { b.trailing_zeros() } // safe for b != 0

#[inline]
fn pop_lsb(bb: &mut u64) -> u32 {
    let i = bb.trailing_zeros();
    *bb &= *bb - 1;
    i
}

const A_FILE: u64 = 0x0101_0101_0101_0101;
const H_FILE: u64 = 0x8080_8080_8080_8080;
const NOT_A_FILE: u64 = !A_FILE;
const NOT_H_FILE: u64 = !H_FILE;

const RANK_2: u64 = 0x0000_0000_0000_FF00;
const RANK_7: u64 = 0x00FF_0000_0000_0000;
const RANK_8: u64 = 0xFF00_0000_0000_0000;
const RANK_1: u64 = 0x0000_0000_0000_00FF;

#[repr(u8)]
#[derive(Copy, Clone)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

const ALL_PIECE: [Piece; 6] = [
    Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King
];

#[derive(Copy, Clone)]
pub enum CastleRights {
    NoRights = 0,
    KingSide = 1,
    QueenSide = 2,
    BothSide = 3,
}

#[derive(Copy, Clone)]
pub struct Square(u8);

impl Square {
    pub fn to_int(self) -> u8 {
        self.0
    }
}

#[derive(Copy, Clone)]
pub struct ChessMove {
    source: Square,
    destination: Square,
    promotion: Option<Piece>,
    is_ep: bool,
    is_castling: bool,
    is_pushing_two: bool,
}

#[derive(Copy, Clone)]
pub struct Position {
    piece_bb: [u64; 6],
    side_bb: [u64; 2],
    side_to_move: Color,
    hash: u64,
    en_passant_square: Option<Square>
}

/* =========================
   Leaper attacks (no tables)
   ========================= */

#[inline]
fn knight_attacks_bb(sq: u8) -> u64 {
    let b = 1u64 << sq;
    let l1 = (b & NOT_H_FILE) << 1;
    let l2 = (b & NOT_H_FILE & NOT_H_FILE) << 2;
    let r1 = (b & NOT_A_FILE) >> 1;
    let r2 = (b & NOT_A_FILE & NOT_A_FILE) >> 2;

    ((l2 | r2) << 8) | ((l2 | r2) >> 8) | ((l1 << 16) | (l1 >> 16)) | ((r1 << 16) | (r1 >> 16))
}

#[inline]
fn king_attacks_bb(sq: u8) -> u64 {
    let b = 1u64 << sq;
    let h = (b & NOT_H_FILE) << 1 | (b & NOT_A_FILE) >> 1;
    let v = (b << 8) | (b >> 8);
    h | v | ((h << 8) | (h >> 8))
}

/* =========================
   Slider attacks via scans
   Replace with magics later
   ========================= */

#[inline]
fn rook_attacks_from(sq: u8, occ: u64) -> u64 {
    let mut attacks = 0u64;
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;

    // north
    let mut rr = r + 1;
    while rr <= 7 {
        let s = (rr * 8 + f) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        rr += 1;
    }
    // south
    let mut rr = r - 1;
    while rr >= 0 {
        let s = (rr * 8 + f) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        rr -= 1;
    }
    // east
    let mut ff = f + 1;
    while ff <= 7 {
        let s = (r * 8 + ff) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        ff += 1;
    }
    // west
    let mut ff = f - 1;
    while ff >= 0 {
        let s = (r * 8 + ff) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        ff -= 1;
    }

    attacks
}

#[inline]
fn bishop_attacks_from(sq: u8, occ: u64) -> u64 {
    let mut attacks = 0u64;
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;

    // NE
    let (mut rr, mut ff) = (r + 1, f + 1);
    while rr <= 7 && ff <= 7 {
        let s = (rr * 8 + ff) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        rr += 1; ff += 1;
    }
    // NW
    let (mut rr, mut ff) = (r + 1, f - 1);
    while rr <= 7 && ff >= 0 {
        let s = (rr * 8 + ff) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        rr += 1; ff -= 1;
    }
    // SE
    let (mut rr, mut ff) = (r - 1, f + 1);
    while rr >= 0 && ff <= 7 {
        let s = (rr * 8 + ff) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        rr -= 1; ff += 1;
    }
    // SW
    let (mut rr, mut ff) = (r - 1, f - 1);
    while rr >= 0 && ff >= 0 {
        let s = (rr * 8 + ff) as u8;
        attacks |= 1u64 << s;
        if (occ >> s) & 1 != 0 { break; }
        rr -= 1; ff -= 1;
    }

    attacks
}

fn bitboard_from_square(square: Square) -> u64 {
    1u64 << square.to_int()
}

impl Position {
    pub fn get_combined_bb(&self) -> u64 {
        self.side_bb[0] | self.side_bb[1]
    }

    pub fn piece_on(&self, square: Square) -> Option<Piece> {
        let piece_bb = bitboard_from_square(square);
        let combined_bb = self.get_combined_bb();
        if piece_bb & combined_bb == 0 {
            return None;
        }

        for p in ALL_PIECE {
            if self.piece_bb[p as usize] & piece_bb != 0 {
                return Some(p);
            }
        }

        panic!("Should have a piece on square but not found :'(");
    }

    pub fn xor(&mut self, bb: u64, piece: Piece, col: Color) {
        self.piece_bb[piece as usize] ^= bb;
        self.side_bb[col as usize] ^= bb;
    }

    pub fn make_move_new(&self, cmove: ChessMove) -> Position {
        let mut new_board = *self;
        new_board.en_passant_square = None;

        let source_bb = bitboard_from_square(cmove.source);
        let dest_bb = bitboard_from_square(cmove.destination);
        let moved = self.piece_on(cmove.source).expect("No piece on source");

        new_board.xor(source_bb, moved, self.side_to_move);
        new_board.xor(dest_bb, moved, self.side_to_move);

        if cmove.is_ep {
            let dst = cmove.destination.to_int();
            let captured_sq = match self.side_to_move {
                Color::White => dst.wrapping_sub(8),
                Color::Black => dst.wrapping_add(8),
            };
            let captured_bb = 1u64 << captured_sq;
            new_board.xor(captured_bb, Piece::Pawn, !self.side_to_move);
        } else if let Some(captured) = self.piece_on(cmove.destination) {
            new_board.xor(dest_bb, captured, !self.side_to_move);
        }

        if let Some(promoted) = cmove.promotion {
            new_board.xor(dest_bb, Piece::Pawn, self.side_to_move);
            new_board.xor(dest_bb, promoted, self.side_to_move);
        }

        if cmove.is_castling {
            let src = cmove.source.to_int();
            let dst = cmove.destination.to_int();

            // Hypothèse d’indexation classique type a1 = 0, h1 = 7, a8 = 56, h8 = 63
            // Roque côté roi: le roi va de e? à g? => la tour va de h? à f?
            // Roque côté dame: le roi va de e? à c? => la tour va de a? à d?
            let (rook_src, rook_dst) = if dst > src {
                // côté roi
                (dst + 1, dst - 1)
            } else {
                // côté dame
                (dst - 2, dst + 1)
            };
            let rook_src_bb = 1u64 << rook_src;
            let rook_dst_bb = 1u64 << rook_dst;
            new_board.xor(rook_src_bb, Piece::Rook, self.side_to_move);
            new_board.xor(rook_dst_bb, Piece::Rook, self.side_to_move);
        }

        // Poussée de deux cases: enregistrer la case en passant
        if cmove.is_pushing_two {
            let src = cmove.source.to_int();
            let ep_sq = match self.side_to_move {
                Color::White => src + 8,
                Color::Black => src.wrapping_sub(8),
            };
            new_board.en_passant_square = Some(Square(ep_sq));
        }

        new_board.side_to_move = !self.side_to_move;

        new_board
    }

    #[inline]
    fn occ(&self) -> u64 { self.side_bb[0] | self.side_bb[1] }
    #[inline]
    fn ours(&self) -> u64 { self.side_bb[self.side_to_move as usize] }
    #[inline]
    fn theirs(&self) -> u64 { self.side_bb[(!self.side_to_move) as usize] }

    #[inline]
    fn bb_piece_of(&self, piece: Piece, side: Color) -> u64 {
        self.piece_bb[piece as usize] & self.side_bb[side as usize]
    }

    /* =========================
       Pseudo legal generation
       ========================= */
    pub fn generate_pseudo_legal_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::with_capacity(96); // typical middlegame bound
        let stm = self.side_to_move;
        let occ = self.occ();
        let empty = !occ;
        let ours = self.ours();
        let theirs = self.theirs();

        /* Pawns: vectorized pushes and captures, then enumerate */
        let pawns = self.bb_piece_of(Piece::Pawn, stm);

        match stm {
            Color::White => {
                // single pushes
                let single_push = ((pawns << 8) & empty) & !0; // within board
                // promotions on rank 8
                let promo_push = single_push & RANK_8;
                let quiet_push = single_push & !RANK_8;

                // enumerate quiet single pushes
                let mut bb = quiet_push;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst - 8;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: false,
                    });
                }
                // enumerate promotion pushes
                let mut bb = promo_push;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst - 8;
                    for promo in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                        moves.push(ChessMove {
                            source: Square(src),
                            destination: Square(dst),
                            promotion: Some(promo),
                            is_ep: false,
                            is_castling: false,
                            is_pushing_two: false,
                        });
                    }
                }
                // double pushes from rank 2
                let rank3 = ((pawns & RANK_2) << 8) & empty;
                let double_push = (rank3 << 8) & empty;
                let mut bb = double_push;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst - 16;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: true,
                    });
                }
                // captures
                let cap_left = ((pawns & NOT_A_FILE) << 7) & theirs;
                let cap_right = ((pawns & NOT_H_FILE) << 9) & theirs;

                // left captures
                let mut bb = cap_left & !RANK_8;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst - 7;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: false,
                    });
                }
                // right captures
                let mut bb = cap_right & !RANK_8;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst - 9;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: false,
                    });
                }
                // promotion captures
                let mut bb = (cap_left | cap_right) & RANK_8;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = if ((1u64 << dst) & cap_left) != 0 { dst - 7 } else { dst - 9 };
                    for promo in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                        moves.push(ChessMove {
                            source: Square(src),
                            destination: Square(dst),
                            promotion: Some(promo),
                            is_ep: false,
                            is_castling: false,
                            is_pushing_two: false,
                        });
                    }
                }
                // en passant
                if let Some(ep) = self.en_passant_square {
                    let epb = 1u64 << ep.to_int();
                    // sources that can hit ep
                    let src_l = ((epb >> 7) & NOT_H_FILE) & pawns; // from ep-7
                    let src_r = ((epb >> 9) & NOT_A_FILE) & pawns; // from ep-9
                    let mut bb = src_l | src_r;
                    while bb != 0 {
                        let src = pop_lsb(&mut bb) as u8;
                        moves.push(ChessMove {
                            source: Square(src),
                            destination: ep,
                            promotion: None,
                            is_ep: true,
                            is_castling: false,
                            is_pushing_two: false,
                        });
                    }
                }
            }
            Color::Black => {
                // single pushes
                let single_push = ((pawns >> 8) & empty) & !0;
                let promo_push = single_push & RANK_1;
                let quiet_push = single_push & !RANK_1;

                let mut bb = quiet_push;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst + 8;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: false,
                    });
                }
                let mut bb = promo_push;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst + 8;
                    for promo in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                        moves.push(ChessMove {
                            source: Square(src),
                            destination: Square(dst),
                            promotion: Some(promo),
                            is_ep: false,
                            is_castling: false,
                            is_pushing_two: false,
                        });
                    }
                }
                let rank6 = ((pawns & RANK_7) >> 8) & empty;
                let double_push = (rank6 >> 8) & empty;
                let mut bb = double_push;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst + 16;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: true,
                    });
                }
                // captures
                let cap_left = ((pawns & NOT_A_FILE) >> 9) & theirs;
                let cap_right = ((pawns & NOT_H_FILE) >> 7) & theirs;

                let mut bb = cap_left & !RANK_1;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst + 9;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: false,
                    });
                }
                let mut bb = cap_right & !RANK_1;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = dst + 7;
                    moves.push(ChessMove {
                        source: Square(src),
                        destination: Square(dst),
                        promotion: None,
                        is_ep: false,
                        is_castling: false,
                        is_pushing_two: false,
                    });
                }
                // promotion captures
                let mut bb = (cap_left | cap_right) & RANK_1;
                while bb != 0 {
                    let dst = pop_lsb(&mut bb) as u8;
                    let src = if ((1u64 << dst) & cap_left) != 0 { dst + 9 } else { dst + 7 };
                    for promo in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
                        moves.push(ChessMove {
                            source: Square(src),
                            destination: Square(dst),
                            promotion: Some(promo),
                            is_ep: false,
                            is_castling: false,
                            is_pushing_two: false,
                        });
                    }
                }
                // en passant
                if let Some(ep) = self.en_passant_square {
                    let epb = 1u64 << ep.to_int();
                    let src_l = ((epb << 9) & NOT_H_FILE) & pawns; // from ep+9
                    let src_r = ((epb << 7) & NOT_A_FILE) & pawns; // from ep+7
                    let mut bb = src_l | src_r;
                    while bb != 0 {
                        let src = pop_lsb(&mut bb) as u8;
                        moves.push(ChessMove {
                            source: Square(src),
                            destination: ep,
                            promotion: None,
                            is_ep: true,
                            is_castling: false,
                            is_pushing_two: false,
                        });
                    }
                }
            }
        }

        /* Knights */
        let mut bb = self.bb_piece_of(Piece::Knight, stm);
        while bb != 0 {
            let src = pop_lsb(&mut bb) as u8;
            let mut targets = knight_attacks_bb(src) & !ours;
            while targets != 0 {
                let dst = pop_lsb(&mut targets) as u8;
                moves.push(ChessMove {
                    source: Square(src),
                    destination: Square(dst),
                    promotion: None,
                    is_ep: false,
                    is_castling: false,
                    is_pushing_two: false,
                });
            }
        }

        /* Bishops */
        let mut bb = self.bb_piece_of(Piece::Bishop, stm);
        while bb != 0 {
            let src = pop_lsb(&mut bb) as u8;
            let mut targets = bishop_attacks_from(src, occ) & !ours;
            while targets != 0 {
                let dst = pop_lsb(&mut targets) as u8;
                moves.push(ChessMove {
                    source: Square(src),
                    destination: Square(dst),
                    promotion: None,
                    is_ep: false,
                    is_castling: false,
                    is_pushing_two: false,
                });
            }
        }

        /* Rooks */
        let mut bb = self.bb_piece_of(Piece::Rook, stm);
        while bb != 0 {
            let src = pop_lsb(&mut bb) as u8;
            let mut targets = rook_attacks_from(src, occ) & !ours;
            while targets != 0 {
                let dst = pop_lsb(&mut targets) as u8;
                moves.push(ChessMove {
                    source: Square(src),
                    destination: Square(dst),
                    promotion: None,
                    is_ep: false,
                    is_castling: false,
                    is_pushing_two: false,
                });
            }
        }

        /* Queens */
        let mut bb = self.bb_piece_of(Piece::Queen, stm);
        while bb != 0 {
            let src = pop_lsb(&mut bb) as u8;
            let mut targets = (rook_attacks_from(src, occ) | bishop_attacks_from(src, occ)) & !ours;
            while targets != 0 {
                let dst = pop_lsb(&mut targets) as u8;
                moves.push(ChessMove {
                    source: Square(src),
                    destination: Square(dst),
                    promotion: None,
                    is_ep: false,
                    is_castling: false,
                    is_pushing_two: false,
                });
            }
        }

        /* King (castling omitted here because Position lacks castling rights) */
        let mut bb = self.bb_piece_of(Piece::King, stm);
        if bb != 0 {
            let src = lsb(bb) as u8;
            let mut targets = king_attacks_bb(src) & !ours;
            while targets != 0 {
                let dst = pop_lsb(&mut targets) as u8;
                moves.push(ChessMove {
                    source: Square(src),
                    destination: Square(dst),
                    promotion: None,
                    is_ep: false,
                    is_castling: false, // fill when you add castling rights
                    is_pushing_two: false,
                });
            }
        }

        moves
    }

    /* =========================
       Attack and legality
       ========================= */

    #[inline]
    fn is_square_attacked_by(&self, sq: Square, by: Color) -> bool {
        let occ = self.occ();
        let target_bb = 1u64 << sq.to_int();

        // Pawns
        let pawns = self.bb_piece_of(Piece::Pawn, by);
        let pawn_attacks = match by {
            Color::White => {
                ((pawns & NOT_A_FILE) << 7) | ((pawns & NOT_H_FILE) << 9)
            }
            Color::Black => {
                ((pawns & NOT_A_FILE) >> 9) | ((pawns & NOT_H_FILE) >> 7)
            }
        };
        if pawn_attacks & target_bb != 0 { return true; }

        // Knights
        let mut bb = self.bb_piece_of(Piece::Knight, by);
        while bb != 0 {
            let s = pop_lsb(&mut bb) as u8;
            if knight_attacks_bb(s) & target_bb != 0 { return true; }
        }

        // Bishops and Queens (diagonals)
        let mut bb = (self.bb_piece_of(Piece::Bishop, by) | self.bb_piece_of(Piece::Queen, by));
        while bb != 0 {
            let s = pop_lsb(&mut bb) as u8;
            if bishop_attacks_from(s, occ) & target_bb != 0 { return true; }
        }

        // Rooks and Queens (orthogonals)
        let mut bb = (self.bb_piece_of(Piece::Rook, by) | self.bb_piece_of(Piece::Queen, by));
        while bb != 0 {
            let s = pop_lsb(&mut bb) as u8;
            if rook_attacks_from(s, occ) & target_bb != 0 { return true; }
        }

        // King
        let king = self.bb_piece_of(Piece::King, by);
        if king != 0 {
            let s = lsb(king) as u8;
            if king_attacks_bb(s) & target_bb != 0 { return true; }
        }

        false
    }

    #[inline]
    fn king_square(&self, side: Color) -> Square {
        let kbb = self.bb_piece_of(Piece::King, side);
        debug_assert!(kbb != 0, "king missing");
        Square(lsb(kbb) as u8)
    }

    #[inline]
    pub fn in_check(&self, side: Color) -> bool {
        let ks = self.king_square(side);
        self.is_square_attacked_by(ks, !side)
    }

    pub fn is_legal_move(&self, m: ChessMove) -> bool {
        let nb = self.make_move_new(m);
        let moved_side = self.side_to_move;
        !nb.in_check(moved_side)
    }

    pub fn startpos() -> Position {
        let wp = 0x0000_0000_0000_FF00u64;
        let bp = 0x00FF_0000_0000_0000u64;
        let wn = 0x0000_0000_0000_0042u64;
        let bn = 0x4200_0000_0000_0000u64;
        let wb = 0x0000_0000_0000_0024u64;
        let bb = 0x2400_0000_0000_0000u64;
        let wr = 0x0000_0000_0000_0081u64;
        let br = 0x8100_0000_0000_0000u64;
        let wq = 0x0000_0000_0000_0008u64;
        let bq = 0x0800_0000_0000_0000u64;
        let wk = 0x0000_0000_0000_0010u64;
        let bk = 0x1000_0000_0000_0000u64;

        let piece_bb = [
            wp | bp,
            wn | bn,
            wb | bb,
            wr | br,
            wq | bq,
            wk | bk,
        ];

        let white_occ = wp | wn | wb | wr | wq | wk;
        let black_occ = bp | bn | bb | br | bq | bk;

        Position {
            piece_bb,
            side_bb: [white_occ, black_occ],
            side_to_move: Color::White,
            hash: 0,
            en_passant_square: None,
        }
    }
}


pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    let moves = pos.generate_pseudo_legal_moves();

    for m in moves {
        if pos.is_legal_move(m) {
            let child = pos.make_move_new(m);
            nodes += perft(&child, depth - 1);
        }
    }
    nodes
}

pub fn perft_timed(pos: &Position, depth: u32) -> u64 {
    let t0 = Instant::now();
    let nodes = perft(pos, depth);
    let dt = t0.elapsed();
    let secs = dt.as_secs_f64().max(1e-9); // évite division par zéro
    let nps = (nodes as f64) / secs;

    println!(
        "perft depth {}: {} nodes en {:.3} s, {:.0} nodes/s",
        depth, nodes, secs, nps
    );
    nodes
}

pub fn perft_from_start_timed(depth: u32) -> u64 {
    let pos = Position::startpos();
    perft_timed(&pos, depth)
}