
/* New helper: compute endgame progress as an integer between 0 and 100 */
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use chess::{BitBoard, Board, BoardStatus, ChessMove, Color, Piece, Square};
use chess::Color::{Black, White};
use crate::{TranspositionTable, EVAL_COUNT, NEG_INFINITY, POS_INFINITY};


/// Evaluate a set of bitboard squares using a piece‐square table + base value.
fn pieces_type_eval(bitboard: &chess::BitBoard, table: &[i32; 64], base_val: i32) -> i32 {
    let mut total = 0;
    for square in *bitboard {
        let idx = square.to_index();
        total += table[idx] + base_val;
    }
    total
}


/// Flip a 64‐element array "vertically" (i.e. mirror across rank 4⇄5).
fn flip_vertical(table: &[i32; 64]) -> [i32; 64] {
    let mut flipped = [0; 64];
    for rank in 0..8 {
        for file in 0..8 {
            // rank 0 ↔ 7, 1 ↔ 6, etc.
            let src_index = rank * 8 + file;
            let dst_index = (7 - rank) * 8 + file;
            flipped[dst_index] = table[src_index];
        }
    }
    flipped
}

// ─────────────────────────────────────────────────────────────────────────────
// Piece‐Square Tables (identical values to the Python version).
// We define the black‐oriented arrays, then derive white by flipping.
// ─────────────────────────────────────────────────────────────────────────────

const PAWNS_VALUE_MAPPING_BLACK: [i32; 64] = [
    0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    10,  10,  20,  30,  30,  20,  10,  10,
    5,   5,  10,  25,  25,  10,   5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    5,  10,  10, -20, -20,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0,
];
lazy_static::lazy_static! {
    static ref PAWNS_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&PAWNS_VALUE_MAPPING_BLACK);
}

const ROOKS_VALUE_MAPPING_BLACK: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0,
];
lazy_static::lazy_static! {
    static ref ROOKS_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&ROOKS_VALUE_MAPPING_BLACK);
}

const KNIGHTS_VALUE_MAPPING_BLACK: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];
lazy_static::lazy_static! {
    static ref KNIGHTS_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&KNIGHTS_VALUE_MAPPING_BLACK);
}

const BISHOPS_VALUE_MAPPING_BLACK: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];
lazy_static::lazy_static! {
    static ref BISHOPS_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&BISHOPS_VALUE_MAPPING_BLACK);
}

const QUEENS_VALUE_MAPPING_BLACK: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,   0,  5,  5,  5,  5,  0, -5,
    0,   0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];
lazy_static::lazy_static! {
    static ref QUEENS_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&QUEENS_VALUE_MAPPING_BLACK);
}

const KING_START_VALUE_MAPPING_BLACK: [i32; 64] = [
    -80, -70, -70, -70, -70, -70, -70, -80,
    -60, -60, -60, -60, -60, -60, -60, -60,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,  -5,  -5,  -5,  -5,  20,  20,
    20,  30,  10,   0,   0,  10,  30,  20,
];
lazy_static::lazy_static! {
    static ref KING_START_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&KING_START_VALUE_MAPPING_BLACK);
}

const KING_END_VALUE_MAPPING_BLACK: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -10, -5,   20,  30,  30,  20,  -5, -10,
    -15, -10,  35,  45,  45,  35, -10, -15,
    -20, -15,  30,  40,  40,  30, -15, -20,
    -25, -20,  20,  25,  25,  20, -20, -25,
    -30, -25,   0,   0,   0,   0, -25, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
];
lazy_static::lazy_static! {
    static ref KING_END_VALUE_MAPPING_WHITE: [i32; 64] =
        flip_vertical(&KING_END_VALUE_MAPPING_BLACK);
}

// Base piece values
const PAWN_BASE_VAL: i32   = 100;
const KNIGHT_BASE_VAL: i32 = 300;
const BISHOP_BASE_VAL: i32 = 300;
const ROOK_BASE_VAL: i32   = 500;
const QUEEN_BASE_VAL: i32  = 900;
const BISHOP_PAIR_VAL: i32 = 30;
const STARTING_MATERIALS: i32 = (PAWN_BASE_VAL * 8 + KNIGHT_BASE_VAL * 2 + BISHOP_BASE_VAL * 2 + ROOK_BASE_VAL * 2 + QUEEN_BASE_VAL) * 2;



fn endgame_progress(board: &Board) -> i32 {
    let white_pawns = (*board.pieces(Piece::Pawn) & board.color_combined(Color::White)).into_iter().count() as i32;
    let white_knights = (*board.pieces(Piece::Knight) & board.color_combined(Color::White)).into_iter().count() as i32;
    let white_bishops = (*board.pieces(Piece::Bishop) & board.color_combined(Color::White)).into_iter().count() as i32;
    let white_rooks = (*board.pieces(Piece::Rook) & board.color_combined(Color::White)).into_iter().count() as i32;
    let white_queens = (*board.pieces(Piece::Queen) & board.color_combined(Color::White)).into_iter().count() as i32;

    let black_pawns = (*board.pieces(Piece::Pawn) & board.color_combined(Color::Black)).into_iter().count() as i32;
    let black_knights = (*board.pieces(Piece::Knight) & board.color_combined(Color::Black)).into_iter().count() as i32;
    let black_bishops = (*board.pieces(Piece::Bishop) & board.color_combined(Color::Black)).into_iter().count() as i32;
    let black_rooks = (*board.pieces(Piece::Rook) & board.color_combined(Color::Black)).into_iter().count() as i32;
    let black_queens = (*board.pieces(Piece::Queen) & board.color_combined(Color::Black)).into_iter().count() as i32;

    let white_material = white_pawns * PAWN_BASE_VAL +
        white_knights * KNIGHT_BASE_VAL +
        white_bishops * BISHOP_BASE_VAL +
        white_rooks * ROOK_BASE_VAL +
        white_queens * QUEEN_BASE_VAL;
    let black_material = black_pawns * PAWN_BASE_VAL +
        black_knights * KNIGHT_BASE_VAL +
        black_bishops * BISHOP_BASE_VAL +
        black_rooks * ROOK_BASE_VAL +
        black_queens * QUEEN_BASE_VAL;

    let current_total = white_material + black_material;
    let material_exchanged = STARTING_MATERIALS - current_total;
    let progress = (material_exchanged * 100) / STARTING_MATERIALS;

    if progress < 0 {
        0
    } else if progress > 100 {
        100
    } else {
        progress
    }
}

/// Evaluate king safety for a given king on a board. This function returns
/// a centipawn adjustment: positive values mean "safer" whereas negative values
/// indicate king exposure.
fn king_safety_eval(board: &Board, king_sq: Square, our_color: Color) -> i32 {
    const NEIGHBORS: &[(i8,i8)] = &[
        (-1,-1), (0,-1), (1,-1),
        (-1, 0),         (1, 0),
        (-1, 1), (0, 1), (1, 1),
    ];

    let mut friendly_shield = 0;
    let mut enemy_encounters = 0;
    let enemy_color = if our_color == Color::White { Color::Black } else { Color::White };

    let file_i = king_sq.get_file().to_index() as i8;
    let rank_i = king_sq.get_rank().to_index() as i8;

    for &(df, dr) in NEIGHBORS {
        let f = file_i + df;
        let r = rank_i + dr;
        if !(0..8).contains(&f) || !(0..8).contains(&r) {
            continue;
        }
        let sq = Square::make_square(
            chess::Rank::from_index(r as usize),
            chess::File::from_index(f as usize),
        );
        match board.piece_on(sq) {
            Some(Piece::Pawn) if board.color_on(sq) == Some(our_color) => {
                friendly_shield += 1;
            }
            Some(_) if board.color_on(sq) == Some(enemy_color) => {
                enemy_encounters += 1;
            }
            _ => {}
        }
    }

    // Parameters tuned in centipawns
    let bonus_per_shield = 15;
    let penalty_per_threat = 20;

    let mut score = friendly_shield * bonus_per_shield
        - enemy_encounters * penalty_per_threat;

    // Bonus if king is on its original back rank (i.e. likely castled or still safe)
    let back_rank = if our_color == Color::White { chess::Rank::First } else { chess::Rank::Eighth };
    if king_sq.get_rank() == back_rank {
        score += 20;
    }

    score
}

/// Compute a static evaluation for `board`. If game over, return ±10000 or 0.
/// Otherwise do a material + piece‐square evaluation.
pub fn eval_board(board: &Board) -> i32 {
    EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
    match board.status() {
        BoardStatus::Checkmate => {
            if board.side_to_move() == Color::White {
                NEG_INFINITY
            } else {
                POS_INFINITY
            }
        }

        BoardStatus::Stalemate => 0,
        BoardStatus::Ongoing => {
            let mut total = 0;

            let white_pawns:   BitBoard = *board.pieces(Piece::Pawn)   & board.color_combined(Color::White);
            let white_rooks:   BitBoard = *board.pieces(Piece::Rook)   & board.color_combined(Color::White);
            let white_knights: BitBoard = *board.pieces(Piece::Knight) & board.color_combined(Color::White);
            let white_bishops: BitBoard = *board.pieces(Piece::Bishop) & board.color_combined(Color::White);
            let white_queens:  BitBoard = *board.pieces(Piece::Queen)  & board.color_combined(Color::White);
            let white_king:    BitBoard = *board.pieces(Piece::King)   & board.color_combined(Color::White);

            let black_pawns:   BitBoard = *board.pieces(Piece::Pawn)   & board.color_combined(Color::Black);
            let black_rooks:   BitBoard = *board.pieces(Piece::Rook)   & board.color_combined(Color::Black);
            let black_knights: BitBoard = *board.pieces(Piece::Knight) & board.color_combined(Color::Black);
            let black_bishops: BitBoard = *board.pieces(Piece::Bishop) & board.color_combined(Color::Black);
            let black_queens:  BitBoard = *board.pieces(Piece::Queen)  & board.color_combined(Color::Black);
            let black_king:    BitBoard = *board.pieces(Piece::King)   & board.color_combined(Color::Black);

            if white_bishops.popcnt() == 2 {
                total += BISHOP_PAIR_VAL;
            }

            if black_bishops.popcnt() == 2 {
                total -= BISHOP_PAIR_VAL;
            }

            total += pieces_type_eval(&white_pawns,   &PAWNS_VALUE_MAPPING_WHITE,   PAWN_BASE_VAL);
            total += pieces_type_eval(&white_rooks,   &ROOKS_VALUE_MAPPING_WHITE,   ROOK_BASE_VAL);
            total += pieces_type_eval(&white_knights, &KNIGHTS_VALUE_MAPPING_WHITE, KNIGHT_BASE_VAL);
            total += pieces_type_eval(&white_bishops, &BISHOPS_VALUE_MAPPING_WHITE, BISHOP_BASE_VAL);
            total += pieces_type_eval(&white_queens,  &QUEENS_VALUE_MAPPING_WHITE,  QUEEN_BASE_VAL);

            total -= pieces_type_eval(&black_pawns,   &PAWNS_VALUE_MAPPING_BLACK,   PAWN_BASE_VAL);
            total -= pieces_type_eval(&black_rooks,   &ROOKS_VALUE_MAPPING_BLACK,   ROOK_BASE_VAL);
            total -= pieces_type_eval(&black_knights, &KNIGHTS_VALUE_MAPPING_BLACK, KNIGHT_BASE_VAL);
            total -= pieces_type_eval(&black_bishops, &BISHOPS_VALUE_MAPPING_BLACK, BISHOP_BASE_VAL);
            total -= pieces_type_eval(&black_queens,  &QUEENS_VALUE_MAPPING_BLACK,  QUEEN_BASE_VAL);

            let prog = endgame_progress(board);

            for sq in white_king {
                // Mapping weightet on endgame_progress
                let idx = sq.to_index();
                let start_val = KING_START_VALUE_MAPPING_WHITE[idx];
                let end_val = KING_END_VALUE_MAPPING_WHITE[idx];
                let weighted = ((100 - prog) * start_val + prog * end_val + 50) / 100;
                total += weighted;

                if prog < 50 {
                    total += (king_safety_eval(board, sq, White) * (100 - prog)) / 100
                }
            }
            for sq in black_king {
                let idx = sq.to_index();
                let start_val = KING_START_VALUE_MAPPING_BLACK[idx];
                let end_val = KING_END_VALUE_MAPPING_BLACK[idx];
                let weighted = ((100 - prog) * start_val + prog * end_val + 50) / 100;
                total -= weighted;

                if prog < 50 {
                    total -= (king_safety_eval(board, sq, Black) * (100 - prog)) / 100
                }
            }


            total
        }
    }
}