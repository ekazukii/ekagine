use chess::{Board, ChessMove, Color, Piece, Square};
/// NNUE Implementation
/// We use a simple architecture (768->1024)x2->1 perspective net architecture
/// Network is initialized at compile time from the 'net.bin' file in thie bins directory.
/// The code is based on the code of Carp and Viridithas adpated to work with the chess crate
/// Most likely this can be copy-pasted for any other engine that use the chess crate
use std::{alloc, array};
use std::mem;
use std::ops::{Deref, DerefMut};

const MAX_DEPTH: usize = 128;

// Network Arch
const FEATURES: usize = 768;
const HIDDEN: usize = 1024;
const COLOR_STRIDE: usize = 64 * 6;
const PIECE_STRIDE: usize = 64;

// Clipped ReLu bounds
const CR_MIN: i16 = 0;
const CR_MAX: i16 = 255;

// Quantization factors
const QA: i32 = 255;
const QAB: i32 = 255 * 64;

// Eval scaling factor
const SCALE: i32 = 400;

type Eval = i32;

/// Container for all network parameters
#[repr(C)]
struct NNUEParams {
    feature_weights: Align64<[i16; FEATURES * HIDDEN]>,
    feature_bias: Align64<[i16; HIDDEN]>,
    output_weights: Align64<[i16; HIDDEN * 2]>,
    output_bias: i16,
}

/// NNUE model is initialized from direct binary values (Viridithas format)
// TODO: Adapt to work with zstd format for the weights
static MODEL: NNUEParams = unsafe { mem::transmute(*include_bytes!("../net.bin")) };

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C, align(64))]
struct Align64<T>(pub T);

impl<T, const SIZE: usize> Deref for Align64<[T; SIZE]> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const SIZE: usize> DerefMut for Align64<[T; SIZE]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

type SideAccumulator = Align64<[i16; HIDDEN]>;

/// Accumulators contain the efficiently updated hidden layer values
/// Each accumulator is perspective, hence both contains the white and black pov
#[derive(Clone, Copy, Debug)]
struct Accumulator {
    white: SideAccumulator,
    black: SideAccumulator,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            white: MODEL.feature_bias,
            black: MODEL.feature_bias,
        }
    }
}

impl Accumulator {
    /// Updates weights for a single feature, either turning them on or off
    fn update_weights<const ON: bool>(&mut self, idx: (usize, usize)) {
        fn update<const ON: bool>(acc: &mut SideAccumulator, idx: usize) {
            let zip = acc
                .iter_mut()
                .zip(&MODEL.feature_weights[idx..idx + HIDDEN]);

            for (acc_val, &weight) in zip {
                if ON {
                    *acc_val += weight;
                } else {
                    *acc_val -= weight;
                }
            }
        }

        update::<ON>(&mut self.white, idx.0);
        update::<ON>(&mut self.black, idx.1);
    }

    /// Update accumulator for a quiet move.
    /// Adds in features for the destination and removes the features of the source
    fn add_sub_weights(&mut self, from: (usize, usize), to: (usize, usize)) {
        fn add_sub(acc: &mut SideAccumulator, from: usize, to: usize) {
            let zip = acc.iter_mut().zip(
                MODEL.feature_weights[from..from + HIDDEN]
                    .iter()
                    .zip(&MODEL.feature_weights[to..to + HIDDEN]),
            );

            for (acc_val, (&remove_weight, &add_weight)) in zip {
                *acc_val += add_weight - remove_weight;
            }
        }

        add_sub(&mut self.white, from.0, to.0);
        add_sub(&mut self.black, from.1, to.1);
    }
}

/// NNUEState is simply a stack of accumulators, updated along the search tree
#[derive(Debug, Clone)]
pub struct NNUEState {
    accumulator_stack: [Accumulator; MAX_DEPTH + 1],
    current_acc: usize,
}

// used for turning on/off features
pub const ON: bool = true;
pub const OFF: bool = false;

impl NNUEState {
    /// Inits nnue state from a board
    /// To be able to run debug builds, heap is allocated manually
    pub fn from_board(board: &Board) -> Box<Self> {
        let mut boxed: Box<Self> = unsafe {
            let layout = alloc::Layout::new::<Self>();
            let ptr = alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        };

        // init with feature biases and add in all features of the board
        boxed.accumulator_stack[0] = Accumulator::default();
        let mut occupied = *board.combined();
        while let Some(sq) = occupied.next() {
            if let (Some(piece), Some(color)) = (board.piece_on(sq), board.color_on(sq)) {
                boxed.manual_update::<ON>(piece, color, sq);
            }
        }

        boxed
    }

    /// Refresh the accumulator stack to the given board
    pub fn refresh(&mut self, board: &Board) {
        // reset the accumulator stack
        self.current_acc = 0;
        self.accumulator_stack[self.current_acc] = Accumulator::default();

        // update the first accumulator
        let mut occupied = *board.combined();
        while let Some(sq) = occupied.next() {
            if let (Some(piece), Some(color)) = (board.piece_on(sq), board.color_on(sq)) {
                self.manual_update::<ON>(piece, color, sq);
            }
        }
    }

    /// Add a new accumulator to the stack by copying the previous top
    pub fn push(&mut self) {
        self.accumulator_stack[self.current_acc + 1] = self.accumulator_stack[self.current_acc];
        self.current_acc += 1;
    }

    /// Pop the top off the accumulator stack
    pub fn pop(&mut self) {
        self.current_acc -= 1;
    }

    /// Manually turn on or off the single given feature
    pub fn manual_update<const ON: bool>(&mut self, piece: Piece, color: Color, sq: Square) {
        self.accumulator_stack[self.current_acc].update_weights::<ON>(nnue_index(piece, color, sq));
    }

    /// Efficiently update accumulator for a quiet move (that is, only changes from/to features)
    pub fn move_update(&mut self, piece: Piece, color: Color, from: Square, to: Square) {
        let from_idx = nnue_index(piece, color, from);
        let to_idx = nnue_index(piece, color, to);

        self.accumulator_stack[self.current_acc].add_sub_weights(from_idx, to_idx);
    }

    /// Apply a move to the top accumulator, assuming the stack already reflects the pre-move board.
    pub fn apply_move(&mut self, board: &Board, mv: ChessMove) {
        let from = mv.get_source();
        let to = mv.get_dest();

        let piece = board
            .piece_on(from)
            .expect("expected piece on move source square");
        let color = board
            .color_on(from)
            .expect("expected color on move source square");

        let from_file = from.get_file().to_index() as i32;
        let to_file = to.get_file().to_index() as i32;
        let is_castling = piece == Piece::King && (from_file - to_file).abs() == 2;
        if is_castling {
            self.manual_update::<OFF>(Piece::King, color, from);
            self.manual_update::<ON>(Piece::King, color, to);

            let (rook_from, rook_to) = match (color, to) {
                (Color::White, Square::G1) => (Square::H1, Square::F1),
                (Color::White, Square::C1) => (Square::A1, Square::D1),
                (Color::Black, Square::G8) => (Square::H8, Square::F8),
                (Color::Black, Square::C8) => (Square::A8, Square::D8),
                _ => panic!("unexpected castling destination"),
            };

            self.manual_update::<OFF>(Piece::Rook, color, rook_from);
            self.manual_update::<ON>(Piece::Rook, color, rook_to);
            return;
        }

        let promotion = mv.get_promotion();
        let dest_piece = board.piece_on(to);
        let is_en_passant = piece == Piece::Pawn
            && dest_piece.is_none()
            && board.en_passant() == Some(to.ubackward(color));

        if promotion.is_none() && !is_en_passant && dest_piece.is_none() {
            self.move_update(piece, color, from, to);
            return;
        }

        if is_en_passant {
            let capture_sq = to.ubackward(color);
            let captured_color = match color {
                Color::White => Color::Black,
                Color::Black => Color::White,
            };
            self.manual_update::<OFF>(Piece::Pawn, captured_color, capture_sq);
        } else if let Some(captured_piece) = dest_piece {
            let captured_color = board
                .color_on(to)
                .expect("expected color on captured square");
            self.manual_update::<OFF>(captured_piece, captured_color, to);
        }

        self.manual_update::<OFF>(piece, color, from);
        let target_piece = promotion.unwrap_or(piece);
        self.manual_update::<ON>(target_piece, color, to);
    }

    /// Evaluate the nn from the current accumulator
    /// Concatenates the accumulators based on the side to move, computes the activation function
    /// with Squared CReLu and multiplies activation by weight. The result is the sum of all these
    /// with the bias.
    /// Since we are squaring activations, we need an extra quantization pass with QA.
    #[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
    pub fn evaluate(&self, side: Color) -> Eval {
        let acc = &self.accumulator_stack[self.current_acc];

        let (us, them) = match side {
            Color::White => (acc.white.iter(), acc.black.iter()),
            Color::Black => (acc.black.iter(), acc.white.iter()),
        };

        let mut out = 0;
        for (&value, &weight) in us.zip(&MODEL.output_weights[..HIDDEN]) {
            out += squared_crelu(value) * (weight as i32);
        }
        for (&value, &weight) in them.zip(&MODEL.output_weights[HIDDEN..]) {
            out += squared_crelu(value) * (weight as i32);
        }

        ((out / QA + MODEL.output_bias as i32) * SCALE / QAB) as Eval
    }

    #[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
    pub fn evaluate(&self, side: Color) -> Eval {
        use std::arch::aarch64::*;
        let acc = &self.accumulator_stack[self.current_acc];
        let (us, them) = match side {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };
        pub type VecI16 = int16x8_t;
        pub type VecI32 = int32x4_t;
        pub const VEC_I16_SIZE: usize = size_of::<VecI16>() / size_of::<i16>();

        unsafe {
            let cr_min = vdupq_n_s16(CR_MIN);
            let cr_max = vdupq_n_s16(CR_MAX);
            let mut sum_us: [VecI32; 8] = [unsafe { vdupq_n_s32(0) }; 8];
            let mut sum_them: [VecI32; 8] = [unsafe { vdupq_n_s32(0) }; 8];

            let acc_us_ptr = us.as_ptr();
            let acc_them_ptr = them.as_ptr();
            let weights1_ptr = MODEL.output_weights.as_ptr();
            let weights2_ptr = MODEL.output_weights.as_ptr().add(HIDDEN);

            for index in (0..HIDDEN).step_by(8 * VEC_I16_SIZE) {
                let x1: [VecI16; 8] = array::from_fn(|i| unsafe { vld1q_s16(acc_us_ptr.add((i * VEC_I16_SIZE) + index)) });
                let x2: [VecI16; 8] = array::from_fn(|i| unsafe { vld1q_s16(acc_them_ptr.add((i * VEC_I16_SIZE) + index)) });
                let w1: [VecI16; 8] = array::from_fn(|i| unsafe { vld1q_s16(weights1_ptr.add((i * VEC_I16_SIZE) + index)) });
                let w2: [VecI16; 8] = array::from_fn(|i| unsafe { vld1q_s16(weights2_ptr.add((i * VEC_I16_SIZE) + index)) });

                let v1: [VecI16; 8] = array::from_fn(|i| unsafe { vminq_s16(cr_max, vmaxq_s16(x1[i], cr_min)) });
                let v2: [VecI16; 8] = array::from_fn(|i| unsafe { vminq_s16(cr_max, vmaxq_s16(x2[i], cr_min)) });

                let vw1: [VecI16; 8] = array::from_fn(|i| unsafe { vmulq_s16(v1[i], w1[i]) });
                let vw2: [VecI16; 8] = array::from_fn(|i| unsafe { vmulq_s16(v2[i], w2[i]) });

                let sum_lo_us: [VecI32; 8] =
                    array::from_fn(|i| unsafe { vmlal_s16(sum_us[i], vget_low_s16(v1[i]), vget_low_s16(vw1[i])) });
                sum_us = array::from_fn(|i| unsafe { vmlal_high_s16(sum_lo_us[i], v1[i], vw1[i]) });

                let sum_lo_them: [VecI32; 8] =
                    array::from_fn(|i| unsafe { vmlal_s16(sum_them[i], vget_low_s16(v2[i]), vget_low_s16(vw2[i])) });
                sum_them = array::from_fn(|i| unsafe { vmlal_high_s16(sum_lo_them[i], v2[i], vw2[i]) });
            }

            let val: [VecI32; 8] = array::from_fn(|i| unsafe { vaddq_s32(sum_us[i], sum_them[i]) });

            let mut sum = val[0];
            for v in val.iter().take(8).skip(1) {
                sum = unsafe { vaddq_s32(sum, *v) };
            }
            let out = vaddvq_s32(sum);

            ((out / QA + MODEL.output_bias as i32) * SCALE / QAB) as Eval
        }
    }
}

const fn build_feature_index() -> [[[(usize, usize); 2]; 64]; 6] {
    let mut table = [[[(0usize, 0usize); 2]; 64]; 6];
    let mut p = 0;
    while p < 6 {
        let mut sq = 0;
        while sq < 64 {
            let file = sq % 8;
            let rank = sq / 8;
            let sq_idx = (7 - rank) * 8 + file;
            let flipped_idx = rank * 8 + file;

            let mut c = 0;
            while c < 2 {
                let white_idx = c * COLOR_STRIDE + p * PIECE_STRIDE + flipped_idx;
                let black_idx = (1 ^ c) * COLOR_STRIDE + p * PIECE_STRIDE + sq_idx;
                table[p][sq][c] = (white_idx * HIDDEN, black_idx * HIDDEN);
                c += 1;
            }

            sq += 1;
        }
        p += 1;
    }
    table
}

const FEATURE_INDEX: [[[(usize, usize); 2]; 64]; 6] = build_feature_index();

/// Returns white and black feature weight index for given feature
#[inline(always)]
fn nnue_index(piece: Piece, color: Color, sq: Square) -> (usize, usize) {
    let p = piece.to_index() as usize;
    let c = color.to_index() as usize;
    let sq_idx = sq.get_rank().to_index() * 8 + sq.get_file().to_index();

    FEATURE_INDEX[p][sq_idx][c]
}

/// Squared Clipped ReLu activation function
fn squared_crelu(value: i16) -> i32 {
    let v = value.clamp(CR_MIN, CR_MAX) as i32;

    v * v
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::ChessMove;
    use std::str::FromStr;

    fn assert_accumulators_equal(a: &NNUEState, a_idx: usize, b: &NNUEState, b_idx: usize) {
        for i in 0..HIDDEN {
            assert_eq!(
                a.accumulator_stack[a_idx].white[i],
                b.accumulator_stack[b_idx].white[i]
            );
            assert_eq!(
                a.accumulator_stack[a_idx].black[i],
                b.accumulator_stack[b_idx].black[i]
            );
        }
    }

    #[test]
    fn test_nnue_stack() {
        let b = Board::default();
        let s1 = NNUEState::from_board(&b);
        let mut s2 = NNUEState::from_board(&b);

        s2.push();
        s2.pop();

        for i in 0..HIDDEN {
            assert_eq!(
                s1.accumulator_stack[0].white[i],
                s2.accumulator_stack[0].white[i]
            );
            assert_eq!(
                s1.accumulator_stack[0].black[i],
                s2.accumulator_stack[0].black[i]
            );
        }
        assert_eq!(s1.current_acc, s2.current_acc);
    }

    #[test]
    fn test_nnue_index() {
        let idx1 = nnue_index(Piece::Pawn, Color::White, Square::A8);
        let idx2 = nnue_index(Piece::Pawn, Color::White, Square::H1);
        let idx3 = nnue_index(Piece::Pawn, Color::Black, Square::A1);
        let idx4 = nnue_index(Piece::King, Color::White, Square::E1);

        assert_eq!(idx1, (HIDDEN * 56, HIDDEN * 384));
        assert_eq!(idx2, (HIDDEN * 7, HIDDEN * 447));
        assert_eq!(idx3, (HIDDEN * 384, HIDDEN * 56));
        assert_eq!(idx4, (HIDDEN * 324, HIDDEN * 764));
    }

    #[test]
    fn test_manual_update() {
        let b: Board = Board::default();
        let mut s1 = NNUEState::from_board(&b);

        let old_acc = s1.accumulator_stack[0];

        s1.manual_update::<ON>(Piece::Pawn, Color::White, Square::A3);
        s1.manual_update::<OFF>(Piece::Pawn, Color::White, Square::A3);

        for i in 0..HIDDEN {
            assert_eq!(old_acc.white[i], s1.accumulator_stack[0].white[i]);
            assert_eq!(old_acc.black[i], s1.accumulator_stack[0].black[i]);
        }
    }

    #[test]
    fn test_incremental_updates() {
        let b1: Board = Board::default();
        let m = ChessMove::new(Square::E2, Square::E4, None);
        let b2: Board = b1.make_move_new(m);

        let mut s1 = NNUEState::from_board(&b1);
        let s2 = NNUEState::from_board(&b2);

        if let (Some(piece), Some(color)) =
            (b1.piece_on(m.get_source()), b1.color_on(m.get_source()))
        {
            s1.move_update(piece, color, m.get_source(), m.get_dest());
        } else {
            panic!("expected piece on source square");
        }

        assert_accumulators_equal(&s1, 0, &s2, 0);
    }

    #[test]
    fn test_apply_move_quiet_matches_rebuild() {
        let board = Board::default();
        let mv = ChessMove::new(Square::G1, Square::F3, None);

        let mut state = NNUEState::from_board(&board);
        let new_board = board.make_move_new(mv);
        let rebuilt = NNUEState::from_board(&new_board);

        state.push();
        state.apply_move(&board, mv);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_apply_move_capture_matches_rebuild() {
        let mut board = Board::default();
        let sequence = [
            ChessMove::new(Square::E2, Square::E4, None),
            ChessMove::new(Square::D7, Square::D5, None),
        ];
        for mv in sequence {
            board = board.make_move_new(mv);
        }
        let capture = ChessMove::new(Square::E4, Square::D5, None);

        let mut state = NNUEState::from_board(&board);
        let new_board = board.make_move_new(capture);
        let rebuilt = NNUEState::from_board(&new_board);

        state.push();
        state.apply_move(&board, capture);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_apply_move_en_passant_matches_rebuild() {
        let mut board = Board::default();
        let sequence = [
            ChessMove::new(Square::E2, Square::E4, None),
            ChessMove::new(Square::D7, Square::D5, None),
            ChessMove::new(Square::E4, Square::E5, None),
            ChessMove::new(Square::F7, Square::F5, None),
        ];
        for mv in sequence {
            board = board.make_move_new(mv);
        }

        assert_eq!(board.en_passant(), Some(Square::F5));

        let ep = ChessMove::new(Square::E5, Square::F6, None);
        let mut state = NNUEState::from_board(&board);
        let new_board = board.make_move_new(ep);
        let rebuilt = NNUEState::from_board(&new_board);

        state.push();
        state.apply_move(&board, ep);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_apply_move_castling_matches_rebuild() {
        let board = Board::from_str("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let mv = ChessMove::new(Square::E1, Square::G1, None);

        let mut state = NNUEState::from_board(&board);
        let new_board = board.make_move_new(mv);
        let rebuilt = NNUEState::from_board(&new_board);

        state.push();
        state.apply_move(&board, mv);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_apply_move_promotion_matches_rebuild() {
        let board = Board::from_str("7k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
        let mv = ChessMove::new(Square::A7, Square::A8, Some(Piece::Queen));

        let mut state = NNUEState::from_board(&board);
        let new_board = board.make_move_new(mv);
        let rebuilt = NNUEState::from_board(&new_board);

        state.push();
        state.apply_move(&board, mv);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_apply_move_promotion_capture_matches_rebuild() {
        let board = Board::from_str("1r5k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
        let mv = ChessMove::new(Square::A7, Square::B8, Some(Piece::Queen));

        let mut state = NNUEState::from_board(&board);
        let new_board = board.make_move_new(mv);
        let rebuilt = NNUEState::from_board(&new_board);

        state.push();
        state.apply_move(&board, mv);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_evaluate_specific_fen_snapshot() {
        let board = Board::from_str("1Q2q3/4P3/p1P1n3/5pb1/3p3p/KP5P/3k4/4R1N1 w - - 0 1").unwrap();
        let state = NNUEState::from_board(&board);

        let eval_white = state.evaluate(Color::White);
        let eval_black = state.evaluate(Color::Black);

        const EXPECTED_WHITE: Eval = 945;
        const EXPECTED_BLACK: Eval = -783;

        assert_eq!(
            eval_white, EXPECTED_WHITE,
            "snapshot mismatch for white POV (current {eval_white})"
        );
        assert_eq!(
            eval_black, EXPECTED_BLACK,
            "snapshot mismatch for black POV (current {eval_black})"
        );
    }
}
