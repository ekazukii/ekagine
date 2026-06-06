use crate::engine_core::{BitBoard, Board, ChessMove, Color, Piece, Square};
use std::mem;
use std::ops::{Deref, DerefMut};
/// NNUE Implementation
///
/// Frozen architecture: `(768 x KB_hm -> L1) x2 -> 1 x OB`
///  - 768 piece-square inputs, king-bucketed (`KB` buckets) with horizontal
///    mirroring: each perspective picks its bucket from its *own* king square,
///    and all of that perspective's features are file-mirrored (sq ^ 7) when
///    the king sits on files e-h.
///  - Single hidden layer of size `L1`, squared clipped ReLU activation.
///  - `OB` output buckets selected by total piece count: `(popcount - 2) / 4`.
///
/// Index semantics replicate bullet's `ChessBucketsMirrored` + `MaterialCount`
/// exactly, so a net trained with the matching trainer config is a drop-in
/// `net.bin`. The file layout (bullet `save_quantised`, padded to 64 bytes) is:
///   l0w [768*KB][L1] i16/QA (factoriser merged: effective clip is +/-2*0.99,
///   i.e. real weight bound ~+/-505, not +/-252), l0b [L1] i16/QA,
///   l1w [OB][2*L1] i16/QB (transposed at save), l1b [OB] i16/(QA*QB)
use std::{alloc, array};

const MAX_DEPTH: usize = 128;

// Network Arch — frozen structure, tunable dimensions.
const KB: usize = 8; // king input buckets (horizontally mirrored)
const L1: usize = 1536; // hidden layer size
const OB: usize = 8; // output buckets (by piece count)
const FEATURES: usize = 768 * KB;
const COLOR_STRIDE: usize = 64 * 6;
const PIECE_STRIDE: usize = 64;

// King-bucket layout for files a-d (files e-h are mirrored onto it),
// 4 entries per rank, rank 1 (own back rank) first.
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
];

/// Expand the 32-entry (files a-d) layout to all 64 squares by folding
/// files e-h back onto d-a. Mirrors bullet's `ChessBucketsMirrored::new`.
const fn expand_layout() -> [usize; 64] {
    let fold = [0usize, 1, 2, 3, 3, 2, 1, 0];
    let mut out = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        out[i] = BUCKET_LAYOUT[(i / 8) * 4 + fold[i % 8]];
        i += 1;
    }
    out
}

const EXPANDED_LAYOUT: [usize; 64] = expand_layout();

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
    feature_weights: Align64<[i16; FEATURES * L1]>,
    feature_bias: Align64<[i16; L1]>,
    output_weights: Align64<[i16; OB * 2 * L1]>,
    output_bias: [i16; OB],
}

// The struct tail is padded to its 64-byte alignment; bullet pads its output
// the same way, so `net.bin` must match `size_of::<NNUEParams>()` exactly.
const NNUE_PARAMS_BYTES: usize = {
    let raw = 2 * (FEATURES * L1 + L1 + OB * 2 * L1 + OB);
    raw.div_ceil(64) * 64
};
const _: () = assert!(mem::size_of::<NNUEParams>() == NNUE_PARAMS_BYTES);

/// NNUE model is initialized from direct binary values (bullet quantised format)
static RAW_MODEL: NNUEParams = unsafe { mem::transmute(*include_bytes!("../net.bin")) };

#[inline(always)]
fn model() -> &'static NNUEParams {
    &RAW_MODEL
}

/// Copy of the parameters in a 2MB-aligned anonymous allocation so transparent
/// hugepages can back the ~19MB weight table (file-backed `.rodata` is mapped
/// in 4K pages). Measured counter-productive on our hosts (KVM VM included),
/// kept disabled for reference.
#[cfg(any())]
#[inline(always)]
fn model_thp_copy() -> &'static NNUEParams {
    use std::sync::OnceLock;
    static CELL: OnceLock<&'static NNUEParams> = OnceLock::new();
    CELL.get_or_init(|| unsafe {
        let layout = alloc::Layout::from_size_align(mem::size_of::<NNUEParams>(), 2 * 1024 * 1024)
            .expect("nnue model layout");
        let ptr = alloc::alloc(layout) as *mut NNUEParams;
        if ptr.is_null() {
            return &RAW_MODEL;
        }
        std::ptr::copy_nonoverlapping(&RAW_MODEL as *const NNUEParams, ptr, 1);
        &*ptr
    })
}

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

type SideAccumulator = Align64<[i16; L1]>;

/// Accumulators contain the efficiently updated hidden layer values.
/// Each accumulator is perspective: both the white and black pov are kept.
#[derive(Clone, Copy, Debug)]
struct Accumulator {
    white: SideAccumulator,
    black: SideAccumulator,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            white: model().feature_bias,
            black: model().feature_bias,
        }
    }
}

/// Finny-table entry: a cached accumulator half for one (perspective, king
/// bucket, mirror) context together with the piece bitboards of the board it
/// was computed for. A bucket refresh becomes a diff against this cache
/// instead of a from-scratch rebuild.
#[derive(Clone, Copy, Debug)]
struct FinnyEntry {
    acc: SideAccumulator,
    pieces: [u64; 6],
    colors: [u64; 2],
    valid: bool,
}

/// `[perspective][bucket * 2 + mirror]`
#[derive(Clone, Copy, Debug)]
struct FinnyTable {
    entries: [[FinnyEntry; KB * 2]; 2],
}

/// Per-perspective feature-indexing context, derived from that perspective's
/// own king square: which input bucket to use and whether to mirror files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PerspCtx {
    /// `bucket * 768`, the row offset of this king bucket's feature block.
    bucket_base: usize,
    /// 0 (no mirror) or 7 (king on files e-h: XOR square with 7).
    flip: usize,
}

/// Map a square to the given perspective's coordinates (vertical flip for black).
#[inline(always)]
fn persp_square(persp: Color, sq: Square) -> usize {
    match persp {
        Color::White => sq.to_index(),
        Color::Black => sq.to_index() ^ 56,
    }
}

/// Context from a perspective-relative king square. Mirrors bullet:
/// `flip = 7 if ksq file > d`, `bucket = EXPANDED_LAYOUT[ksq]`.
#[inline(always)]
fn persp_ctx(king_sq_persp: usize) -> PerspCtx {
    PerspCtx {
        bucket_base: EXPANDED_LAYOUT[king_sq_persp] * 768,
        flip: if (king_sq_persp & 7) > 3 { 7 } else { 0 },
    }
}

/// The perspective's own king square on `board`.
#[inline(always)]
fn own_king_sq(board: &Board, color: Color) -> Square {
    let bb: BitBoard = *board.pieces(Piece::King) & board.color_combined(color);
    bb.to_square()
}

/// Weight-row offset for one feature in one perspective's half.
/// `(bucket*768 + rel_color*384 + piece*64 + (sq ^ flip)) * L1`
#[inline(always)]
fn half_index(persp: Color, ctx: PerspCtx, piece: Piece, color: Color, sq: Square) -> usize {
    let rel_color = if color == persp { 0 } else { 1 };
    let sq_p = persp_square(persp, sq) ^ ctx.flip;
    (ctx.bucket_base + rel_color * COLOR_STRIDE + piece.to_index() * PIECE_STRIDE + sq_p) * L1
}

/// Prefetch the head of a 3KB weight row (the hardware prefetcher streams the
/// rest once the access pattern is established).
#[inline(always)]
#[allow(unused_variables)]
fn prefetch_row(ptr: *const i16) {
    // aarch64 (Apple Silicon): measured regression — the hardware prefetcher
    // already hides the row latency there; software prefetch only adds
    // instruction overhead. x86 hosts (small L3, costly TLB) are the target.
    // x86: mesuré contre-productif sur notre VM KVM (pollution L1 + overhead),
    // le hardware prefetcher fait mieux seul. Helper conservé, no-op partout.
    let _ = ptr;
}

// ─── Single-half fused accumulator updates ──────────────────────────────────
// All indices are weight-row offsets (already multiplied by L1) and in bounds
// by construction, so bounds checks are elided via `get_unchecked`.

/// `dst[i] += w[add + i]` (used when rebuilding from scratch).
#[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn half_add(dst: &mut [i16; L1], add: usize) {
    use std::arch::aarch64::*;
    let w: &[i16] = &model().feature_weights.0;
    unsafe {
        let wp = w.as_ptr().add(add);
        prefetch_row(wp);
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i < L1 {
            vst1q_s16(
                dp.add(i),
                vaddq_s16(vld1q_s16(dp.add(i) as *const i16), vld1q_s16(wp.add(i))),
            );
            vst1q_s16(
                dp.add(i + 8),
                vaddq_s16(
                    vld1q_s16(dp.add(i + 8) as *const i16),
                    vld1q_s16(wp.add(i + 8)),
                ),
            );
            vst1q_s16(
                dp.add(i + 16),
                vaddq_s16(
                    vld1q_s16(dp.add(i + 16) as *const i16),
                    vld1q_s16(wp.add(i + 16)),
                ),
            );
            vst1q_s16(
                dp.add(i + 24),
                vaddq_s16(
                    vld1q_s16(dp.add(i + 24) as *const i16),
                    vld1q_s16(wp.add(i + 24)),
                ),
            );
            i += 32;
        }
    }
}

/// Fused quiet move: `dst[i] = prev[i] + w[add+i] - w[sub+i]`.
#[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn half_add_sub_from(dst: &mut [i16; L1], prev: &[i16; L1], add: usize, sub: usize) {
    use std::arch::aarch64::*;
    let w: &[i16] = &model().feature_weights.0;
    unsafe {
        let (wa, ws) = (w.as_ptr().add(add), w.as_ptr().add(sub));
        prefetch_row(wa);
        prefetch_row(ws);
        let pp = prev.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i < L1 {
            let mut k = 0;
            while k < 32 {
                let o = i + k;
                let v = vaddq_s16(
                    vld1q_s16(pp.add(o)),
                    vsubq_s16(vld1q_s16(wa.add(o)), vld1q_s16(ws.add(o))),
                );
                vst1q_s16(dp.add(o), v);
                k += 8;
            }
            i += 32;
        }
    }
}

/// Fused capture: `dst[i] = prev[i] + w[add+i] - w[sub0+i] - w[sub1+i]`.
#[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn half_add_sub_sub_from(
    dst: &mut [i16; L1],
    prev: &[i16; L1],
    add: usize,
    sub0: usize,
    sub1: usize,
) {
    use std::arch::aarch64::*;
    let w: &[i16] = &model().feature_weights.0;
    unsafe {
        let (wa, ws0, ws1) = (
            w.as_ptr().add(add),
            w.as_ptr().add(sub0),
            w.as_ptr().add(sub1),
        );
        prefetch_row(wa);
        prefetch_row(ws0);
        prefetch_row(ws1);
        let pp = prev.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i < L1 {
            let mut k = 0;
            while k < 32 {
                let o = i + k;
                let v = vsubq_s16(
                    vaddq_s16(
                        vld1q_s16(pp.add(o)),
                        vsubq_s16(vld1q_s16(wa.add(o)), vld1q_s16(ws0.add(o))),
                    ),
                    vld1q_s16(ws1.add(o)),
                );
                vst1q_s16(dp.add(o), v);
                k += 8;
            }
            i += 32;
        }
    }
}

/// Fused castling: `dst[i] = prev[i] + w[add0] + w[add1] - w[sub0] - w[sub1]`.
#[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn half_add_add_sub_sub_from(
    dst: &mut [i16; L1],
    prev: &[i16; L1],
    add0: usize,
    add1: usize,
    sub0: usize,
    sub1: usize,
) {
    use std::arch::aarch64::*;
    let w: &[i16] = &model().feature_weights.0;
    unsafe {
        let (wa0, wa1) = (w.as_ptr().add(add0), w.as_ptr().add(add1));
        let (ws0, ws1) = (w.as_ptr().add(sub0), w.as_ptr().add(sub1));
        prefetch_row(wa0);
        prefetch_row(wa1);
        prefetch_row(ws0);
        prefetch_row(ws1);
        let pp = prev.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i < L1 {
            let mut k = 0;
            while k < 32 {
                let o = i + k;
                let v = vaddq_s16(
                    vaddq_s16(
                        vld1q_s16(pp.add(o)),
                        vsubq_s16(vld1q_s16(wa0.add(o)), vld1q_s16(ws0.add(o))),
                    ),
                    vsubq_s16(vld1q_s16(wa1.add(o)), vld1q_s16(ws1.add(o))),
                );
                vst1q_s16(dp.add(o), v);
                k += 8;
            }
            i += 32;
        }
    }
}

/// `dst[i] += w[add + i]` (used when rebuilding from scratch).
#[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
fn half_add(dst: &mut [i16; L1], add: usize) {
    let w: &[i16] = &model().feature_weights.0;
    for i in 0..L1 {
        unsafe {
            *dst.get_unchecked_mut(i) += *w.get_unchecked(add + i);
        }
    }
}

/// Fused quiet move: `dst[i] = prev[i] + w[add+i] - w[sub+i]`.
#[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
fn half_add_sub_from(dst: &mut [i16; L1], prev: &[i16; L1], add: usize, sub: usize) {
    let w: &[i16] = &model().feature_weights.0;
    for i in 0..L1 {
        unsafe {
            *dst.get_unchecked_mut(i) =
                *prev.get_unchecked(i) + *w.get_unchecked(add + i) - *w.get_unchecked(sub + i);
        }
    }
}

/// Fused capture: `dst[i] = prev[i] + w[add+i] - w[sub0+i] - w[sub1+i]`.
#[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
fn half_add_sub_sub_from(
    dst: &mut [i16; L1],
    prev: &[i16; L1],
    add: usize,
    sub0: usize,
    sub1: usize,
) {
    let w: &[i16] = &model().feature_weights.0;
    for i in 0..L1 {
        unsafe {
            *dst.get_unchecked_mut(i) = *prev.get_unchecked(i) + *w.get_unchecked(add + i)
                - *w.get_unchecked(sub0 + i)
                - *w.get_unchecked(sub1 + i);
        }
    }
}

/// Fused castling: `dst[i] = prev[i] + w[add0] + w[add1] - w[sub0] - w[sub1]`.
#[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
fn half_add_add_sub_sub_from(
    dst: &mut [i16; L1],
    prev: &[i16; L1],
    add0: usize,
    add1: usize,
    sub0: usize,
    sub1: usize,
) {
    let w: &[i16] = &model().feature_weights.0;
    for i in 0..L1 {
        unsafe {
            *dst.get_unchecked_mut(i) =
                *prev.get_unchecked(i) + *w.get_unchecked(add0 + i) + *w.get_unchecked(add1 + i)
                    - *w.get_unchecked(sub0 + i)
                    - *w.get_unchecked(sub1 + i);
        }
    }
}

/// `dst[i] -= w[sub + i]` (finny cache diff).
#[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn half_sub(dst: &mut [i16; L1], sub: usize) {
    use std::arch::aarch64::*;
    let w: &[i16] = &model().feature_weights.0;
    unsafe {
        let wp = w.as_ptr().add(sub);
        prefetch_row(wp);
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i < L1 {
            let mut k = 0;
            while k < 32 {
                let o = i + k;
                vst1q_s16(
                    dp.add(o),
                    vsubq_s16(vld1q_s16(dp.add(o) as *const i16), vld1q_s16(wp.add(o))),
                );
                k += 8;
            }
            i += 32;
        }
    }
}

/// `dst[i] -= w[sub + i]` (finny cache diff).
#[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
fn half_sub(dst: &mut [i16; L1], sub: usize) {
    let w: &[i16] = &model().feature_weights.0;
    for i in 0..L1 {
        unsafe {
            *dst.get_unchecked_mut(i) -= *w.get_unchecked(sub + i);
        }
    }
}

/// Rebuild one perspective's half from scratch for `board` (bias + features).
/// Needed whenever that perspective's king changes bucket or mirror state.
fn rebuild_half(dst: &mut [i16; L1], board: &Board, persp: Color) {
    let ctx = persp_ctx(persp_square(persp, own_king_sq(board, persp)));
    dst.copy_from_slice(&model().feature_bias.0);
    let occupied = *board.combined();
    for sq in occupied {
        if let (Some(piece), Some(color)) = (board.piece_on(sq), board.color_on(sq)) {
            half_add(dst, half_index(persp, ctx, piece, color, sq));
        }
    }
}

/// Rebuild one perspective's half through the finny cache: diff the cached
/// board of this (perspective, bucket, mirror) context against `board`,
/// apply only the changed features, then copy the cached accumulator out.
fn rebuild_half_finny(
    entry: &mut FinnyEntry,
    dst: &mut [i16; L1],
    board: &Board,
    persp: Color,
    ctx: PerspCtx,
) {
    if !entry.valid {
        entry.acc.0.copy_from_slice(&model().feature_bias.0);
        entry.pieces = [0; 6];
        entry.colors = [0; 2];
        entry.valid = true;
    }

    for (ci, color) in [Color::White, Color::Black].into_iter().enumerate() {
        let cur_color = board.color_combined(color).0;
        for pi in 0..6 {
            let piece = PIECES_BY_INDEX[pi];
            let cur: u64 = board.pieces(piece).0 & cur_color;
            let old: u64 = entry.pieces[pi] & entry.colors[ci];
            let mut added = cur & !old;
            let mut removed = old & !cur;
            while added != 0 {
                let sq = Square::new(added.trailing_zeros() as u8);
                half_add(&mut entry.acc.0, half_index(persp, ctx, piece, color, sq));
                added &= added - 1;
            }
            while removed != 0 {
                let sq = Square::new(removed.trailing_zeros() as u8);
                half_sub(&mut entry.acc.0, half_index(persp, ctx, piece, color, sq));
                removed &= removed - 1;
            }
        }
    }
    // Cache the new board occupancy per piece type and color.
    for pi in 0..6 {
        entry.pieces[pi] = board.pieces(PIECES_BY_INDEX[pi]).0;
    }
    entry.colors[0] = board.color_combined(Color::White).0;
    entry.colors[1] = board.color_combined(Color::Black).0;

    dst.copy_from_slice(&entry.acc.0);
}

const PIECES_BY_INDEX: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Incremental update of one perspective's half for `mv` on `board` (pre-move),
/// assuming this perspective's context is unchanged by the move.
fn apply_move_half(
    dst: &mut [i16; L1],
    prev: &[i16; L1],
    persp: Color,
    ctx: PerspCtx,
    board: &Board,
    mv: ChessMove,
    piece: Piece,
    color: Color,
) {
    let from = mv.get_source();
    let to = mv.get_dest();
    let idx = |p: Piece, c: Color, s: Square| half_index(persp, ctx, p, c, s);

    let from_file = from.get_file().to_index() as i32;
    let to_file = to.get_file().to_index() as i32;
    let is_castling = piece == Piece::King && (from_file - to_file).abs() == 2;
    if is_castling {
        let (rook_from, rook_to) = match (color, to) {
            (Color::White, Square::G1) => (Square::H1, Square::F1),
            (Color::White, Square::C1) => (Square::A1, Square::D1),
            (Color::Black, Square::G8) => (Square::H8, Square::F8),
            (Color::Black, Square::C8) => (Square::A8, Square::D8),
            _ => panic!("unexpected castling destination"),
        };
        half_add_add_sub_sub_from(
            dst,
            prev,
            idx(Piece::King, color, to),
            idx(Piece::Rook, color, rook_to),
            idx(Piece::King, color, from),
            idx(Piece::Rook, color, rook_from),
        );
        return;
    }

    let promotion = mv.get_promotion();
    let dest_piece = board.piece_on(to);
    let is_en_passant =
        piece == Piece::Pawn && dest_piece.is_none() && board.en_passant() == Some(to);

    let target_piece = promotion.unwrap_or(piece);
    let add = idx(target_piece, color, to);
    let sub = idx(piece, color, from);

    let captured = if is_en_passant {
        let capture_sq = to.ubackward(color);
        let captured_color = match color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        Some(idx(Piece::Pawn, captured_color, capture_sq))
    } else if let Some(captured_piece) = dest_piece {
        let captured_color = board
            .color_on(to)
            .expect("expected color on captured square");
        Some(idx(captured_piece, captured_color, to))
    } else {
        None
    };

    match captured {
        Some(cap) => half_add_sub_sub_from(dst, prev, add, sub, cap),
        None => half_add_sub_from(dst, prev, add, sub),
    }
}

/// NNUEState is simply a stack of accumulators, updated along the search tree
#[derive(Debug, Clone)]
pub struct NNUEState {
    accumulator_stack: [Accumulator; MAX_DEPTH + 1],
    current_acc: usize,
    finny: FinnyTable,
}

impl NNUEState {
    /// Inits nnue state from a board.
    /// To be able to run debug builds, heap is allocated manually.
    pub fn from_board(board: &Board) -> Box<Self> {
        let mut boxed: Box<Self> = unsafe {
            let layout = alloc::Layout::new::<Self>();
            let ptr = alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        };

        boxed.current_acc = 0;
        let acc = &mut boxed.accumulator_stack[0];
        rebuild_half(&mut acc.white.0, board, Color::White);
        rebuild_half(&mut acc.black.0, board, Color::Black);

        boxed
    }

    /// Refresh the accumulator stack to the given board
    pub fn refresh(&mut self, board: &Board) {
        self.current_acc = 0;
        let acc = &mut self.accumulator_stack[0];
        rebuild_half(&mut acc.white.0, board, Color::White);
        rebuild_half(&mut acc.black.0, board, Color::Black);
    }

    /// Open a new accumulator slot on the stack. The new top is left
    /// *uninitialized*; `apply_move` must be called next to populate it from the
    /// previous slot. Every `push()` in the search is immediately followed by
    /// `apply_move`; the null-move case does not push (the accumulator is
    /// unchanged, so the child reuses the parent's slot).
    pub fn push(&mut self) {
        self.current_acc += 1;
    }

    /// Pop the top off the accumulator stack
    pub fn pop(&mut self) {
        self.current_acc -= 1;
    }

    /// Apply a move, producing the new top accumulator from the previous slot.
    /// `board` is the position *before* the move. For each perspective: if its
    /// own king changes bucket or mirror state, that half is rebuilt from the
    /// post-move board; otherwise it is updated with a single fused pass.
    pub fn apply_move(&mut self, board: &Board, mv: ChessMove) {
        let from = mv.get_source();
        let to = mv.get_dest();

        let piece = board
            .piece_on(from)
            .expect("expected piece on move source square");
        let color = board
            .color_on(from)
            .expect("expected color on move source square");

        // Does the mover's own-perspective context change?
        let needs_rebuild = piece == Piece::King
            && persp_ctx(persp_square(color, from)) != persp_ctx(persp_square(color, to));
        // The post-move board is only materialized when a rebuild is required
        // (king bucket/mirror crossings are rare).
        let board_after = if needs_rebuild {
            Some(board.make_move_new(mv))
        } else {
            None
        };

        let cur = self.current_acc;
        let (lower, upper) = self.accumulator_stack.split_at_mut(cur);
        let prev = &lower[cur - 1];
        let dst = &mut upper[0];

        for persp in [Color::White, Color::Black] {
            let (dst_half, prev_half) = match persp {
                Color::White => (&mut dst.white.0, &prev.white.0),
                Color::Black => (&mut dst.black.0, &prev.black.0),
            };

            if needs_rebuild && persp == color {
                let after = board_after.as_ref().unwrap();
                let ksq_p = persp_square(persp, own_king_sq(after, persp));
                let ctx = persp_ctx(ksq_p);
                let entry = &mut self.finny.entries[persp.to_index()]
                    [(ctx.bucket_base / 768) * 2 + usize::from(ctx.flip == 7)];
                rebuild_half_finny(entry, dst_half, after, persp, ctx);
            } else {
                // This perspective's king did not change context: its king
                // either belongs to the opponent or stayed within its bucket.
                let ctx = persp_ctx(persp_square(persp, own_king_sq(board, persp)));
                apply_move_half(dst_half, prev_half, persp, ctx, board, mv, piece, color);
            }
        }
    }

    /// Output bucket by piece count, mirroring bullet's `MaterialCount<OB>`.
    #[inline(always)]
    fn output_bucket(board: &Board) -> usize {
        let divisor = 32usize.div_ceil(OB);
        (((board.combined().popcnt().max(2) - 2) as usize) / divisor).min(OB - 1)
    }

    /// Evaluate the nn from the current accumulator.
    /// Concatenates the accumulators based on the side to move, computes the
    /// Squared CReLU activation and the bucketed output layer.
    /// Since we are squaring activations, we need an extra quantization pass with QA.
    #[cfg(not(any(target_arch = "aarch64", target_feature = "neon")))]
    pub fn evaluate(&self, board: &Board) -> Eval {
        let acc = &self.accumulator_stack[self.current_acc];
        let bucket = Self::output_bucket(board);

        let (us, them) = match board.side_to_move() {
            Color::White => (acc.white.iter(), acc.black.iter()),
            Color::Black => (acc.black.iter(), acc.white.iter()),
        };

        let ow = &model().output_weights[bucket * 2 * L1..(bucket + 1) * 2 * L1];

        // i64 accumulation: the theoretical worst case of the i32 sum can
        // overflow (audit finding); i64 removes it at no practical cost here.
        let mut out: i64 = 0;
        for (&value, &weight) in us.zip(&ow[..L1]) {
            out += (squared_crelu(value) * (weight as i32)) as i64;
        }
        for (&value, &weight) in them.zip(&ow[L1..]) {
            out += (squared_crelu(value) * (weight as i32)) as i64;
        }

        ((out / QA as i64 + model().output_bias[bucket] as i64) * SCALE as i64 / QAB as i64) as Eval
    }

    #[cfg(any(target_arch = "aarch64", target_feature = "neon"))]
    pub fn evaluate(&self, board: &Board) -> Eval {
        use std::arch::aarch64::*;
        let acc = &self.accumulator_stack[self.current_acc];
        let bucket = Self::output_bucket(board);
        let (us, them) = match board.side_to_move() {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };
        pub type VecI16 = int16x8_t;
        pub type VecI32 = int32x4_t;
        pub const VEC_I16_SIZE: usize = size_of::<VecI16>() / size_of::<i16>();

        unsafe {
            let cr_min = vdupq_n_s16(CR_MIN);
            let cr_max = vdupq_n_s16(CR_MAX);
            let mut sum_us: [VecI32; 8] = [vdupq_n_s32(0); 8];
            let mut sum_them: [VecI32; 8] = [vdupq_n_s32(0); 8];

            let acc_us_ptr = us.as_ptr();
            let acc_them_ptr = them.as_ptr();
            let weights1_ptr = model().output_weights.as_ptr().add(bucket * 2 * L1);
            let weights2_ptr = model().output_weights.as_ptr().add(bucket * 2 * L1 + L1);

            for index in (0..L1).step_by(8 * VEC_I16_SIZE) {
                let x1: [VecI16; 8] =
                    array::from_fn(|i| vld1q_s16(acc_us_ptr.add((i * VEC_I16_SIZE) + index)));
                let x2: [VecI16; 8] =
                    array::from_fn(|i| vld1q_s16(acc_them_ptr.add((i * VEC_I16_SIZE) + index)));
                let w1: [VecI16; 8] =
                    array::from_fn(|i| vld1q_s16(weights1_ptr.add((i * VEC_I16_SIZE) + index)));
                let w2: [VecI16; 8] =
                    array::from_fn(|i| vld1q_s16(weights2_ptr.add((i * VEC_I16_SIZE) + index)));

                let v1: [VecI16; 8] =
                    array::from_fn(|i| vminq_s16(cr_max, vmaxq_s16(x1[i], cr_min)));
                let v2: [VecI16; 8] =
                    array::from_fn(|i| vminq_s16(cr_max, vmaxq_s16(x2[i], cr_min)));

                let vw1: [VecI16; 8] = array::from_fn(|i| vmulq_s16(v1[i], w1[i]));
                let vw2: [VecI16; 8] = array::from_fn(|i| vmulq_s16(v2[i], w2[i]));

                let sum_lo_us: [VecI32; 8] = array::from_fn(|i| {
                    vmlal_s16(sum_us[i], vget_low_s16(v1[i]), vget_low_s16(vw1[i]))
                });
                sum_us = array::from_fn(|i| vmlal_high_s16(sum_lo_us[i], v1[i], vw1[i]));

                let sum_lo_them: [VecI32; 8] = array::from_fn(|i| {
                    vmlal_s16(sum_them[i], vget_low_s16(v2[i]), vget_low_s16(vw2[i]))
                });
                sum_them = array::from_fn(|i| vmlal_high_s16(sum_lo_them[i], v2[i], vw2[i]));
            }

            // Per-lane partial sums hold <= 48 terms each (safe in i32); the
            // final horizontal reduction is widened to i64 to remove the
            // theoretical overflow of summing 64 partials (audit finding).
            let mut out: i64 = 0;
            for i in 0..8 {
                out += vaddlvq_s32(sum_us[i]);
                out += vaddlvq_s32(sum_them[i]);
            }

            ((out / QA as i64 + model().output_bias[bucket] as i64) * SCALE as i64 / QAB as i64)
                as Eval
        }
    }
}

/// Squared Clipped ReLu activation function
fn squared_crelu(value: i16) -> i32 {
    let v = value.clamp(CR_MIN, CR_MAX) as i32;

    v * v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_core::ChessMove;
    use std::str::FromStr;

    fn assert_accumulators_equal(a: &NNUEState, a_idx: usize, b: &NNUEState, b_idx: usize) {
        for i in 0..L1 {
            assert_eq!(
                a.accumulator_stack[a_idx].white[i], b.accumulator_stack[b_idx].white[i],
                "white mismatch at {i}"
            );
            assert_eq!(
                a.accumulator_stack[a_idx].black[i], b.accumulator_stack[b_idx].black[i],
                "black mismatch at {i}"
            );
        }
    }

    /// apply_move(board, mv) must agree with a from-scratch rebuild of the
    /// post-move board, for every move type and king-context change.
    fn assert_apply_matches_rebuild(fen: &str, mv: ChessMove) {
        let board = Board::from_str(fen).unwrap();
        let mut state = NNUEState::from_board(&board);
        let rebuilt = NNUEState::from_board(&board.make_move_new(mv));

        state.push();
        state.apply_move(&board, mv);

        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_nnue_stack() {
        let b = Board::default();
        let s1 = NNUEState::from_board(&b);
        let mut s2 = NNUEState::from_board(&b);

        s2.push();
        s2.pop();

        assert_accumulators_equal(&s1, 0, &s2, 0);
        assert_eq!(s1.current_acc, s2.current_acc);
    }

    #[test]
    fn test_layout_expansion_matches_bullet() {
        // bullet: expanded[idx] = layout[(idx / 8) * 4 + [0,1,2,3,3,2,1,0][idx % 8]]
        assert_eq!(EXPANDED_LAYOUT[0], 0); // a1
        assert_eq!(EXPANDED_LAYOUT[3], 3); // d1
        assert_eq!(EXPANDED_LAYOUT[4], 3); // e1 folds onto d1
        assert_eq!(EXPANDED_LAYOUT[7], 0); // h1 folds onto a1
        assert_eq!(EXPANDED_LAYOUT[8], 4); // a2
        assert_eq!(EXPANDED_LAYOUT[15], 4); // h2 folds onto a2
        assert_eq!(EXPANDED_LAYOUT[63], 7); // h8
    }

    #[test]
    fn test_half_index_mirroring() {
        // White king on e1 (file e => mirrored): white pawn a2 maps like h2.
        let ctx_e1 = persp_ctx(Square::E1.to_index());
        assert_eq!(ctx_e1.flip, 7);
        let a2 = half_index(Color::White, ctx_e1, Piece::Pawn, Color::White, Square::A2);
        // mirrored square of a2 (idx 8) is h2 (idx 15)
        assert_eq!(a2, (ctx_e1.bucket_base + 15) * L1);

        // White king on d1 (no mirror): white pawn a2 maps to a2.
        let ctx_d1 = persp_ctx(Square::D1.to_index());
        assert_eq!(ctx_d1.flip, 0);
        let a2 = half_index(Color::White, ctx_d1, Piece::Pawn, Color::White, Square::A2);
        assert_eq!(a2, (ctx_d1.bucket_base + 8) * L1);
    }

    #[test]
    fn test_output_bucket() {
        let start = Board::default();
        assert_eq!(NNUEState::output_bucket(&start), 7); // 32 pieces -> (32-2)/4 = 7
        let kk = Board::from_str("8/8/8/3k4/8/3K4/4P3/8 w - - 0 1").unwrap();
        assert_eq!(NNUEState::output_bucket(&kk), 0); // 3 pieces -> 0
    }

    #[test]
    fn test_apply_move_quiet_matches_rebuild() {
        assert_apply_matches_rebuild(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ChessMove::new(Square::G1, Square::F3, None),
        );
    }

    #[test]
    fn test_apply_move_capture_matches_rebuild() {
        assert_apply_matches_rebuild(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            ChessMove::new(Square::E4, Square::D5, None),
        );
    }

    #[test]
    fn test_apply_move_en_passant_matches_rebuild() {
        let mut board = Board::default();
        for mv in [
            ChessMove::new(Square::E2, Square::E4, None),
            ChessMove::new(Square::D7, Square::D5, None),
            ChessMove::new(Square::E4, Square::E5, None),
            ChessMove::new(Square::F7, Square::F5, None),
        ] {
            board = board.make_move_new(mv);
        }
        assert_eq!(board.en_passant(), Some(Square::F6));

        let ep = ChessMove::new(Square::E5, Square::F6, None);
        let mut state = NNUEState::from_board(&board);
        let rebuilt = NNUEState::from_board(&board.make_move_new(ep));
        state.push();
        state.apply_move(&board, ep);
        assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
    }

    #[test]
    fn test_apply_move_castling_matches_rebuild() {
        // Castling moves the king across buckets => rebuild path.
        assert_apply_matches_rebuild(
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            ChessMove::new(Square::E1, Square::G1, None),
        );
        assert_apply_matches_rebuild(
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            ChessMove::new(Square::E8, Square::C8, None),
        );
    }

    #[test]
    fn test_apply_move_promotion_matches_rebuild() {
        assert_apply_matches_rebuild(
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
            ChessMove::new(Square::A7, Square::A8, Some(Piece::Queen)),
        );
        assert_apply_matches_rebuild(
            "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
            ChessMove::new(Square::A7, Square::B8, Some(Piece::Queen)),
        );
    }

    #[test]
    fn test_king_move_same_context_incremental() {
        // g2 -> h2: both fold to bucket 4 with mirror on => incremental path.
        assert_apply_matches_rebuild(
            "7k/8/8/8/8/8/6K1/8 w - - 0 1",
            ChessMove::new(Square::G2, Square::H2, None),
        );
    }

    #[test]
    fn test_king_move_mirror_crossing_rebuild() {
        // d4 -> e4 crosses the mirror line (and stays in bucket 6/7 region).
        assert_apply_matches_rebuild(
            "7k/8/8/8/3K4/8/8/8 w - - 0 1",
            ChessMove::new(Square::D4, Square::E4, None),
        );
    }

    #[test]
    fn test_king_move_bucket_crossing_rebuild() {
        // a1 -> a2 crosses from bucket 0 to bucket 4.
        assert_apply_matches_rebuild(
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            ChessMove::new(Square::A1, Square::A2, None),
        );
    }

    #[test]
    fn test_black_king_context_uses_own_perspective() {
        // Black king e8: from black's perspective e8 is file e => mirrored.
        // Move ke8->d8 crosses the mirror line for black only.
        assert_apply_matches_rebuild(
            "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
            ChessMove::new(Square::E8, Square::D8, None),
        );
    }

    #[test]
    fn test_finny_cache_reuse_across_repeated_crossings() {
        // King shuttles across the bucket/mirror boundary several times with
        // board changes in between: every rebuild after the first hits the
        // finny cache (diff path), which must stay exact vs a fresh rebuild.
        let mut board = Board::from_str("r3k3/p6p/8/8/8/8/P6P/4K2R w - - 0 1").unwrap();
        let mut state = NNUEState::from_board(&board);

        let moves = [
            ChessMove::new(Square::E1, Square::D1, None), // mirror cross
            ChessMove::new(Square::E8, Square::D8, None), // black mirror cross
            ChessMove::new(Square::D1, Square::E1, None), // cross back (cache hit)
            ChessMove::new(Square::A7, Square::A5, None), // board changes
            ChessMove::new(Square::E1, Square::D1, None), // cross again (diff)
            ChessMove::new(Square::H7, Square::H5, None),
            ChessMove::new(Square::H1, Square::H5, None), // capture
            ChessMove::new(Square::D8, Square::E8, None), // black cross back
        ];
        for mv in moves {
            state.push();
            state.apply_move(&board, mv);
            board = board.make_move_new(mv);
            let rebuilt = NNUEState::from_board(&board);
            assert_accumulators_equal(&state, state.current_acc, &rebuilt, 0);
        }
    }

    #[test]
    fn test_evaluate_sane_and_side_consistent() {
        let board = Board::default();
        let state = NNUEState::from_board(&board);
        let eval = state.evaluate(&board);
        assert!(eval.abs() < 20_000, "startpos eval out of range: {eval}");

        // A deterministic position with few pieces uses a low output bucket.
        let kk = Board::from_str("8/8/8/3k4/8/3K4/4P3/8 w - - 0 1").unwrap();
        let state2 = NNUEState::from_board(&kk);
        let eval2 = state2.evaluate(&kk);
        assert!(eval2.abs() < 20_000, "endgame eval out of range: {eval2}");
    }
}
