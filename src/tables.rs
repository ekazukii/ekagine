use crate::engine_core::{ChessMove, Color, File, Piece, Rank, Square};
use std::sync::atomic::{AtomicI16, AtomicU64, AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TTFlag {
    #[default]
    Exact,
    Lower,
    Upper,
}

/// Uncompressed representation of a single TTEntry
/// This use globally the same representation as Viridithas engine for PackedInfo (Age/Flag/PV)
/// but using the full zobrish hash of 64 bits
///
/// The total packed representation is 128 bits, stored as below
/// key -> 64b          : 64b
/// best_move -> 16b    : 80b
/// score -> 16b        : 96b
/// depth -> 7b         : 103b
/// flag -> 2b          : 105b
/// eval -> 16v         : 121b
/// age -> 6b           : 127b
/// pv -> 1b            : 128b
#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub key: u64,
    pub best_move: Option<ChessMove>,
    pub score: i32,
    pub depth: i16,
    pub flag: TTFlag,
    pub eval: Option<i32>,
    pub age: u8,
    pub pv: bool,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            depth: 0,
            score: 0,
            flag: TTFlag::Exact,
            best_move: None,
            eval: None,
            key: 0,
            age: 0,
            pv: false,
        }
    }
}

const TT_DEFAULT_MB: usize = 128;
const TT_REPLACE_OFFSET: usize = 2;

const MOVE_NONE: u16 = 0xFFFF;
// Sentinel for "no static eval cached" in the parallel eval array. -32768 is
// outside the clamped eval range (±32767), so it never collides with a real eval.
const EVAL_NONE_I16: i16 = i16::MIN;

// Single-u64 entry layout (8 bytes/slot, direct-mapped). The position is keyed
// by a 16-bit checksum (the low bits of the Zobrist key — the slot index uses
// the high bits, so the two are independent). Index + checksum give ~40 bits of
// discrimination; the residual ~1/2^40 collision is caught by the TT-move
// legality filter in the move generator. A single 64-bit atomic load/store is
// never torn, so no lockless key/data XOR is needed. The static eval is NOT
// stored (it doesn't fit in 8 bytes and is cheap to recompute incrementally).
//
//   checksum 16 | move 16 | score 16 | depth 7 | flag 2 | age 6 | pv 1  = 64
const CHECKSUM_SHIFT: u64 = 0;
const MOVE_SHIFT: u64 = 16;
const SCORE_SHIFT: u64 = 32;
const DEPTH_SHIFT: u64 = 48;
const FLAG_SHIFT: u64 = 55;
const AGE_SHIFT: u64 = 57;
const PV_SHIFT: u64 = 63;

const CHECKSUM_MASK: u64 = 0xFFFF;
const MOVE_MASK: u64 = 0xFFFF;
const SCORE_MASK: u64 = 0xFFFF;
const DEPTH_MASK: u64 = 0x7F;
const FLAG_MASK: u64 = 0x3;
const AGE_MASK: u64 = 0x3F;

#[inline]
fn key_checksum(key: u64) -> u16 {
    (key & CHECKSUM_MASK) as u16
}

#[allow(clippy::too_many_arguments)]
fn pack_entry(
    checksum: u16,
    best_move: Option<ChessMove>,
    score: i32,
    depth: i16,
    flag: TTFlag,
    age: u8,
    pv: bool,
) -> u64 {
    let score_clamped = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let depth_clamped = (depth.clamp(-1, 126) + 1) as u64;
    let mut data = 0u64;
    data |= (checksum as u64) << CHECKSUM_SHIFT;
    data |= (encode_move(best_move) as u64) << MOVE_SHIFT;
    data |= (score_clamped as u16 as u64) << SCORE_SHIFT;
    data |= (depth_clamped & DEPTH_MASK) << DEPTH_SHIFT;
    data |= (encode_flag(flag) as u64) << FLAG_SHIFT;
    data |= ((age as u64) & AGE_MASK) << AGE_SHIFT;
    data |= (pv as u64) << PV_SHIFT;
    data
}

impl From<u64> for TTEntry {
    /// Unpack a stored 64-bit slot. Only called on non-empty slots (probe/store
    /// guard `data != 0`). `key` is not recoverable (only the checksum is
    /// stored) and `eval` is not stored, so both come back zeroed/None.
    fn from(data: u64) -> Self {
        let move_bits = ((data >> MOVE_SHIFT) & MOVE_MASK) as u16;
        let score_bits = ((data >> SCORE_SHIFT) & SCORE_MASK) as u16;
        let depth_bits = ((data >> DEPTH_SHIFT) & DEPTH_MASK) as u8;
        let flag_bits = ((data >> FLAG_SHIFT) & FLAG_MASK) as u8;
        let age_bits = ((data >> AGE_SHIFT) & AGE_MASK) as u8;
        let pv_bit = ((data >> PV_SHIFT) & 0x1) != 0;

        TTEntry {
            key: 0,
            best_move: decode_move(move_bits),
            score: (score_bits as i16) as i32,
            depth: depth_bits as i16 - 1,
            flag: match flag_bits {
                0 => TTFlag::Exact,
                1 => TTFlag::Lower,
                2 => TTFlag::Upper,
                _ => TTFlag::Exact,
            },
            eval: None,
            age: age_bits,
            pv: pv_bit,
        }
    }
}

fn encode_move(mv: Option<ChessMove>) -> u16 {
    mv.map_or(MOVE_NONE, |mv| {
        let from = mv.get_source().to_index() as u16 & 0x3F;
        let to = mv.get_dest().to_index() as u16 & 0x3F;
        let promo = encode_promotion(mv.get_promotion()) & 0xF;
        from | (to << 6) | (promo << 12)
    })
}

fn decode_move(bits: u16) -> Option<ChessMove> {
    if bits == MOVE_NONE {
        return None;
    }

    let from_idx = bits & 0x3F;
    let to_idx = (bits >> 6) & 0x3F;
    let promo_code = (bits >> 12) & 0xF;

    let from = square_from_index(from_idx)?;
    let to = square_from_index(to_idx)?;
    let promotion = decode_promotion(promo_code);

    Some(ChessMove::new(from, to, promotion))
}

fn encode_promotion(piece: Option<Piece>) -> u16 {
    match piece {
        None => 0,
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        Some(Piece::Queen) => 4,
        Some(_) => 0,
    }
}

fn decode_promotion(code: u16) -> Option<Piece> {
    match code {
        0 => None,
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    }
}

fn square_from_index(idx: u16) -> Option<Square> {
    if idx >= 64 {
        return None;
    }
    let rank = Rank::from_index((idx / 8) as usize);
    let file = File::from_index((idx % 8) as usize);
    Some(Square::make_square(rank, file))
}

fn encode_flag(flag: TTFlag) -> u8 {
    match flag {
        TTFlag::Exact => 0,
        TTFlag::Lower => 1,
        TTFlag::Upper => 2,
    }
}

/// Direct-mapped transposition table.
///
/// Each Zobrist key maps to a single 8-byte main slot (`AtomicU64`) storing a
/// 16-bit checksum of the key, plus a parallel 2-byte static-eval cache (10 B
/// per slot total). A single 64-bit atomic load/store is tear-free, so there is
/// no lockless key/data XOR. Replacement keeps the recency policy that suits
/// this engine: a colliding position always overwrites; a same-position store is
/// depth-preferred. Keeping the eval out of the main entry lets the entry stay
/// 8 B (more capacity) while still caching the eval (no NNUE recompute per hit).
pub struct TranspositionTable {
    table: Vec<AtomicU64>,
    evals: Vec<AtomicI16>,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    pub fn with_mb(size_mb: usize) -> Self {
        // 8 B main entry + 2 B side eval per slot. The multiply-shift index works
        // for any slot count, so we size exactly to the requested bytes (no
        // power-of-two rounding / overshoot).
        let bytes_per_slot =
            (std::mem::size_of::<AtomicU64>() + std::mem::size_of::<AtomicI16>()).max(1);
        let requested_bytes = size_mb.saturating_mul(1024 * 1024).max(bytes_per_slot);
        let slots = (requested_bytes / bytes_per_slot).max(1);

        let mut table = Vec::with_capacity(slots);
        let mut evals = Vec::with_capacity(slots);
        for _ in 0..slots {
            table.push(AtomicU64::new(0));
            evals.push(AtomicI16::new(EVAL_NONE_I16));
        }

        Self {
            table,
            evals,
            age: AtomicU8::new(0),
        }
    }

    /// Approximate fill level in permille (0..=1000), sampled over the first
    /// 1000 slots — the standard UCI `hashfull` estimate.
    pub fn hashfull(&self) -> usize {
        let sample = self.table.len().min(1000);
        if sample == 0 {
            return 0;
        }
        let mut used = 0usize;
        for slot in self.table.iter().take(sample) {
            if slot.load(Ordering::Relaxed) != 0 {
                used += 1;
            }
        }
        used * 1000 / sample
    }

    #[inline]
    fn get_key(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.table.len() as u128;

        ((key * len) >> 64) as usize
    }

    #[inline]
    fn slot_checksum(data: u64) -> u16 {
        ((data >> CHECKSUM_SHIFT) & CHECKSUM_MASK) as u16
    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = self.get_key(key);
        let data = self.table[idx].load(Ordering::Relaxed);
        if data != 0 && Self::slot_checksum(data) == key_checksum(key) {
            let mut entry = TTEntry::from(data);
            let cached_eval = self.evals[idx].load(Ordering::Relaxed);
            if cached_eval != EVAL_NONE_I16 {
                entry.eval = Some(cached_eval as i32);
            }
            Some(entry)
        } else {
            None
        }
    }

    pub fn store(
        &self,
        key: u64,
        depth: i16,
        score: i32,
        flag: TTFlag,
        mut best_move: Option<ChessMove>,
        eval: Option<i32>,
        pv: bool,
    ) {
        debug_assert!(!self.table.is_empty());
        let idx = self.get_key(key);
        let slot = &self.table[idx];
        let checksum = key_checksum(key);
        let current = slot.load(Ordering::Relaxed);
        let same_position = current != 0 && Self::slot_checksum(current) == checksum;
        let age_now = self.age.load(Ordering::Relaxed) & (AGE_MASK as u8);
        let exact = flag == TTFlag::Exact;

        if same_position {
            // Depth-preferred only for the same position; a colliding or empty
            // slot is always overwritten (recency) — this engine's existing
            // direct-mapped policy.
            let prev = TTEntry::from(current);
            let old_depth = prev.depth.max(0) as usize;
            let should_replace = age_now != prev.age
                || exact
                || depth.max(0) as usize + TT_REPLACE_OFFSET + 2 * usize::from(exact) > old_depth;
            if !should_replace {
                return;
            }
            if best_move.is_none() {
                best_move = prev.best_move;
            }
        }

        slot.store(
            pack_entry(checksum, best_move, score, depth, flag, age_now, pv),
            Ordering::Relaxed,
        );
        // Side eval cache: store the provided eval. On a no-eval store keep the
        // previous value only when refreshing the same position; a new position
        // with no eval clears it (so we never read a prior occupant's eval).
        match eval {
            Some(v) => self.evals[idx].store(v.clamp(-32767, 32767) as i16, Ordering::Relaxed),
            None if !same_position => self.evals[idx].store(EVAL_NONE_I16, Ordering::Relaxed),
            None => {}
        }
    }

    pub fn bump_generation(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for slot in &self.table {
            slot.store(0, Ordering::Relaxed);
        }
        for e in &self.evals {
            e.store(EVAL_NONE_I16, Ordering::Relaxed);
        }
        self.age.store(0, Ordering::Relaxed);
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------

const MAX_KILLER_MOVES: usize = 2;

#[derive(Debug, Clone)]
pub struct KillerTable {
    moves: Vec<[Option<ChessMove>; MAX_KILLER_MOVES]>,
}

impl KillerTable {
    /// Clear all killer slots in place (reused between unrelated searches).
    pub fn reset_all(&mut self) {
        for m in self.moves.iter_mut() {
            *m = [None; MAX_KILLER_MOVES];
        }
    }

    pub fn new(initial_depth: usize) -> Self {
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

    pub fn killers_for(&mut self, ply: usize) -> [Option<ChessMove>; MAX_KILLER_MOVES] {
        self.ensure_capacity(ply);
        self.moves[ply]
    }

    pub fn record(&mut self, ply: usize, mv: ChessMove) {
        self.ensure_capacity(ply);
        let entry = &mut self.moves[ply];
        if entry.contains(&Some(mv)) {
            if entry[1] == Some(mv) {
                entry.swap(0, 1);
            }
            return;
        }
        entry[1] = entry[0];
        entry[0] = Some(mv);
    }
}

const HISTORY_CAP: i32 = 16384;

#[derive(Debug, Clone)]
pub struct HistoryTable {
    entries: [[[i32; 64]; 64]; 2],
}

impl HistoryTable {
    pub fn new() -> Self {
        Self {
            entries: [[[0; 64]; 64]; 2],
        }
    }

    #[inline]
    pub fn score(&self, color: Color, mv: ChessMove) -> i32 {
        let color_idx = color.to_index();
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        self.entries[color_idx][from][to]
    }

    #[inline]
    fn update(&mut self, color: Color, mv: ChessMove, delta: i32) {
        let color_idx = color.to_index();
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        let entry = &mut self.entries[color_idx][from][to];
        *entry += delta - *entry * delta.abs() / HISTORY_CAP;
    }

    pub fn reward(&mut self, color: Color, mv: ChessMove, depth: i16) {
        let bonus = history_bonus(depth);
        if bonus > 0 {
            self.update(color, mv, bonus);
        }
    }

    pub fn reward_soft(&mut self, color: Color, mv: ChessMove, depth: i16) {
        let bonus = history_bonus(depth) / 2;
        if bonus > 0 {
            self.update(color, mv, bonus.max(1));
        }
    }

    pub fn penalize(&mut self, color: Color, mv: ChessMove, depth: i16) {
        let malus = history_malus(depth);
        if malus > 0 {
            self.update(color, mv, -malus);
        }
    }
}

impl Default for HistoryTable {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn history_bonus(depth: i16) -> i32 {
    // Stockfish-style: a meaningfully large bonus relative to the (now much
    // smaller) gravity cap, so history actually accumulates a usable signal
    // for both move ordering and history-based LMR.
    let d = (depth.max(1) as i32).min(20);
    (16 * d * d + 32 * d - 16).min(1536)
}

#[inline]
fn history_malus(depth: i16) -> i32 {
    let d = (depth.max(1) as i32).min(20);
    (16 * d * d + 32 * d - 16).min(1536)
}

/// Countermove Table - tracks the best quiet move response to opponent moves
/// Indexed by [color][from_square][to_square] of the previous move
pub struct CountermoveTable {
    entries: [[[Option<ChessMove>; 64]; 64]; 2],
}

impl CountermoveTable {
    pub fn new() -> Self {
        Self {
            entries: [[[None; 64]; 64]; 2],
        }
    }

    /// Get the countermove for a given previous move
    #[inline]
    pub fn get(&self, prev_move: ChessMove, prev_color: Color) -> Option<ChessMove> {
        let color_idx = prev_color.to_index();
        let from = prev_move.get_source().to_index();
        let to = prev_move.get_dest().to_index();
        self.entries[color_idx][from][to]
    }

    /// Record a countermove that caused a beta cutoff
    #[inline]
    pub fn record(&mut self, prev_move: ChessMove, prev_color: Color, countermove: ChessMove) {
        let color_idx = prev_color.to_index();
        let from = prev_move.get_source().to_index();
        let to = prev_move.get_dest().to_index();
        self.entries[color_idx][from][to] = Some(countermove);
    }
}

impl Default for CountermoveTable {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Capture History
// ─────────────────────────────────────────────────────────────────────────────

/// Capture History Table
///
/// Tracks which captures work well, indexed by
/// [color][moving_piece][to_square][captured_piece].
/// 2 × 6 × 64 × 6 = 4,608 entries (~18 KB).
pub struct CaptureHistoryTable {
    entries: [[[[i32; 6]; 64]; 6]; 2],
}

impl CaptureHistoryTable {
    pub fn new() -> Self {
        Self {
            entries: [[[[0i32; 6]; 64]; 6]; 2],
        }
    }

    #[inline]
    pub fn score(&self, color: Color, piece: Piece, to: Square, captured: Piece) -> i32 {
        self.entries[color.to_index()][piece.to_index()][to.to_index()][captured.to_index()]
    }

    #[inline]
    fn update(&mut self, color: Color, piece: Piece, to: Square, captured: Piece, delta: i32) {
        let entry = &mut self.entries[color.to_index()][piece.to_index()][to.to_index()]
            [captured.to_index()];
        // Gravity-based update to prevent saturation
        *entry += delta - *entry * delta.abs() / HISTORY_CAP;
    }

    pub fn reward(&mut self, color: Color, piece: Piece, to: Square, captured: Piece, depth: i16) {
        let bonus = history_bonus(depth);
        if bonus > 0 {
            self.update(color, piece, to, captured, bonus);
        }
    }

    pub fn penalize(
        &mut self,
        color: Color,
        piece: Piece,
        to: Square,
        captured: Piece,
        depth: i16,
    ) {
        let malus = history_malus(depth);
        if malus > 0 {
            self.update(color, piece, to, captured, -malus);
        }
    }
}

impl Default for CaptureHistoryTable {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Continuation History
// ─────────────────────────────────────────────────────────────────────────────

/// Continuation history: quiet-move history conditioned on a previous move in
/// the line. Indexed by (prev_piece, prev_to, cur_piece, cur_to). Two instances
/// are used during search: one keyed on the move 1 ply back (the opponent's
/// reply context) and one on the move 2 plies back (our own prior move). Each
/// table is 6*64*6*64 = 147,456 i32 (~0.59 MB), heap-allocated to keep the
/// search stack small.
const CONT_LEN: usize = 6 * 64 * 6 * 64;

pub struct ContHist {
    e: Box<[i32]>,
}

impl ContHist {
    pub fn new() -> Self {
        Self {
            e: vec![0i32; CONT_LEN].into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn idx(prev_piece: Piece, prev_to: Square, piece: Piece, to: Square) -> usize {
        let pp = prev_piece.to_index();
        let pt = prev_to.to_index();
        let p = piece.to_index();
        let t = to.to_index();
        ((pp * 64 + pt) * 6 + p) * 64 + t
    }

    #[inline]
    pub fn score(&self, prev_piece: Piece, prev_to: Square, piece: Piece, to: Square) -> i32 {
        self.e[Self::idx(prev_piece, prev_to, piece, to)]
    }

    #[inline]
    fn update(&mut self, prev_piece: Piece, prev_to: Square, piece: Piece, to: Square, delta: i32) {
        let entry = &mut self.e[Self::idx(prev_piece, prev_to, piece, to)];
        *entry += delta - *entry * delta.abs() / HISTORY_CAP;
    }

    pub fn reward(
        &mut self,
        prev_piece: Piece,
        prev_to: Square,
        piece: Piece,
        to: Square,
        depth: i16,
    ) {
        let bonus = history_bonus(depth);
        if bonus > 0 {
            self.update(prev_piece, prev_to, piece, to, bonus);
        }
    }

    pub fn reward_soft(
        &mut self,
        prev_piece: Piece,
        prev_to: Square,
        piece: Piece,
        to: Square,
        depth: i16,
    ) {
        let bonus = history_bonus(depth) / 2;
        if bonus > 0 {
            self.update(prev_piece, prev_to, piece, to, bonus.max(1));
        }
    }

    pub fn penalize(
        &mut self,
        prev_piece: Piece,
        prev_to: Square,
        piece: Piece,
        to: Square,
        depth: i16,
    ) {
        let malus = history_malus(depth);
        if malus > 0 {
            self.update(prev_piece, prev_to, piece, to, -malus);
        }
    }
}

impl Default for ContHist {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Correction History
// ─────────────────────────────────────────────────────────────────────────────

/// Correction history: a per-(side, pawn-structure) running estimate of the
/// systematic error of the static eval vs. the search result. The applied
/// correction is bounded to a small centipawn range; the bucket is updated with
/// a depth-weighted, gravity-bounded step toward the observed residual.
const CORR_SIZE: usize = 16384;
const CORR_MASK: usize = CORR_SIZE - 1;
/// applied correction (cp) = bucket / CORR_GRAIN, clamped to ±CORR_MAX_CP.
const CORR_GRAIN: i32 = 256;
/// gravity bound on the bucket; equilibrium |bucket| ≈ CORR_LIMIT.
const CORR_LIMIT: i32 = 16384;
/// hard clamp on the applied pawn-structure correction in centipawns.
const CORR_MAX_CP: i32 = 64;
/// tighter clamp per non-pawn table: the two non-pawn tables stack on top of the
/// pawn table, so each is bounded smaller to keep the summed correction in a
/// sane range (max total ±128cp) and avoid over-correction in positions where
/// every key is stable. Prime SPSA knob if this regresses.
const CORR_MAX_CP_NP: i32 = 32;
/// continuation correction-history size, per side: previous move's piece × dest.
const CONT_CORR_SIZE: usize = 6 * 64;

pub struct CorrHist {
    /// keyed by (side-to-move, pawn-structure)
    pawn: Box<[i32]>,
    /// keyed by (side-to-move, white non-pawn piece placement)
    nonpawn_w: Box<[i32]>,
    /// keyed by (side-to-move, black non-pawn piece placement)
    nonpawn_b: Box<[i32]>,
    /// keyed by (side-to-move, previous move's piece & destination square)
    cont: Box<[i32]>,
}

impl CorrHist {
    pub fn new() -> Self {
        Self {
            pawn: vec![0i32; 2 * CORR_SIZE].into_boxed_slice(),
            nonpawn_w: vec![0i32; 2 * CORR_SIZE].into_boxed_slice(),
            nonpawn_b: vec![0i32; 2 * CORR_SIZE].into_boxed_slice(),
            cont: vec![0i32; 2 * CONT_CORR_SIZE].into_boxed_slice(),
        }
    }

    /// Zero the tables in place (reused between unrelated searches).
    pub fn clear(&mut self) {
        self.pawn.fill(0);
        self.nonpawn_w.fill(0);
        self.nonpawn_b.fill(0);
        self.cont.fill(0);
    }

    #[inline(always)]
    fn idx(side: Color, key: usize) -> usize {
        side.to_index() * CORR_SIZE + (key & CORR_MASK)
    }

    #[inline(always)]
    fn cont_idx(side: Color, piece: Piece, to: Square) -> usize {
        side.to_index() * CONT_CORR_SIZE + piece.to_index() * 64 + to.to_index()
    }

    #[inline]
    fn read_one(table: &[i32], side: Color, key: usize, max_cp: i32) -> i32 {
        (table[Self::idx(side, key)] / CORR_GRAIN).clamp(-max_cp, max_cp)
    }

    #[inline]
    fn update_one(table: &mut [i32], side: Color, key: usize, diff: i32, depth: i16) {
        let weight = (depth as i32).clamp(1, 8);
        let bonus = (diff * weight).clamp(-CORR_LIMIT / 4, CORR_LIMIT / 4);
        let entry = &mut table[Self::idx(side, key)];
        *entry += bonus - *entry * bonus.abs() / CORR_LIMIT;
    }

    /// Combined correction (cp): pawn-structure + per-color non-pawn placement +
    /// (when there is a previous move) the continuation key. Each sub-table is
    /// gravity-bounded and summed (Viridithas/Stockfish style); the caller clamps
    /// the final (eval + correction) away from mate scores.
    #[inline]
    pub fn correction(
        &self,
        side: Color,
        pawn_key: usize,
        npw_key: usize,
        npb_key: usize,
        cont: Option<(Piece, Square)>,
    ) -> i32 {
        let mut c = Self::read_one(&self.pawn, side, pawn_key, CORR_MAX_CP)
            + Self::read_one(&self.nonpawn_w, side, npw_key, CORR_MAX_CP_NP)
            + Self::read_one(&self.nonpawn_b, side, npb_key, CORR_MAX_CP_NP);
        if let Some((piece, to)) = cont {
            c += (self.cont[Self::cont_idx(side, piece, to)] / CORR_GRAIN)
                .clamp(-CORR_MAX_CP_NP, CORR_MAX_CP_NP);
        }
        c
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        side: Color,
        pawn_key: usize,
        npw_key: usize,
        npb_key: usize,
        cont: Option<(Piece, Square)>,
        diff: i32,
        depth: i16,
    ) {
        Self::update_one(&mut self.pawn, side, pawn_key, diff, depth);
        Self::update_one(&mut self.nonpawn_w, side, npw_key, diff, depth);
        Self::update_one(&mut self.nonpawn_b, side, npb_key, diff, depth);
        if let Some((piece, to)) = cont {
            let weight = (depth as i32).clamp(1, 8);
            let bonus = (diff * weight).clamp(-CORR_LIMIT / 4, CORR_LIMIT / 4);
            let i = Self::cont_idx(side, piece, to);
            self.cont[i] += bonus - self.cont[i] * bonus.abs() / CORR_LIMIT;
        }
    }
}

impl Default for CorrHist {
    fn default() -> Self {
        Self::new()
    }
}
