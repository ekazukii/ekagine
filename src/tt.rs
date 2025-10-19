use chess::{ChessMove, File, Piece, Rank, Square};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use crate::has_lse2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    Lower,
    Upper,
}

impl Default for TTFlag {
    fn default() -> Self {
        TTFlag::Exact
    }
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

// ------------------------ Fast path using LSE2 ------------------------
#[repr(C, align(16))]
struct Lse2Impl {
    pair: core::cell::UnsafeCell<[u64; 2]>,
}
impl Default for Lse2Impl {
    #[inline]
    fn default() -> Self {
        Self { pair: core::cell::UnsafeCell::new([0, 0]) }
    }
}
unsafe impl Send for Lse2Impl {}
unsafe impl Sync for Lse2Impl {}

impl Lse2Impl {
    #[inline]
    fn load(&self) -> (u64, u64) {
        unsafe {
            use core::arch::asm;
            let addr = (*self.pair.get()).as_ptr();
            let mut lo: u64;
            let mut hi: u64;
            // Acquire load: DMB ISHLD after LDP
            asm!(
            "ldp {lo}, {hi}, [{addr}]",
            "dmb ishld",
            lo = out(reg) lo,
            hi = out(reg) hi,
            addr = in(reg) addr,
            options(nostack, preserves_flags)
            );
            (lo, hi)
        }
    }

    #[inline]
    fn store(&self, key: u64, data: u64) {
        unsafe {
            use core::arch::asm;
            let addr = (*self.pair.get()).as_mut_ptr();
            // Release store: DMB ISH before STP
            asm!(
            "dmb ish",
            "stp {key}, {data}, [{addr}]",
            key = in(reg) key,
            data = in(reg) data,
            addr = in(reg) addr,
            options(nostack, preserves_flags)
            );
        }
    }
}

// ------------------------ Fallback XOR scheme ------------------------

struct FallbackImpl {
    key: AtomicU64,
    data: AtomicU64,
}
impl Default for FallbackImpl {
    #[inline]
    fn default() -> Self {
        Self { key: AtomicU64::new(0), data: AtomicU64::new(0) }
    }
}
impl FallbackImpl {
    #[inline]
    fn load(&self) -> (u64, u64) {
        let key_xor = self.key.load(Ordering::Acquire);
        let data = self.data.load(Ordering::Acquire);
        (key_xor ^ data, data)
    }
    #[inline]
    fn store(&self, key: u64, data: u64) {
        let key_xor = key ^ data;
        self.data.store(data, Ordering::Relaxed);
        self.key.store(key_xor, Ordering::Release);
    }
}

// ------------------------ Public type with runtime dispatch ------------------------

enum Backend {
    Lse2(Lse2Impl),
    Fallback(FallbackImpl),
}

pub struct CompressedTTEntry(Backend);

impl Default for CompressedTTEntry {
    #[inline]
    fn default() -> Self {
        if has_lse2() {
            CompressedTTEntry(Backend::Lse2(Lse2Impl::default()))
        } else {
            CompressedTTEntry(Backend::Fallback(FallbackImpl::default()))
        }
    }
}

unsafe impl Send for CompressedTTEntry {}
unsafe impl Sync for CompressedTTEntry {}

impl CompressedTTEntry {
    #[inline]
    pub fn load(&self) -> (u64, u64) {
        match &self.0 {
            Backend::Lse2(x) => x.load(),
            Backend::Fallback(x) => x.load(),
        }
    }

    #[inline]
    pub fn store(&self, key: u64, data: u64) {
        match &self.0 {
            Backend::Lse2(x) => x.store(key, data),
            Backend::Fallback(x) => x.store(key, data),
        }
    }

    #[inline]
    pub fn store_entry(&self, entry: TTEntry) {
        let (key, data) = entry.into();
        self.store(key, data);
    }
}



const TT_DEFAULT_MB: usize = 32;
const TT_REPLACE_OFFSET: usize = 2;

const MOVE_NONE: u16 = 0xFFFF;
const EVAL_NONE: u16 = 0x8000;

const SCORE_SHIFT: u64 = 16;
const DEPTH_SHIFT: u64 = 32;
const FLAG_SHIFT: u64 = 39;
const EVAL_SHIFT: u64 = 41;
const AGE_SHIFT: u64 = 57;
const PV_SHIFT: u64 = 63;

const MOVE_MASK: u64 = 0xFFFF;
const SCORE_MASK: u64 = 0xFFFF;
const DEPTH_MASK: u64 = 0x7F;
const FLAG_MASK: u64 = 0x3;
const EVAL_MASK: u64 = 0xFFFF;
const AGE_MASK: u64 = 0x3F;

impl From<(u64, u64)> for TTEntry {
    fn from(raw: (u64, u64)) -> Self {
        let (key, data) = raw;
        if key == 0 && data == 0 {
            return TTEntry::default();
        }

        let best_move_bits = (data & MOVE_MASK) as u16;
        let score_bits = ((data >> SCORE_SHIFT) & SCORE_MASK) as u16;
        let depth_bits = ((data >> DEPTH_SHIFT) & DEPTH_MASK) as u8;
        let flag_bits = ((data >> FLAG_SHIFT) & FLAG_MASK) as u8;
        let eval_bits = ((data >> EVAL_SHIFT) & EVAL_MASK) as u16;
        let age_bits = ((data >> AGE_SHIFT) & AGE_MASK) as u8;
        let pv_bit = ((data >> PV_SHIFT) & 0x1) != 0;

        TTEntry {
            key,
            best_move: decode_move(best_move_bits),
            score: (score_bits as i16) as i32,
            depth: depth_bits as i16 - 1,
            flag: match flag_bits {
                0 => TTFlag::Exact,
                1 => TTFlag::Lower,
                2 => TTFlag::Upper,
                _ => TTFlag::Exact,
            },
            eval: if eval_bits == EVAL_NONE {
                None
            } else {
                Some((eval_bits as i16) as i32)
            },
            age: age_bits,
            pv: pv_bit,
        }
    }
}

impl From<TTEntry> for (u64, u64) {
    fn from(entry: TTEntry) -> Self {
        let age_bits = entry.age & AGE_MASK as u8;
        let score_clamped = entry.score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let depth_clamped = entry.depth.clamp(-1, 126) + 1;
        let eval_bits = match entry.eval {
            Some(val) => {
                let clamped = val.clamp(-32767, 32767) as i16;
                clamped as u16
            }
            None => EVAL_NONE,
        };

        let mut data = 0u64;
        data |= encode_move(entry.best_move) as u64;
        data |= (score_clamped as u16 as u64) << SCORE_SHIFT;
        data |= ((depth_clamped as u8 as u64) & DEPTH_MASK) << DEPTH_SHIFT;
        data |= (encode_flag(entry.flag) as u64) << FLAG_SHIFT;
        data |= (eval_bits as u64) << EVAL_SHIFT;
        data |= ((age_bits as u64) & AGE_MASK) << AGE_SHIFT;
        data |= (entry.pv as u64) << PV_SHIFT;

        (entry.key, data)
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
/// Each Zobrist key hashes to a single slot. Replacement is guided by the current search
/// age, the node depth, and whether the stored value is part of the principal variation.
pub struct TranspositionTable {
    table: Vec<CompressedTTEntry>,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    pub fn with_mb(size_mb: usize) -> Self {
        let bytes_per_slot = std::mem::size_of::<CompressedTTEntry>().max(1);
        let requested_bytes = size_mb.saturating_mul(1024 * 1024).max(bytes_per_slot);

        let mut slots = 1usize;
        while slots.saturating_mul(bytes_per_slot) < requested_bytes {
            slots <<= 1;
            if slots == 0 {
                slots = 1;
                break;
            }
        }

        let mut table = Vec::with_capacity(slots);
        for _ in 0..slots {
            table.push(CompressedTTEntry::default());
        }

        Self {
            table,
            age: AtomicU8::new(0),
        }
    }

    #[inline]
    fn get_key(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.table.len() as u128;

        ((key * len) >> 64) as usize
    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = self.get_key(key);
        let slot = &self.table[idx];
        let (stored_key, stored_data) = slot.load();
        if stored_key == key {
            let stored_entry = TTEntry::from((stored_key, stored_data));
            Some(Self::visible_entry(stored_entry))
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
    ) {
        debug_assert!(!self.table.is_empty());
        let idx = self.get_key(key);
        let slot = &self.table[idx];
        let (stored_key, stored_data) = slot.load();
        let current_entry = TTEntry::from((stored_key, stored_data));
        let pv = flag == TTFlag::Exact;
        let same_position = stored_key == key;
        let previous_entry = if same_position {
            Some(current_entry)
        } else {
            None
        };
        let previous_age = current_entry.age;
        let age_now = self.age.load(Ordering::Relaxed) & 0x3F;

        let old_depth = previous_entry.map_or(0, |entry| entry.depth.max(0) as usize);

        let should_replace = age_now != previous_age
            || !same_position
            || flag == TTFlag::Exact
            || depth.max(0) as usize + TT_REPLACE_OFFSET + 2 * usize::from(pv) > old_depth;

        if !should_replace {
            return;
        }

        if best_move.is_none() {
            if let Some(entry) = previous_entry {
                best_move = entry.best_move;
            }
        }

        let final_eval = match eval {
            Some(val) => Some(val),
            None => previous_entry.and_then(|entry| entry.eval),
        };

        let new_entry = TTEntry {
            depth,
            score,
            flag,
            best_move,
            eval: final_eval,
            key,
            age: age_now,
            pv,
        };

        slot.store_entry(new_entry);
    }

    pub fn bump_generation(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for slot in &self.table {
            slot.store(0, 0);
        }
        self.age.store(0, Ordering::Relaxed);
    }
}

impl TranspositionTable {
    #[inline]
    fn visible_entry(entry: TTEntry) -> TTEntry {
        TTEntry {
            key: 0,
            depth: entry.depth,
            score: entry.score,
            flag: entry.flag,
            best_move: entry.best_move,
            eval: entry.eval,
            age: 0,
            pv: false,
        }
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}
