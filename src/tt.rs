use chess::{ChessMove, File, Piece, Rank, Square};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

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

pub struct CompressedTTEntry {
    pub key: AtomicU64,
    pub data: AtomicU64,
}

impl Default for CompressedTTEntry {
    fn default() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

impl CompressedTTEntry {
    fn load(&self) -> (u64, u64) {
        let key_xor = self.key.load(Ordering::Acquire);
        let data = self.data.load(Ordering::Acquire);
        (key_xor ^ data, data)
    }

    fn store(&self, key: u64, data: u64) {
        let key_xor = key ^ data;
        self.data.store(data, Ordering::Relaxed);
        self.key.store(key_xor, Ordering::Release);
    }

    fn store_entry(&self, entry: TTEntry) {
        let (key, data) = entry.into();
        self.store(key, data);
    }
}

const TT_DEFAULT_MB: usize = 32;
const TT_REPLACE_OFFSET: usize = 2;
const CLUSTER_SIZE: usize = 4;

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

/// Transposition table storing 4 entries per bucket.
///
/// Each position is hashed to a small cluster and replacement considers generation, depth,
/// and PV information to reduce overwriting valuable entries under heavy load.
pub struct TranspositionTable {
    table: Vec<CompressedTTEntry>,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    pub fn with_mb(size_mb: usize) -> Self {
        let bytes_per_bucket = std::mem::size_of::<CompressedTTEntry>().max(1) * CLUSTER_SIZE;
        let requested_bytes = size_mb.saturating_mul(1024 * 1024).max(bytes_per_bucket);

        let mut buckets = 1usize;
        while buckets.saturating_mul(bytes_per_bucket) < requested_bytes {
            buckets <<= 1;
            if buckets == 0 {
                buckets = 1;
                break;
            }
        }

        let total_slots = buckets.saturating_mul(CLUSTER_SIZE).max(CLUSTER_SIZE);

        let mut table = Vec::with_capacity(total_slots);
        for _ in 0..total_slots {
            table.push(CompressedTTEntry::default());
        }

        Self {
            table,
            age: AtomicU8::new(0),
        }
    }

    #[inline]
    fn bucket_count(&self) -> usize {
        let len = self.table.len();
        if len == 0 {
            1
        } else {
            (len / CLUSTER_SIZE).max(1)
        }
    }

    #[inline]
    fn get_key(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.bucket_count() as u128;

        ((key * len) >> 64) as usize
    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let bucket_idx = self.get_key(key);
        let start = bucket_idx * CLUSTER_SIZE;

        for offset in 0..CLUSTER_SIZE {
            let slot = &self.table[start + offset];
            let (stored_key, stored_data) = slot.load();
            if stored_key == key {
                let stored_entry = TTEntry::from((stored_key, stored_data));
                return Some(Self::visible_entry(stored_entry));
            }
        }

        None
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

        let bucket_idx = self.get_key(key);
        let start = bucket_idx * CLUSTER_SIZE;
        let age_now = self.age.load(Ordering::Relaxed) & 0x3F;
        let new_is_pv = flag == TTFlag::Exact;

        let mut previous_entry: Option<TTEntry> = None;
        let mut target_slot: Option<usize> = None;
        let mut empty_slot: Option<usize> = None;
        let mut weakest_slot: usize = start;
        let mut weakest_score: i32 = i32::MAX;

        for offset in 0..CLUSTER_SIZE {
            let idx = start + offset;
            let slot = &self.table[idx];
            let (stored_key, stored_data) = slot.load();

            if stored_key == key {
                let existing = TTEntry::from((stored_key, stored_data));
                let existing_depth = existing.depth.max(0) as usize;
                let new_depth = depth.max(0) as usize;
                let should_replace = existing.age != age_now
                    || new_is_pv
                    || new_depth + TT_REPLACE_OFFSET + 2 * usize::from(new_is_pv) > existing_depth;

                if !should_replace {
                    if best_move.is_none() {
                        best_move = existing.best_move;
                    }
                    return;
                }

                if best_move.is_none() {
                    best_move = existing.best_move;
                }

                previous_entry = Some(existing);
                target_slot = Some(idx);
                break;
            }

            if stored_key == 0 {
                if empty_slot.is_none() {
                    empty_slot = Some(idx);
                }
                continue;
            }

            let existing = TTEntry::from((stored_key, stored_data));
            let score = replacement_score(&existing, age_now);
            if score < weakest_score {
                weakest_score = score;
                weakest_slot = idx;
            }
        }

        let slot_index = target_slot.or(empty_slot).unwrap_or(weakest_slot);
        let slot = &self.table[slot_index];

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
            pv: new_is_pv,
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

#[inline]
fn replacement_score(entry: &TTEntry, current_age: u8) -> i32 {
    let depth = entry.depth.max(0) as i32;
    let pv_bonus = if entry.pv { 32 } else { 0 };
    let age_diff = current_age.wrapping_sub(entry.age) & 0x3F;
    let age_penalty = (age_diff as i32) * 8;
    depth + pv_bonus - age_penalty
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}
