//! Clustered transposition table tuned for chess search.
//!
//! We keep the implementation here instead of relying on `HashMap` so we can control
//! memory layout and replacement. Probing the TT happens at almost every node, so
//! predictable cache-friendly access beats a generic hashmap even with a custom hasher.
//!
//! Chess engines also benefit from custom replacement rules (depth/age/PV-aware) and
//! avoiding allocator churn. A fixed bucket array lets us do both.

use chess::ChessMove;

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

#[derive(Clone, Copy, Debug, Default)]
pub struct TTEntry {
    pub depth: i16,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Option<ChessMove>,
    pub eval: Option<i32>,
}

const TT_BUCKET_SIZE: usize = 4;
const TT_DEFAULT_MB: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
struct TTSlot {
    key: u64,
    entry: TTEntry,
    generation: u8,
    is_pv: bool,
    used: bool,
}

/// Fixed-size clustered transposition table.
///
/// We hash the 64-bit Zobrist key onto a single bucket and scan at most four slots. This
/// keeps the probe O(1) and cache-friendly, while the bucketed replacement policy lets us
/// hang on to deep or PV-critical entries without allocating or resizing.
pub struct TranspositionTable {
    buckets: Vec<[TTSlot; TT_BUCKET_SIZE]>,
    mask: usize,
    generation: u8,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    pub fn with_mb(size_mb: usize) -> Self {
        let bytes_per_bucket = TT_BUCKET_SIZE * std::mem::size_of::<TTSlot>();
        let requested_bytes = size_mb.saturating_mul(1024 * 1024).max(bytes_per_bucket);

        let mut buckets = 1usize;
        while buckets * bytes_per_bucket < requested_bytes {
            buckets <<= 1;
            if buckets == 0 {
                buckets = 1;
                break;
            }
        }

        let mask = buckets - 1;
        let mut table = Vec::with_capacity(buckets);
        for _ in 0..buckets {
            table.push([TTSlot::default(); TT_BUCKET_SIZE]);
        }

        Self {
            buckets: table,
            mask,
            generation: 0,
        }
    }

    #[inline]
    fn bucket_index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    #[inline]
    fn bucket(&self, key: u64) -> &[TTSlot; TT_BUCKET_SIZE] {
        &self.buckets[self.bucket_index(key)]
    }

    #[inline]
    fn bucket_mut(&mut self, key: u64) -> &mut [TTSlot; TT_BUCKET_SIZE] {
        let idx = self.bucket_index(key);
        &mut self.buckets[idx]
    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        for slot in self.bucket(key) {
            if slot.used && slot.key == key {
                return Some(slot.entry);
            }
        }
        None
    }

    pub fn store(
        &mut self,
        key: u64,
        depth: i16,
        score: i32,
        flag: TTFlag,
        best_move: Option<ChessMove>,
        eval: Option<i32>,
    ) {
        let is_pv = flag == TTFlag::Exact;
        let generation = self.generation;
        let bucket = self.bucket_mut(key);

        if let Some(slot) = bucket.iter_mut().find(|slot| slot.used && slot.key == key) {
            let should_replace = depth > slot.entry.depth
                || (flag == TTFlag::Exact && slot.entry.flag != TTFlag::Exact)
                || (depth == slot.entry.depth && flag == TTFlag::Exact);

            if should_replace {
                slot.entry.depth = depth;
                slot.entry.score = score;
                slot.entry.flag = flag;
            }

            if let Some(mv) = best_move {
                slot.entry.best_move = Some(mv);
            }

            if let Some(eval_val) = eval {
                slot.entry.eval = Some(eval_val);
            }

            slot.generation = generation;
            slot.is_pv |= is_pv;
            return;
        }

        // No direct match: pick a victim within the bucket. Replacement is driven by
        // age/depth/PV priority so good data survives while stale nodes get evicted.
        let target_idx = Self::select_replacement(bucket, generation, is_pv);
        let slot = &mut bucket[target_idx];
        slot.used = true;
        slot.key = key;
        slot.entry = TTEntry {
            depth,
            score,
            flag,
            best_move,
            eval,
        };
        slot.generation = generation;
        slot.is_pv = is_pv;
    }

    pub fn store_eval(&mut self, key: u64, eval: i32) {
        let generation = self.generation;
        let bucket = self.bucket_mut(key);
        if let Some(slot) = bucket.iter_mut().find(|slot| slot.used && slot.key == key) {
            slot.entry.eval = Some(eval);
            slot.generation = generation;
            return;
        }

        let target_idx = Self::select_replacement(bucket, generation, false);
        let slot = &mut bucket[target_idx];
        slot.used = true;
        slot.key = key;
        slot.entry = TTEntry {
            depth: -1,
            score: 0,
            flag: TTFlag::Exact,
            best_move: None,
            eval: Some(eval),
        };
        slot.generation = generation;
        slot.is_pv = false;
    }

    fn select_replacement(bucket: &[TTSlot; TT_BUCKET_SIZE], generation: u8, prefer_pv: bool) -> usize {
        let mut target_idx = 0;
        let mut best_score = i32::MIN;

        for (i, slot) in bucket.iter().enumerate() {
            if !slot.used {
                return i;
            }

            let age = generation.wrapping_sub(slot.generation) as i32;
            let depth_penalty = slot.entry.depth.max(0) as i32;
            let pv_penalty = if slot.is_pv { 48 } else { 0 };
            let mut score = age * 64 - depth_penalty - pv_penalty;

            // If we are about to write a PV node, lean toward evicting non-PV data.
            if prefer_pv && slot.is_pv {
                score -= 64;
            }

            if score > best_score {
                best_score = score;
                target_idx = i;
            }
        }

        target_idx
    }

    pub fn bump_generation(&mut self) {
        // Each root iteration increments the generation. Older entries age out naturally
        // in replacements, which approximates an LRU without extra bookkeeping.
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = [TTSlot::default(); TT_BUCKET_SIZE];
        }
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}
