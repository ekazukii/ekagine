use std::sync::atomic::AtomicU64;
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

//impl From<(u64, u64)> for TTEntry {}


const TT_DEFAULT_MB: usize = 32;
const TT_REPLACE_OFFSET: usize = 2;

/// Direct-mapped transposition table.
///
/// Each Zobrist key hashes to a single slot. Replacement is guided by the current search
/// age, the node depth, and whether the stored value is part of the principal variation.
pub struct TranspositionTable {
    table: Vec<TTEntry>,
    age: u8,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    pub fn with_mb(size_mb: usize) -> Self {
        let bytes_per_slot = std::mem::size_of::<TTEntry>().max(1);
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
        table.resize(slots, TTEntry::default());

        Self { table, age: 0 }
    }

    #[inline]
    fn get_key(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.table.len() as u128;

        ((key * len) >> 64) as usize
    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = self.get_key(key);
        let stored = self.table[idx];
        if stored.key == key {
            Some(Self::visible_entry(stored))
        } else {
            None
        }
    }

    pub fn store(
        &mut self,
        key: u64,
        depth: i16,
        score: i32,
        flag: TTFlag,
        mut best_move: Option<ChessMove>,
        eval: Option<i32>,
    ) {
        debug_assert!(!self.table.is_empty());
        let idx = self.get_key(key);
        let entry = &mut self.table[idx];
        let pv = flag == TTFlag::Exact;
        let same_position = entry.key == key;
        let previous_entry = if same_position { Some(*entry) } else { None };
        let previous_age = entry.age;

        let old_depth = previous_entry.map_or(0, |entry| entry.depth.max(0) as usize);

        let should_replace = self.age != previous_age
            || !same_position
            || flag == TTFlag::Exact
            || depth.max(0) as usize + TT_REPLACE_OFFSET + 2 * usize::from(pv) > old_depth;

        if !should_replace {
            if let Some(mv) = best_move {
                entry.best_move = Some(mv);
            }
            if let Some(eval_val) = eval {
                entry.eval = Some(eval_val);
            }
            entry.age = self.age;
            entry.pv |= pv;
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

        *entry = TTEntry {
            depth,
            score,
            flag,
            best_move,
            eval: final_eval,
            key,
            age: self.age,
            pv,
        };
    }

    pub fn store_eval(&mut self, key: u64, eval: i32) {
        debug_assert!(!self.table.is_empty());
        let idx = self.get_key(key);
        let entry = &mut self.table[idx];
        let same_position = entry.key == key;

        if same_position {
            entry.eval = Some(eval);
            entry.age = self.age;
            return;
        }

        *entry = TTEntry {
            depth: -1,
            score: 0,
            flag: TTFlag::Exact,
            best_move: None,
            eval: Some(eval),
            key,
            age: self.age,
            pv: false,
        };
    }

    pub fn bump_generation(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        for entry in &mut self.table {
            *entry = TTEntry::default();
        }
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
