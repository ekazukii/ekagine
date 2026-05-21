use super::types::{BitBoard, Color, File, Square};
use std::sync::OnceLock;

#[derive(Copy, Clone)]
enum Slider {
    Bishop,
    Rook,
}

struct Tables {
    knight: [u64; 64],
    king: [u64; 64],
    pawn: [[u64; 64]; 2],
    between: [[u64; 64]; 64],
    line: [[u64; 64]; 64],
    file_bb: [u64; 8],
    adjacent_files: [u64; 8],
    bishop_masks: [u64; 64],
    bishop_magics: [u64; 64],
    bishop_shifts: [u32; 64],
    bishop_offsets: [usize; 64],
    bishop_attacks: Vec<u64>,
    rook_masks: [u64; 64],
    rook_magics: [u64; 64],
    rook_shifts: [u32; 64],
    rook_offsets: [usize; 64],
    rook_attacks: Vec<u64>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

/// Initialize all lookup tables. Idempotent.
pub fn ensure_init() {
    let _ = TABLES.get_or_init(init_tables);
}

#[inline(always)]
fn t() -> &'static Tables {
    TABLES.get_or_init(init_tables)
}

fn init_tables() -> Tables {
    let knight = init_knight();
    let king = init_king();
    let pawn = init_pawn();
    let between = init_between();
    let line = init_line();
    let file_bb = init_file_bb();
    let adjacent_files = init_adjacent(&file_bb);
    let (bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_attacks) =
        init_magic(Slider::Bishop);
    let (rook_masks, rook_magics, rook_shifts, rook_offsets, rook_attacks) =
        init_magic(Slider::Rook);
    Tables {
        knight,
        king,
        pawn,
        between,
        line,
        file_bb,
        adjacent_files,
        bishop_masks,
        bishop_magics,
        bishop_shifts,
        bishop_offsets,
        bishop_attacks,
        rook_masks,
        rook_magics,
        rook_shifts,
        rook_offsets,
        rook_attacks,
    }
}

fn init_knight() -> [u64; 64] {
    let mut table = [0u64; 64];
    let deltas: [(i32, i32); 8] = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ];
    for sq in 0..64 {
        let r = (sq / 8) as i32;
        let f = (sq % 8) as i32;
        let mut bb = 0u64;
        for &(dr, df) in &deltas {
            let nr = r + dr;
            let nf = f + df;
            if (0..8).contains(&nr) && (0..8).contains(&nf) {
                bb |= 1u64 << (nr * 8 + nf);
            }
        }
        table[sq] = bb;
    }
    table
}

fn init_king() -> [u64; 64] {
    let mut table = [0u64; 64];
    for sq in 0..64 {
        let r = (sq / 8) as i32;
        let f = (sq % 8) as i32;
        let mut bb = 0u64;
        for dr in -1..=1 {
            for df in -1..=1 {
                if dr == 0 && df == 0 {
                    continue;
                }
                let nr = r + dr;
                let nf = f + df;
                if (0..8).contains(&nr) && (0..8).contains(&nf) {
                    bb |= 1u64 << (nr * 8 + nf);
                }
            }
        }
        table[sq] = bb;
    }
    table
}

fn init_pawn() -> [[u64; 64]; 2] {
    let mut table = [[0u64; 64]; 2];
    for sq in 0..64 {
        let r = (sq / 8) as i32;
        let f = (sq % 8) as i32;
        let mut w = 0u64;
        let mut b = 0u64;
        for df in [-1, 1] {
            let nf = f + df;
            if (0..8).contains(&nf) {
                let wr = r + 1;
                if (0..8).contains(&wr) {
                    w |= 1u64 << (wr * 8 + nf);
                }
                let br = r - 1;
                if (0..8).contains(&br) {
                    b |= 1u64 << (br * 8 + nf);
                }
            }
        }
        table[0][sq] = w;
        table[1][sq] = b;
    }
    table
}

fn init_between() -> [[u64; 64]; 64] {
    let mut table = [[0u64; 64]; 64];
    for a in 0..64 {
        let ar = (a / 8) as i32;
        let af = (a % 8) as i32;
        for b in 0..64 {
            if a == b {
                continue;
            }
            let br = (b / 8) as i32;
            let bf = (b % 8) as i32;
            let same_rank = ar == br;
            let same_file = af == bf;
            let same_diag = (br - ar).abs() == (bf - af).abs();
            if !(same_rank || same_file || same_diag) {
                continue;
            }
            let dr = (br - ar).signum();
            let df = (bf - af).signum();
            let mut bb = 0u64;
            let mut r = ar + dr;
            let mut f = af + df;
            while r != br || f != bf {
                bb |= 1u64 << (r * 8 + f);
                r += dr;
                f += df;
            }
            table[a][b] = bb;
        }
    }
    table
}

fn init_line() -> [[u64; 64]; 64] {
    let mut table = [[0u64; 64]; 64];
    for a in 0..64 {
        let ar = (a / 8) as i32;
        let af = (a % 8) as i32;
        for b in 0..64 {
            if a == b {
                continue;
            }
            let br = (b / 8) as i32;
            let bf = (b % 8) as i32;
            let same_rank = ar == br;
            let same_file = af == bf;
            let same_diag = (br - ar).abs() == (bf - af).abs();
            if !(same_rank || same_file || same_diag) {
                continue;
            }
            let dr = (br - ar).signum();
            let df = (bf - af).signum();
            let mut bb = 1u64 << (ar * 8 + af);
            let mut r = ar + dr;
            let mut f = af + df;
            while (0..8).contains(&r) && (0..8).contains(&f) {
                bb |= 1u64 << (r * 8 + f);
                r += dr;
                f += df;
            }
            let mut r = ar - dr;
            let mut f = af - df;
            while (0..8).contains(&r) && (0..8).contains(&f) {
                bb |= 1u64 << (r * 8 + f);
                r -= dr;
                f -= df;
            }
            table[a][b] = bb;
        }
    }
    table
}

fn init_file_bb() -> [u64; 8] {
    let mut table = [0u64; 8];
    for f in 0..8 {
        let mut bb = 0u64;
        for r in 0..8 {
            bb |= 1u64 << (r * 8 + f);
        }
        table[f] = bb;
    }
    table
}

fn init_adjacent(files: &[u64; 8]) -> [u64; 8] {
    let mut table = [0u64; 8];
    for f in 0..8 {
        let mut bb = 0u64;
        if f > 0 {
            bb |= files[f - 1];
        }
        if f < 7 {
            bb |= files[f + 1];
        }
        table[f] = bb;
    }
    table
}

fn slider_mask(sq: usize, kind: Slider) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut bb = 0u64;
    let dirs: &[(i32, i32)] = match kind {
        Slider::Bishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
        Slider::Rook => &[(0, 1), (0, -1), (1, 0), (-1, 0)],
    };
    for &(dr, df) in dirs {
        let mut nr = r + dr;
        let mut nf = f + df;
        loop {
            let next_r = nr + dr;
            let next_f = nf + df;
            if !(0..8).contains(&next_r) || !(0..8).contains(&next_f) {
                break;
            }
            bb |= 1u64 << (nr * 8 + nf);
            nr = next_r;
            nf = next_f;
        }
    }
    bb
}

fn slider_attacks(sq: usize, occ: u64, kind: Slider) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut bb = 0u64;
    let dirs: &[(i32, i32)] = match kind {
        Slider::Bishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
        Slider::Rook => &[(0, 1), (0, -1), (1, 0), (-1, 0)],
    };
    for &(dr, df) in dirs {
        let mut nr = r + dr;
        let mut nf = f + df;
        while (0..8).contains(&nr) && (0..8).contains(&nf) {
            let bit = 1u64 << (nr * 8 + nf);
            bb |= bit;
            if (occ & bit) != 0 {
                break;
            }
            nr += dr;
            nf += df;
        }
    }
    bb
}

fn enumerate_subset(idx: usize, mask: u64) -> u64 {
    let mut subset = 0u64;
    let mut m = mask;
    let mut i = idx;
    while m != 0 {
        let lsb = m & m.wrapping_neg();
        if (i & 1) != 0 {
            subset |= lsb;
        }
        i >>= 1;
        m &= m - 1;
    }
    subset
}

struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

fn find_magic_for(sq: usize, kind: Slider, rng: &mut XorShift64) -> (u64, u64, Vec<u64>) {
    let mask = slider_mask(sq, kind);
    let nbits = mask.count_ones() as usize;
    let nvars = 1usize << nbits;

    let mut subsets = vec![0u64; nvars];
    let mut attacks = vec![0u64; nvars];
    for i in 0..nvars {
        subsets[i] = enumerate_subset(i, mask);
        attacks[i] = slider_attacks(sq, subsets[i], kind);
    }

    let shift = (64 - nbits) as u32;
    loop {
        let magic = rng.sparse();
        if ((mask.wrapping_mul(magic)) >> 56).count_ones() < 6 {
            continue;
        }
        let mut used = vec![u64::MAX; nvars];
        let mut ok = true;
        for i in 0..nvars {
            let idx = ((subsets[i].wrapping_mul(magic)) >> shift) as usize;
            if used[idx] == u64::MAX {
                used[idx] = attacks[i];
            } else if used[idx] != attacks[i] {
                ok = false;
                break;
            }
        }
        if ok {
            for slot in used.iter_mut() {
                if *slot == u64::MAX {
                    *slot = 0;
                }
            }
            return (mask, magic, used);
        }
    }
}

fn init_magic(
    kind: Slider,
) -> (
    [u64; 64],
    [u64; 64],
    [u32; 64],
    [usize; 64],
    Vec<u64>,
) {
    let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_BABE_u64);
    let mut masks = [0u64; 64];
    let mut magics = [0u64; 64];
    let mut shifts = [0u32; 64];
    let mut offsets = [0usize; 64];
    let mut combined: Vec<u64> = Vec::new();

    for sq in 0..64 {
        let (mask, magic, attacks) = find_magic_for(sq, kind, &mut rng);
        masks[sq] = mask;
        magics[sq] = magic;
        shifts[sq] = 64 - mask.count_ones();
        offsets[sq] = combined.len();
        combined.extend_from_slice(&attacks);
    }

    (masks, magics, shifts, offsets, combined)
}

#[inline(always)]
pub fn knight_attacks_u64(sq: usize) -> u64 {
    unsafe { *t().knight.get_unchecked(sq) }
}

#[inline(always)]
pub fn king_attacks_u64(sq: usize) -> u64 {
    unsafe { *t().king.get_unchecked(sq) }
}

#[inline(always)]
pub fn pawn_attacks_u64(color: Color, sq: usize) -> u64 {
    unsafe { *t().pawn.get_unchecked(color.to_index()).get_unchecked(sq) }
}

#[inline(always)]
pub fn between_u64(a: usize, b: usize) -> u64 {
    unsafe { *t().between.get_unchecked(a).get_unchecked(b) }
}

#[inline(always)]
pub fn line_u64(a: usize, b: usize) -> u64 {
    unsafe { *t().line.get_unchecked(a).get_unchecked(b) }
}

#[inline(always)]
pub fn bishop_attacks_u64(sq: usize, occ: u64) -> u64 {
    let tbl = t();
    unsafe {
        let mask = *tbl.bishop_masks.get_unchecked(sq);
        let magic = *tbl.bishop_magics.get_unchecked(sq);
        let shift = *tbl.bishop_shifts.get_unchecked(sq);
        let offset = *tbl.bishop_offsets.get_unchecked(sq);
        let mag_idx = ((occ & mask).wrapping_mul(magic)) >> shift;
        *tbl.bishop_attacks.get_unchecked(offset + mag_idx as usize)
    }
}

#[inline(always)]
pub fn rook_attacks_u64(sq: usize, occ: u64) -> u64 {
    let tbl = t();
    unsafe {
        let mask = *tbl.rook_masks.get_unchecked(sq);
        let magic = *tbl.rook_magics.get_unchecked(sq);
        let shift = *tbl.rook_shifts.get_unchecked(sq);
        let offset = *tbl.rook_offsets.get_unchecked(sq);
        let mag_idx = ((occ & mask).wrapping_mul(magic)) >> shift;
        *tbl.rook_attacks.get_unchecked(offset + mag_idx as usize)
    }
}

#[inline(always)]
pub fn get_knight_moves(sq: Square) -> BitBoard {
    BitBoard(knight_attacks_u64(sq.to_index()))
}

#[inline(always)]
pub fn get_king_moves(sq: Square) -> BitBoard {
    BitBoard(king_attacks_u64(sq.to_index()))
}

#[inline(always)]
pub fn get_bishop_moves(sq: Square, blockers: BitBoard) -> BitBoard {
    BitBoard(bishop_attacks_u64(sq.to_index(), blockers.0))
}

#[inline(always)]
pub fn get_rook_moves(sq: Square, blockers: BitBoard) -> BitBoard {
    BitBoard(rook_attacks_u64(sq.to_index(), blockers.0))
}

#[inline(always)]
pub fn get_file(file: File) -> BitBoard {
    BitBoard(unsafe { *t().file_bb.get_unchecked(file.to_index() as usize) })
}

#[inline(always)]
pub fn get_adjacent_files(file: File) -> BitBoard {
    BitBoard(unsafe { *t().adjacent_files.get_unchecked(file.to_index() as usize) })
}
