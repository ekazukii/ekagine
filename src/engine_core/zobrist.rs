use super::types::{Color, Piece, Square};

/// Compile-time Zobrist key table (deterministic, generated via SplitMix64).
struct Keys {
    piece_square: [[[u64; 64]; 6]; 2],
    side_to_move: u64,
    castling: [u64; 16],
    en_passant_file: [u64; 8],
}

const KEYS: Keys = build_keys();

const fn build_keys() -> Keys {
    let mut state: u64 = 0x9D39_247E_3377_6D41;
    let mut piece_square = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut s = 0;
            while s < 64 {
                let (next, ns) = splitmix64(state);
                state = ns;
                piece_square[c][p][s] = next;
                s += 1;
            }
            p += 1;
        }
        c += 1;
    }
    let (side_to_move, ns) = splitmix64(state);
    state = ns;
    let mut castling = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        let (n, ns2) = splitmix64(state);
        state = ns2;
        castling[i] = n;
        i += 1;
    }
    let mut en_passant_file = [0u64; 8];
    let mut i = 0;
    while i < 8 {
        let (n, ns2) = splitmix64(state);
        state = ns2;
        en_passant_file[i] = n;
        i += 1;
    }
    Keys {
        piece_square,
        side_to_move,
        castling,
        en_passant_file,
    }
}

const fn splitmix64(state: u64) -> (u64, u64) {
    let new_state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = new_state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let value = z ^ (z >> 31);
    (value, new_state)
}

#[inline(always)]
pub fn key_piece(color: Color, piece: Piece, sq: Square) -> u64 {
    KEYS.piece_square[color.to_index()][piece.to_index() as usize][sq.to_index()]
}

#[inline(always)]
pub fn key_side() -> u64 {
    KEYS.side_to_move
}

#[inline(always)]
pub fn key_castling(rights: u8) -> u64 {
    KEYS.castling[(rights & 0x0F) as usize]
}

#[inline(always)]
pub fn key_ep_file(file: u8) -> u64 {
    KEYS.en_passant_file[(file & 0x07) as usize]
}
