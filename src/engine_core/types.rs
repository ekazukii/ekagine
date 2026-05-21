use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const COUNT: usize = 2;

    #[inline(always)]
    pub const fn to_index(self) -> usize {
        self as usize
    }
}

impl Not for Color {
    type Output = Color;
    #[inline(always)]
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    pub const COUNT: usize = 6;

    #[inline(always)]
    pub const fn to_index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum File {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
}

impl File {
    pub const COUNT: usize = 8;

    #[inline]
    pub fn from_index(i: usize) -> File {
        match i & 7 {
            0 => File::A,
            1 => File::B,
            2 => File::C,
            3 => File::D,
            4 => File::E,
            5 => File::F,
            6 => File::G,
            _ => File::H,
        }
    }

    #[inline(always)]
    pub const fn to_index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Rank {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
    Fifth = 4,
    Sixth = 5,
    Seventh = 6,
    Eighth = 7,
}

impl Rank {
    pub const COUNT: usize = 8;

    #[inline]
    pub fn from_index(i: usize) -> Rank {
        match i & 7 {
            0 => Rank::First,
            1 => Rank::Second,
            2 => Rank::Third,
            3 => Rank::Fourth,
            4 => Rank::Fifth,
            5 => Rank::Sixth,
            6 => Rank::Seventh,
            _ => Rank::Eighth,
        }
    }

    #[inline(always)]
    pub const fn to_index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square(pub(crate) u8);

impl Square {
    #[inline(always)]
    pub const fn new(idx: u8) -> Square {
        Square(idx & 63)
    }

    #[inline(always)]
    pub fn make_square(rank: Rank, file: File) -> Square {
        Square((rank as u8) * 8 + (file as u8))
    }

    #[inline(always)]
    pub const fn to_index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn get_rank(self) -> Rank {
        Rank::from_index((self.0 / 8) as usize)
    }

    #[inline]
    pub fn get_file(self) -> File {
        File::from_index((self.0 % 8) as usize)
    }

    #[inline]
    pub fn up(self) -> Option<Square> {
        if self.0 < 56 {
            Some(Square(self.0 + 8))
        } else {
            None
        }
    }

    #[inline]
    pub fn down(self) -> Option<Square> {
        if self.0 >= 8 {
            Some(Square(self.0 - 8))
        } else {
            None
        }
    }

    #[inline]
    pub fn left(self) -> Option<Square> {
        if self.0 % 8 != 0 {
            Some(Square(self.0 - 1))
        } else {
            None
        }
    }

    #[inline]
    pub fn right(self) -> Option<Square> {
        if self.0 % 8 != 7 {
            Some(Square(self.0 + 1))
        } else {
            None
        }
    }

    #[inline]
    pub fn backward(self, color: Color) -> Option<Square> {
        match color {
            Color::White => self.down(),
            Color::Black => self.up(),
        }
    }

    #[inline]
    pub fn ubackward(self, color: Color) -> Square {
        match color {
            Color::White => Square(self.0.wrapping_sub(8)),
            Color::Black => Square(self.0.wrapping_add(8)),
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + (self.0 % 8)) as char;
        let rank = (b'1' + (self.0 / 8)) as char;
        write!(f, "{}{}", file, rank)
    }
}

impl Square {
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A2: Square = Square(8);
    pub const B2: Square = Square(9);
    pub const C2: Square = Square(10);
    pub const D2: Square = Square(11);
    pub const E2: Square = Square(12);
    pub const F2: Square = Square(13);
    pub const G2: Square = Square(14);
    pub const H2: Square = Square(15);
    pub const A3: Square = Square(16);
    pub const B3: Square = Square(17);
    pub const C3: Square = Square(18);
    pub const D3: Square = Square(19);
    pub const E3: Square = Square(20);
    pub const F3: Square = Square(21);
    pub const G3: Square = Square(22);
    pub const H3: Square = Square(23);
    pub const A4: Square = Square(24);
    pub const B4: Square = Square(25);
    pub const C4: Square = Square(26);
    pub const D4: Square = Square(27);
    pub const E4: Square = Square(28);
    pub const F4: Square = Square(29);
    pub const G4: Square = Square(30);
    pub const H4: Square = Square(31);
    pub const A5: Square = Square(32);
    pub const B5: Square = Square(33);
    pub const C5: Square = Square(34);
    pub const D5: Square = Square(35);
    pub const E5: Square = Square(36);
    pub const F5: Square = Square(37);
    pub const G5: Square = Square(38);
    pub const H5: Square = Square(39);
    pub const A6: Square = Square(40);
    pub const B6: Square = Square(41);
    pub const C6: Square = Square(42);
    pub const D6: Square = Square(43);
    pub const E6: Square = Square(44);
    pub const F6: Square = Square(45);
    pub const G6: Square = Square(46);
    pub const H6: Square = Square(47);
    pub const A7: Square = Square(48);
    pub const B7: Square = Square(49);
    pub const C7: Square = Square(50);
    pub const D7: Square = Square(51);
    pub const E7: Square = Square(52);
    pub const F7: Square = Square(53);
    pub const G7: Square = Square(54);
    pub const H7: Square = Square(55);
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BitBoard(pub u64);

pub const EMPTY: BitBoard = BitBoard(0);

impl BitBoard {
    #[inline(always)]
    pub const fn new(value: u64) -> Self {
        BitBoard(value)
    }

    #[inline(always)]
    pub fn from_square(sq: Square) -> Self {
        BitBoard(1u64 << sq.to_index())
    }

    #[inline(always)]
    pub fn set(rank: Rank, file: File) -> Self {
        BitBoard::from_square(Square::make_square(rank, file))
    }

    #[inline(always)]
    pub const fn popcnt(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub fn to_square(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }
}

impl Iterator for BitBoard {
    type Item = Square;
    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let lsb = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            Some(Square(lsb as u8))
        }
    }
}

macro_rules! impl_bitop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for BitBoard {
            type Output = BitBoard;
            #[inline(always)]
            fn $method(self, rhs: BitBoard) -> BitBoard { BitBoard(self.0 $op rhs.0) }
        }
        impl $trait<&BitBoard> for BitBoard {
            type Output = BitBoard;
            #[inline(always)]
            fn $method(self, rhs: &BitBoard) -> BitBoard { BitBoard(self.0 $op rhs.0) }
        }
        impl $trait<BitBoard> for &BitBoard {
            type Output = BitBoard;
            #[inline(always)]
            fn $method(self, rhs: BitBoard) -> BitBoard { BitBoard(self.0 $op rhs.0) }
        }
        impl $trait<&BitBoard> for &BitBoard {
            type Output = BitBoard;
            #[inline(always)]
            fn $method(self, rhs: &BitBoard) -> BitBoard { BitBoard(self.0 $op rhs.0) }
        }
    };
}

impl_bitop!(BitAnd, bitand, &);
impl_bitop!(BitOr, bitor, |);
impl_bitop!(BitXor, bitxor, ^);

impl BitAndAssign for BitBoard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: BitBoard) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for BitBoard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: BitBoard) {
        self.0 |= rhs.0;
    }
}

impl BitXorAssign for BitBoard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: BitBoard) {
        self.0 ^= rhs.0;
    }
}

impl Not for BitBoard {
    type Output = BitBoard;
    #[inline(always)]
    fn not(self) -> BitBoard {
        BitBoard(!self.0)
    }
}

impl Not for &BitBoard {
    type Output = BitBoard;
    #[inline(always)]
    fn not(self) -> BitBoard {
        BitBoard(!self.0)
    }
}

/// 16-bit packed move: bits 0..6 = from, 6..12 = to, 12..16 = promotion code
/// (0=None, 1=Knight, 2=Bishop, 3=Rook, 4=Queen).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChessMove(u16);

impl ChessMove {
    #[inline(always)]
    pub const fn new(source: Square, dest: Square, promotion: Option<Piece>) -> Self {
        let from = (source.0 as u16) & 0x3F;
        let to = (dest.0 as u16) & 0x3F;
        let promo = match promotion {
            None => 0u16,
            Some(Piece::Knight) => 1,
            Some(Piece::Bishop) => 2,
            Some(Piece::Rook) => 3,
            Some(Piece::Queen) => 4,
            // Pawn/King in promotion field are illegal; treat as None.
            _ => 0,
        };
        ChessMove(from | (to << 6) | (promo << 12))
    }

    #[inline(always)]
    pub const fn get_source(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn get_dest(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn get_promotion(self) -> Option<Piece> {
        match (self.0 >> 12) & 0xF {
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            _ => None,
        }
    }

    pub fn from_san<B>(_board: &B, _s: &str) -> Result<Self, &'static str> {
        Err("SAN parsing not implemented; fall through to from_str")
    }
}

impl FromStr for ChessMove {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = s.as_bytes();
        if bytes.len() < 4 || bytes.len() > 5 {
            return Err("move string length");
        }
        let from_file = bytes[0].wrapping_sub(b'a');
        let from_rank = bytes[1].wrapping_sub(b'1');
        let to_file = bytes[2].wrapping_sub(b'a');
        let to_rank = bytes[3].wrapping_sub(b'1');
        if from_file >= 8 || from_rank >= 8 || to_file >= 8 || to_rank >= 8 {
            return Err("move squares out of range");
        }
        let from = Square::make_square(
            Rank::from_index(from_rank as usize),
            File::from_index(from_file as usize),
        );
        let to = Square::make_square(
            Rank::from_index(to_rank as usize),
            File::from_index(to_file as usize),
        );
        let promotion = if bytes.len() == 5 {
            match bytes[4] {
                b'q' | b'Q' => Some(Piece::Queen),
                b'r' | b'R' => Some(Piece::Rook),
                b'b' | b'B' => Some(Piece::Bishop),
                b'n' | b'N' => Some(Piece::Knight),
                _ => return Err("invalid promotion char"),
            }
        } else {
            None
        };
        Ok(ChessMove::new(from, to, promotion))
    }
}

impl fmt::Display for ChessMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let from = self.get_source().0;
        let to = self.get_dest().0;
        write!(
            f,
            "{}{}{}{}",
            (b'a' + (from % 8)) as char,
            (b'1' + (from / 8)) as char,
            (b'a' + (to % 8)) as char,
            (b'1' + (to / 8)) as char,
        )?;
        if let Some(p) = self.get_promotion() {
            let c = match p {
                Piece::Knight => 'n',
                Piece::Bishop => 'b',
                Piece::Rook => 'r',
                Piece::Queen => 'q',
                _ => return Err(fmt::Error),
            };
            write!(f, "{}", c)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardStatus {
    Ongoing,
    Checkmate,
    Stalemate,
}
