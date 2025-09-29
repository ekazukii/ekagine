use chess::{BitBoard, Board, ChessMove, MoveGen, EMPTY};
use smallvec::SmallVec;
use crate::{PVTable, TranspositionTable};
use crate::movegen::MoveGenState::{Captures, PrincipalVariation, Remaining, TTStep};

pub const MVV_LVA_TABLE: [[u8; 6]; 6] = [
    // Victim = Pawn
    [15, 14, 13, 12, 11, 10],
    // Victim = Knight
    [25, 24, 23, 22, 21, 20],
    // Victim = Bishop
    [35, 34, 33, 32, 31, 30],
    // Victim = Rook
    [45, 44, 43, 42, 41, 40],
    // Victim = Queen
    [55, 54, 53, 52, 51, 50],
    // Victim = King (we never actually capture kings in legal move generation)
    [0, 0, 0, 0, 0, 0],
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum MoveGenState {
    PrincipalVariation,
    TTStep,
    Captures,
    Remaining
}

pub struct IncrementalMoveGen<'a> {
    iterator: Option<MoveGen>,
    state: MoveGenState,
    board: &'a Board,
    pv_move: Option<ChessMove>,
    tt_move: Option<ChessMove>,
    move_buff: SmallVec<[ChessMove; 64]>,
    capt_idx: Option<usize>,
    capture_gen_pending: bool,
}

impl<'a> IncrementalMoveGen<'a> {
    pub fn new(board: &'a Board, pv_table: &PVTable, tt: &TranspositionTable) -> Self {
        let zob = board.get_hash();
        Self {
            iterator: None,
            state: PrincipalVariation,
            board,
            pv_move: pv_table.get(&zob).copied(),
            tt_move: tt.probe(zob).and_then(|e| e.best_move),
            move_buff: SmallVec::new(),
            capt_idx: None,
            capture_gen_pending: false,
        }
    }

    pub fn take_capture_generation_event(&mut self) -> bool {
        if self.capture_gen_pending {
            self.capture_gen_pending = false;
            true
        } else {
            false
        }
    }
}

impl Iterator for IncrementalMoveGen<'_> {
    type Item = ChessMove;
    fn next(&mut self) -> Option<Self::Item> {
        if self.state == PrincipalVariation  {
            if self.pv_move.is_some() {
                self.state = TTStep;
                return self.pv_move;
            }
            self.state = TTStep;
        }

        if self.state == TTStep {
            if self.tt_move.is_some() && self.tt_move != self.pv_move {
                self.state = Captures;
                return self.tt_move;
            }
            self.state = Captures;
        }

        let mut iter = self.iterator.get_or_insert_with(|| MoveGen::new_legal(&self.board));

        if self.state == Captures {
            if self.capt_idx.is_none() {
                // Sort for captures
                let targets = self.board.color_combined(!self.board.side_to_move());
                iter.set_iterator_mask(*targets);

                let mut scored_moves: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();
                for mv in  &mut iter {
                    if Some(mv) == self.tt_move || Some(mv) == self.pv_move {
                        continue
                    }

                    let attacker_idx = self.board.piece_on(mv.get_source()).unwrap().to_index();
                    let score = if let Some(victim) = self.board.piece_on(mv.get_dest()) {
                        MVV_LVA_TABLE[victim.to_index()][attacker_idx] as i32
                    } else {
                        // En passant
                        25i32
                    };
                    scored_moves.push((mv, score));

                }
                scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                self.move_buff.extend(scored_moves.into_iter().map(|(m, _)| m));
                self.capt_idx = Some(0);
                self.capture_gen_pending = true;
            }

            if (self.capt_idx.unwrap()) < self.move_buff.len() {
                let mv = self.move_buff[self.capt_idx.unwrap()];
                self.capt_idx = Some(self.capt_idx.unwrap() + 1);
                return Some(mv);
            } else {
                self.state = Remaining;
                iter.set_iterator_mask(!EMPTY);
            }
        }


        loop {
            let mv = iter.next();
            if (self.pv_move.is_some() && mv == self.pv_move) || (self.tt_move.is_some() && mv == self.tt_move) {
                continue; // skip duplicates
            }
            return mv;
        }

    }
}
