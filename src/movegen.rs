use crate::movegen::MoveGenState::{PrincipalVariation, Remaining, Sorted, TTStep};
use crate::{PVTable, TranspositionTable};
use chess::{Board, ChessMove, MoveGen, EMPTY};
use smallvec::SmallVec;
use crate::search::see_for_sort;

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
    Sorted,
    Remaining,
}

pub struct IncrementalMoveGen<'a> {
    iterator: MoveGen,
    state: MoveGenState,
    board: &'a Board,
    pv_move: Option<ChessMove>,
    tt_move: Option<ChessMove>,
    killer_moves: [Option<ChessMove>; 2],
    move_buff: SmallVec<[ChessMove; 64]>,
    capt_idx: Option<usize>,
    capture_gen_pending: bool,
    killer_idx: usize,
}

impl<'a> IncrementalMoveGen<'a> {
    pub fn new(
        board: &'a Board,
        pv_table: &PVTable,
        tt: &TranspositionTable,
        killer_moves: [Option<ChessMove>; 2],
    ) -> Self {
        let zob = board.get_hash();
        Self {
            iterator: MoveGen::new_legal(&board),
            state: PrincipalVariation,
            board,
            pv_move: pv_table.get(&zob).copied(),
            tt_move: tt.probe(zob).and_then(|e| e.best_move),
            killer_moves,
            move_buff: SmallVec::new(),
            capt_idx: None,
            capture_gen_pending: false,
            killer_idx: 0,
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

    // Game's over if player cannot play a single move
    pub fn is_over(&self) -> bool {
        self.iterator.len() == 0
    }
}

impl Iterator for IncrementalMoveGen<'_> {
    type Item = ChessMove;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state == PrincipalVariation {
            if self.pv_move.is_some() {
                self.state = TTStep;
                return self.pv_move;
            }
            self.state = TTStep;
        }

        if self.state == TTStep {
            if self.tt_move.is_some() && self.tt_move != self.pv_move {
                self.state = Sorted;
                return self.tt_move;
            }
            self.state = Sorted;
        }

        if self.state == Sorted {
            if self.capt_idx.is_none() {
                // Sort for captures
                let targets = self.board.color_combined(!self.board.side_to_move());
                self.iterator.set_iterator_mask(*targets);

                let mut scored_moves: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();
                for mv in &mut self.iterator {
                    if Some(mv) == self.tt_move || Some(mv) == self.pv_move {
                        continue;
                    }

                    let mut score = see_for_sort(self.board, mv);
                    if score >= 0 {
                        score += 1;
                    }
                    scored_moves.push((mv, score));
                }


                while self.killer_idx < self.killer_moves.len() {
                    let killer = self.killer_moves[self.killer_idx];
                    self.killer_idx += 1;
                    if let Some(kmv) = killer {
                        if Some(kmv) == self.pv_move || Some(kmv) == self.tt_move {
                            continue;
                        }
                        if self.board.piece_on(kmv.get_dest()).is_some() {
                            continue;
                        }
                        if self.board.legal(kmv) {
                            scored_moves.push((kmv, 0));
                        }
                    }
                }

                scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                self.move_buff
                    .extend(scored_moves.into_iter().map(|(m, _)| m));
                self.capt_idx = Some(0);
                self.capture_gen_pending = true;
            }



            if (self.capt_idx.unwrap()) < self.move_buff.len() {
                let mv = self.move_buff[self.capt_idx.unwrap()];
                self.capt_idx = Some(self.capt_idx.unwrap() + 1);
                return Some(mv);
            } else {
                self.state = Remaining;
                self.iterator.set_iterator_mask(!EMPTY);
            }
        }

        loop {
            let mv = self.iterator.next();
            if (self.pv_move.is_some() && mv == self.pv_move)
                || (self.tt_move.is_some() && mv == self.tt_move)
            {
                continue; // skip duplicates
            }
            return mv;
        }
    }
}
