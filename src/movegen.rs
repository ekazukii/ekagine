use crate::search::see_for_sort;
use crate::TranspositionTable;
use chess::{BitBoard, Board, ChessMove, MoveGen, EMPTY};
use smallvec::SmallVec;

type MoveList = SmallVec<[ChessMove; 64]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveGenPhase {
    TTMove,
    GoodCaptures,
    GoodQuiet,
    BadCaptures,
    BadQuiet,
    Done,
}

pub struct IncrementalMoveGen<'a> {
    board: &'a Board,
    tt_move: Option<ChessMove>,
    killer_moves: [Option<ChessMove>; 2],
    move_gen_iter: MoveGen,
    phase: MoveGenPhase,
    has_legal_moves: bool,
    capture_gen_pending: bool,
    captures_generated: bool,
    quiet_generated: bool,
    good_captures: MoveList,
    bad_captures: MoveList,
    killer_quiet: MoveList,
    good_quiet: MoveList,
    bad_quiet: MoveList,
    good_capture_idx: usize,
    bad_capture_idx: usize,
    killer_quiet_idx: usize,
    good_quiet_idx: usize,
    bad_quiet_idx: usize,
}

impl<'a> IncrementalMoveGen<'a> {
    pub fn new(
        board: &'a Board,
        tt: &TranspositionTable,
        killer_moves: [Option<ChessMove>; 2],
    ) -> Self {
        let probe_iter = MoveGen::new_legal(board);
        let has_legal_moves = probe_iter.len() > 0;

        let zob = board.get_hash();
        let tt_move = tt
            .probe(zob)
            .and_then(|e| e.best_move)
            .filter(|mv| board.legal(*mv));

        Self {
            board,
            tt_move,
            killer_moves,
            phase: MoveGenPhase::TTMove,
            has_legal_moves,
            capture_gen_pending: false,
            captures_generated: false,
            quiet_generated: false,
            move_gen_iter: probe_iter,
            good_captures: MoveList::new(),
            bad_captures: MoveList::new(),
            killer_quiet: MoveList::new(),
            good_quiet: MoveList::new(),
            bad_quiet: MoveList::new(),
            good_capture_idx: 0,
            bad_capture_idx: 0,
            killer_quiet_idx: 0,
            good_quiet_idx: 0,
            bad_quiet_idx: 0,
        }
    }

    fn ensure_captures(&mut self) {
        if self.captures_generated {
            return;
        }
        self.captures_generated = true;
        self.capture_gen_pending = true;

        let mut capture_mask = *self.board.color_combined(!self.board.side_to_move());
        if let Some(ep_sq) = self.board.en_passant() {
            capture_mask |= BitBoard::from_square(ep_sq);
        }

        self.move_gen_iter.set_iterator_mask(capture_mask);

        let mut good_scored: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();
        let mut bad_scored: SmallVec<[(ChessMove, i32); 32]> = SmallVec::new();

        for mv in self.move_gen_iter.by_ref() {
            if Some(mv) == self.tt_move {
                continue;
            }

            let mut score = see_for_sort(self.board, mv);
            if score >= 0 {
                score += 1;
                good_scored.push((mv, score));
            } else {
                bad_scored.push((mv, score));
            }
        }

        good_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        bad_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        self.good_captures
            .extend(good_scored.into_iter().map(|(mv, _)| mv));
        self.bad_captures
            .extend(bad_scored.into_iter().map(|(mv, _)| mv));
    }

    fn ensure_quiet(&mut self) {
        if self.quiet_generated {
            return;
        }
        self.quiet_generated = true;

        let mut killer_hits: MoveList = MoveList::new();
        let mut good_quiet: MoveList = MoveList::new();
        let mut bad_quiet: MoveList = MoveList::new();
        let mut killer_seen = [false; 2];

        self.move_gen_iter.set_iterator_mask(!EMPTY);
        for mv in self.move_gen_iter.by_ref() {
            if Some(mv) == self.tt_move {
                continue;
            }
            if self.board.piece_on(mv.get_dest()).is_some() {
                continue;
            }

            let mut matched_killer = false;
            for (idx, killer) in self.killer_moves.iter().enumerate() {
                if killer_seen[idx] {
                    continue;
                }
                if *killer == Some(mv) {
                    killer_seen[idx] = true;
                    killer_hits.push(mv);
                    matched_killer = true;
                    break;
                }
            }
            if matched_killer {
                continue;
            }

            if mv.get_promotion().is_some() {
                good_quiet.push(mv);
            } else {
                bad_quiet.push(mv);
            }
        }

        self.killer_quiet = killer_hits;
        self.good_quiet = good_quiet;
        self.bad_quiet = bad_quiet;
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
        !self.has_legal_moves
    }
}

impl Iterator for IncrementalMoveGen<'_> {
    type Item = ChessMove;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.phase {
                MoveGenPhase::TTMove => {
                    self.phase = MoveGenPhase::GoodCaptures;
                    if let Some(tt_mv) = self.tt_move {
                        return Some(tt_mv);
                    }
                }
                MoveGenPhase::GoodCaptures => {
                    self.ensure_captures();
                    if self.good_capture_idx < self.good_captures.len() {
                        let mv = self.good_captures[self.good_capture_idx];
                        self.good_capture_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::GoodQuiet;
                }
                MoveGenPhase::GoodQuiet => {
                    self.ensure_quiet();
                    if self.killer_quiet_idx < self.killer_quiet.len() {
                        let mv = self.killer_quiet[self.killer_quiet_idx];
                        self.killer_quiet_idx += 1;
                        return Some(mv);
                    }
                    if self.good_quiet_idx < self.good_quiet.len() {
                        let mv = self.good_quiet[self.good_quiet_idx];
                        self.good_quiet_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::BadCaptures;
                }
                MoveGenPhase::BadCaptures => {
                    self.ensure_captures();
                    if self.bad_capture_idx < self.bad_captures.len() {
                        let mv = self.bad_captures[self.bad_capture_idx];
                        self.bad_capture_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::BadQuiet;
                }
                MoveGenPhase::BadQuiet => {
                    self.ensure_quiet();
                    if self.bad_quiet_idx < self.bad_quiet.len() {
                        let mv = self.bad_quiet[self.bad_quiet_idx];
                        self.bad_quiet_idx += 1;
                        return Some(mv);
                    }
                    self.phase = MoveGenPhase::Done;
                }
                MoveGenPhase::Done => return None,
            }
        }
    }
}
