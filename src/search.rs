use std::collections::HashMap;
use std::io::Stdout;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{io, thread};
use std::time::{Duration, Instant};
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};
use crate::{board_do_move, board_pop, send_message, PVTable, RepetitionTable, StopFlag, TranspositionTable, CACHE_COUNT, DEPTH_COUNT, EVAL_COUNT, NEG_INFINITY, POS_INFINITY, QUIESCE_REMAIN};
use crate::eval::eval_board;

/* For reference
type TranspositionTable = HashMap<u64, i32>;
type RepetitionTable    = HashMap<u64, usize>;
type PVTable = HashMap<u64, ChessMove>;
 */


enum SearchScore {
    CANCELLED,
    //MATE(u8),
    EVAL(i32)
}

#[inline(always)]
fn should_stop(flag: &StopFlag) -> bool {
    flag.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Quiescence search (captures‑only) — negamax style
// ---------------------------------------------------------------------------
fn quiesce_negamax_it(
    board: &Board,
    mut alpha: i32,
    mut beta: i32,
    remain_quiet: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    color: i32, // +1 if White to move in this node, −1 otherwise
) -> i32 {
    match board.status() {
        BoardStatus::Checkmate => return NEG_INFINITY,
        BoardStatus::Stalemate => return 0,
        BoardStatus::Ongoing => {}
    }

    if remain_quiet == 0 || is_in_threefold_scenario(board, repetition_table) {
        return color * cache_eval(board, transpo_table, repetition_table);
    }

    let stand_pat = color * cache_eval(board, transpo_table, repetition_table);
    if stand_pat >= beta { return stand_pat; }
    if stand_pat > alpha { alpha = stand_pat; }

    for mv in sort_moves(board) {
        if board.piece_on(mv.get_dest()).is_none() { continue; }
        let new_board = board_do_move(board, mv, repetition_table);
        let score = -quiesce_negamax_it(
            &new_board,
            -beta,
            -alpha,
            remain_quiet - 1,
            transpo_table,
            repetition_table,
            -color,
        );
        board_pop(&new_board, repetition_table);

        if score >= beta { return score; }
        if score > alpha { alpha = score; }
    }

    alpha
}


/// Check if this position's hash has occurred more than once already.
/// (i.e. `> 1` means third occurrence => threefold draw).
///
///
fn is_in_threefold_scenario(board: &Board, repetition_table: &RepetitionTable) -> bool {
    let zob = board.get_hash();
    repetition_table.get(&zob).cloned().unwrap_or(0) > 1
}

/// Cached evaluation: if threefold, return 0, else look up in transposition table.
/// If not found, compute via `eval_board`, insert into the table, and return.
///
///
fn cache_eval(
    board: &Board,
    transpo_table: &mut TranspositionTable,
    repetition_table: &RepetitionTable,
) -> i32 {
    if is_in_threefold_scenario(board, repetition_table) {
        return 0;
    }
    let zob = board.get_hash();
    if let Some(&cached) = transpo_table.get(&zob) {
        CACHE_COUNT.fetch_add(1, Ordering::Relaxed);
        cached
    } else {
        let val = eval_board(board);
        transpo_table.insert(zob, val);
        val
    }
}

const CHECK_BONUS: u8 = 60;
const CHECKMATE_BONUS: u8 = 100;
const MVV_LVA_TABLE: [[u8; 6]; 6] = [
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
    [ 0,  0,  0,  0,  0,  0],
];

fn sort_moves(board: &Board) -> Vec<ChessMove> {
    let mut scored_moves: Vec<(ChessMove, u8)> = Vec::new();

    for mv in MoveGen::new_legal(board) {
        let score = if let Some(victim_piece) = board.piece_on(mv.get_dest()) {
            // It's a capture; find the attacker piece type
            if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
                let victim_idx = victim_piece.to_index() as usize;
                let attacker_idx = attacker_piece.to_index() as usize;
                MVV_LVA_TABLE[victim_idx][attacker_idx]
            } else {
                0
            }
        } else {
            0
        };

        let board_after = board.make_move_new(mv);

        let check_bonus = if board_after.checkers().popcnt() > 0 {
            // match board_after.status() {
            //    BoardStatus::Checkmate => {
            //        CHECKMATE_BONUS
            //    }
            //    _ => CHECK_BONUS,
            //}
            CHECK_BONUS
        } else {
            0
        };


        scored_moves.push((mv, score + check_bonus));

        //scored_moves.push((mv, score));
    }

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}


// ---------------------------------------------------------------------------
// Move‑ordering helper:
//   • Copies the usual MVV‑LVA ordering from `sort_moves`.
//   • Promote the principal variation
// ---------------------------------------------------------------------------
fn ordered_moves(board: &Board, pv_table: &PVTable, depth: usize) -> Vec<ChessMove> {
    let mut moves = sort_moves(board);
    let mut ordered: Vec<ChessMove> = Vec::with_capacity(moves.len());

    // First, add PV move if any
    if let Some(pvs) = pv_table.get(&board.get_hash()) {
        ordered.push(pvs.clone());
    }

    // Add remaining moves
    ordered.extend(moves);
    ordered
}

// ---------------------------------------------------------------------------
// Core Negamax with α/β pruning + PV maintenance — renamed `negamax_it`.
// ---------------------------------------------------------------------------
fn negamax_it(
    board: Board,
    depth: usize,
    max_depth: usize,
    mut alpha: i32,
    mut beta: i32,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    color: i32,
    stop: &StopFlag,
) -> SearchScore {
    if should_stop(stop) {
        let mut out = io::stdout();
        let info_line = format!("info string break stop");
        send_message(&mut out, &info_line);
        return SearchScore::CANCELLED;
    }

    if is_in_threefold_scenario(&board, repetition_table) { return SearchScore::EVAL(0); }

    match board.status() {
        BoardStatus::Checkmate => return SearchScore::EVAL(NEG_INFINITY),
        BoardStatus::Stalemate => return SearchScore::EVAL(0),
        BoardStatus::Ongoing => {}
    }

    // Leaf depth → quiescence
    if depth == max_depth {
        return SearchScore::EVAL(quiesce_negamax_it(
            &board,
            alpha,
            beta,
            QUIESCE_REMAIN,
            transpo_table,
            repetition_table,
            color,
        ));
    }

    let mut best_value = NEG_INFINITY;
    let mut best_move_opt: Option<ChessMove> = None;

    for mv in ordered_moves(&board, pv_table, depth) {
        let new_board = board_do_move(&board, mv, repetition_table);
        match negamax_it(
            new_board,
            depth + 1,
            max_depth,
            -beta,
            -alpha,
            transpo_table,
            repetition_table,
            pv_table,
            -color,
            stop
        ) {
            SearchScore::CANCELLED => {
                return SearchScore::CANCELLED
            },
            SearchScore::EVAL(v) => {
                let value = -v;

                board_pop(&new_board, repetition_table);

                if value > best_value {
                    best_value = value;
                    best_move_opt = Some(mv);
                }
                if value > alpha { alpha = value; }
                if alpha >= beta {
                    break;
                }
            }
        }
    }


    if best_move_opt.is_some() {
        pv_table.insert(board.get_hash(), best_move_opt.unwrap());
    } else {
        return SearchScore::CANCELLED;
    }

    SearchScore::EVAL(best_value)
}

// ---------------------------------------------------------------------------
// Depth‑limited root search (uses `negamax_it`)
// ---------------------------------------------------------------------------
fn root_search(
    board: &Board,
    max_depth: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
    pv_table: &mut PVTable,
    stop: &StopFlag,
) -> (Option<ChessMove>, i32) {
    let mut alpha = NEG_INFINITY;
    let mut beta = POS_INFINITY;
    let mut best_value = NEG_INFINITY;
    let mut best_move = None;
    let color = if board.side_to_move() == Color::White { 1 } else { -1 };

    let mut out = io::stdout();

    for mv in ordered_moves(board, pv_table, 0) {
        if should_stop(stop) { break; }

        if best_move.is_some() {
            let bm: ChessMove = best_move.unwrap();
            let info_line = format!(
                "info string [{}] evaluating move : {}, curr best is {:?} ({})",
                max_depth, mv.to_string(), bm.to_string(), best_value
            );
            send_message(&mut out, &info_line);
        } else {
            let info_line = format!(
                "info string [{}] evaluating move : {}",
                max_depth, mv.to_string()
            );
            send_message(&mut out, &info_line);
        }

        let new_board = board_do_move(board, mv, repetition_table);
        match negamax_it(
            new_board,
            1,
            max_depth,
            -beta,
            -alpha,
            transpo_table,
            repetition_table,
            pv_table,
            -color,
            stop
        ) {
            SearchScore::CANCELLED => {
                break;
            },
            SearchScore::EVAL(v) => {
                let value = -v;
                board_pop(&new_board, repetition_table);

                if value > best_value {
                    best_value = value;
                    best_move = Some(mv);

                    let bm: ChessMove = best_move.unwrap();
                    let info_line = format!(
                        "info string [{}] replacing best move with {}, {} > {}",
                        max_depth, mv.to_string(), value, best_value
                    );
                    send_message(&mut out, &info_line);
                }
                if value > alpha { alpha = value; }
                if alpha >= beta { break; }
            }
        }
    }

    if best_move.is_some() {
        pv_table.insert(board.get_hash(), best_move.unwrap());
    }

    //(best_move, best_value * color)
    (best_move, best_value)
}

// ---------------------------------------------------------------------------
// Public interface: iterative deepening — renamed `best_move_using_iterative_deepening`.
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn best_move_using_iterative_deepening(
    board: &Board,
    max_depth: usize,
    transpo_table: &mut TranspositionTable,
    repetition_table: &mut RepetitionTable,
) -> (Option<ChessMove>, i32) {
    let mut pv_table: PVTable = HashMap::new();
    let mut final_best_move = None;
    let mut final_best_score = 0;
    let stop      = Arc::new(AtomicBool::new(false));

    for depth in 1..=max_depth {
        let (bm, score) = root_search(
            board,
            depth,
            transpo_table,
            repetition_table,
            &mut pv_table,
            &stop,
        );
        if bm.is_some() {
            final_best_move = bm;
            final_best_score = score;
        }
    }

    (final_best_move, final_best_score)
}

/// Iterative-deepening search that can be stopped by a time budget.
/// If `stdout_opt` is `Some(&mut Stdout)` the routine prints Stockfish-
/// compatible "info ..." lines through your existing `send_message`.
///
/// * `board`          – starting position
/// * `time_budget`    – wall-clock budget (hard stop)
/// * `max_depth_cap`  – fail-safe depth limit (e.g. 99)
/// * `stdout_opt`     – `Some(&mut std::io::Stdout)` to enable printing /
///                      logging, or `None` to run silently.
///
/// Returns `(best_move, score_in_cp)`
pub fn best_move_interruptible(
    board: &Board,
    time_budget: Duration,
    max_depth_cap: usize,
    mut repetition_table: RepetitionTable,
    mut stdout_opt: Option<&mut Stdout>,
) -> (Option<ChessMove>, i32) {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop_clone = stop.clone();
        thread::spawn(move || {
            thread::sleep(time_budget);
            stop_clone.store(true, Ordering::Relaxed);
        });
    }

    let mut transpo_table = TranspositionTable::new();
    let mut pv_table = PVTable::default();

    let color = if board.side_to_move() == Color::White { 1 } else { -1 };
    let t0 = Instant::now();
    let mut depth = 1;
    let mut best_move = None;
    let mut best_score = 0;

    EVAL_COUNT.store(0, Ordering::Relaxed);
    DEPTH_COUNT.store(0, Ordering::Relaxed);

    while depth <= max_depth_cap && !should_stop(&stop) {
        let (bm, score) = root_search(
            board,
            depth,
            &mut transpo_table,
            &mut repetition_table,
            &mut pv_table,
            &stop,
        );

        if bm.is_some() {
            best_move = bm;
            best_score = score;
        }
        DEPTH_COUNT.store(depth, Ordering::Relaxed);

        if should_stop(&stop) { break; }

        //if should_stop(&stop) { break; }

        if let Some(out) = stdout_opt.as_mut() {
            let nodes = EVAL_COUNT.load(Ordering::Relaxed) as u64;
            let time_ms = t0.elapsed().as_millis() as u64;
            let nps = if time_ms > 0 { nodes * 1_000 / time_ms } else { 0 };

            let pv_txt_opt = pv_table.get(&board.get_hash());

            if let Some(pv_txt) = pv_txt_opt {
                let score_str = if best_score.abs() >= 100_000 {
                    format!("mate {}", if best_score > 0 { 0 } else { -0 })
                } else {
                    format!("cp {}", best_score * color)
                };

                let info_line = format!(
                    "info depth {} score {} nodes {} nps {} time {} pv {}",
                    depth, score_str, nodes, nps, time_ms, pv_txt
                );
                send_message(&mut **out, &info_line);
            }

        }

        depth += 1;
        EVAL_COUNT.store(0, Ordering::Relaxed);
    }

    (best_move, best_score)
}

