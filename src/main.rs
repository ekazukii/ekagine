// Cargo.toml dependencies (for reference):
// [dependencies]
// chess = "3.2.0"
// once_cell = "1.18.0"    # for lazy_static

mod eval;
mod search;
mod tt;
mod movegen;

use crate::search::{
    best_move_interruptible, best_move_using_iterative_deepening, uci_score_string,
};
use chess::Color::{Black, White};
use chess::{BitBoard, Board, BoardStatus, ChessMove, Color, MoveGen, Piece, Square};
use lazy_static::lazy_static;
use serde::Deserialize;
use std::collections::{hash_map, HashMap};
use std::fmt::format;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Stdout, Write};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{env, thread};

static EVAL_COUNT: AtomicUsize = AtomicUsize::new(0);
static CACHE_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEPTH_COUNT: AtomicUsize = AtomicUsize::new(0);

type StopFlag = Arc<AtomicBool>;

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ─────────────────────────────────────────────────────────────────────────────
// Search Tables keyed by the built‐in Zobrist hash (u64).
// ─────────────────────────────────────────────────────────────────────────────

pub use tt::{TTEntry, TTFlag, TranspositionTable};
type RepetitionTable = HashMap<u64, usize>;
type PVTable = HashMap<u64, ChessMove>;

/// Apply a move and update `repetition_table` by incrementing the count
/// for the new position's Zobrist hash. Returns the new Board.
///
/// Uses `Board::get_hash()` (which is the polyglot Zobrist) instead of FEN.
///
///
fn board_do_move(board: &Board, mv: ChessMove, repetition_table: &mut RepetitionTable) -> Board {
    let new_board = board.make_move_new(mv);
    let zob = new_board.get_hash();
    let counter = repetition_table.entry(zob).or_insert(0);
    *counter += 1;
    new_board
}

/// Decrement the repetition‐count for the given position's hash.
/// Called after we return from exploring a branch.
///
///
fn board_pop(board: &Board, repetition_table: &mut RepetitionTable) {
    let zob = board.get_hash();
    if let Some(count) = repetition_table.get_mut(&zob) {
        if *count > 0 {
            *count -= 1;
        }
    }
}

fn send_message(stdout: &mut Stdout, message: &str) {
    writeln!(stdout, "{}", message.clone()).unwrap();
}

/// Decide how long to think for this move.
///
/// `tokens`  – slice starting with `"go"`, exactly what you split from stdin
/// `side`    – `board.side_to_move()`
///
/// * If the GUI sent `go movetime N`, we honour that exactly.
/// * Otherwise we look at `wtime/btime`, `winc/binc`, and (optionally)
///   `movestogo` and give you a sensible **milliseconds** budget
///   with a small safety margin so you don't flag.
pub fn choose_time_for_move(tokens: &[&str], side: Color) -> Duration {
    // Defaults
    let mut wtime: u64 = 0;
    let mut btime: u64 = 0;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movestogo: u64 = 30; // assume 30 moves left if not specified
    let mut movetime: Option<u64> = None;

    let mut i = 1; // skip the "go" token itself
    while i < tokens.len() {
        match tokens[i] {
            "movetime" => {
                i += 1;
                movetime = tokens.get(i).and_then(|v| v.parse().ok());
            }
            "wtime" => {
                i += 1;
                wtime = tokens.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "btime" => {
                i += 1;
                btime = tokens.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "winc" => {
                i += 1;
                winc = tokens.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "binc" => {
                i += 1;
                binc = tokens.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "movestogo" => {
                i += 1;
                movestogo = tokens
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(movestogo);
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(ms) = movetime {
        let ms_with_margin = ms - 5;
        return Duration::from_millis(ms_with_margin.max(10));
    }

    let (time_left, increment) = match side {
        Color::White => (wtime, winc),
        Color::Black => (btime, binc),
    };

    const SAFETY_MS: u64 = 50;

    let base = time_left.saturating_sub(SAFETY_MS) / movestogo.max(1);
    let budget = base + increment / 2;

    Duration::from_millis(budget.max(10))
}

fn uci_loop() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut board = Board::default();
    let mut transpo_table = TranspositionTable::new();
    let mut repetition_table: RepetitionTable = HashMap::new();

    for line in stdin.lock().lines() {
        let input = line.unwrap();
        let tokens: Vec<&str> = input.trim().split_whitespace().collect();

        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                send_message(&mut stdout, "id name Ekagine-v1.19.3");
                send_message(&mut stdout, "id author BaptisteLoison");
                send_message(&mut stdout, "uciok");
            }
            "isready" => {
                send_message(&mut stdout, "readyok");
            }
            "ucinewgame" => {
                board = Board::default();
                transpo_table.clear();
                repetition_table.clear();
            }
            "position" => {
                repetition_table.clear();

                let mut i = 1;
                if i < tokens.len() && tokens[i] == "startpos" {
                    board = Board::default();
                    i += 1;
                } else if i < tokens.len() && tokens[i] == "fen" {
                    let fen = tokens[i + 1..i + 7].join(" ");
                    board = Board::from_str(&fen).unwrap();
                    i += 7;
                }

                if i < tokens.len() && tokens[i] == "moves" {
                    i += 1;
                    for mv_str in &tokens[i..] {
                        if let Ok(mv) = ChessMove::from_san(&board, mv_str) {
                            board = board.make_move_new(mv);
                        } else if let Ok(mv) = ChessMove::from_str(mv_str) {
                            board = board.make_move_new(mv);
                        }
                        let zob = board.get_hash();
                        *repetition_table.entry(zob).or_insert(0) += 1;
                    }
                }
            }
            "go" => {
                let time_budget = choose_time_for_move(&tokens, board.side_to_move());

                EVAL_COUNT.store(0, Ordering::Relaxed);
                CACHE_COUNT.store(0, Ordering::Relaxed);
                let zob_start = board.get_hash();
                repetition_table.insert(
                    zob_start,
                    repetition_table.get(&zob_start).cloned().unwrap_or(0) + 1,
                );

                let max_ply = 100;
                let outcome = best_move_interruptible(
                    &board,
                    time_budget,
                    99,
                    repetition_table.clone(),
                    &mut transpo_table,
                    Some(&mut stdout),
                );
                if let Some(mv) = outcome.best_move {
                    send_message(&mut stdout, format!("bestmove {}", mv).as_str());
                } else {
                    send_message(&mut stdout, "info string move is none wtf");
                    send_message(&mut stdout, "bestmove 0000");
                }
            }
            "quit" => break,
            _ => {}
        }

        stdout.flush().unwrap();
    }
}

#[derive(Deserialize, Debug)]
struct PV {
    // cp is optional because if the position is mate the engine may only provide mate.
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
    line: String,
}

#[derive(Deserialize, Debug)]
struct Eval {
    knodes: Option<u64>,
    depth: u8,
    pvs: Vec<PV>,
}

#[derive(Deserialize, Debug)]
struct PositionEvaluation {
    fen: String,
    evals: Vec<Eval>,
}

/// Reads a file with each line being a JSON object, and returns a map from FEN (String)
/// to the cp value (u8) from the first principal variation in the eval with the highest depth.
fn process_file(filepath: &str) -> io::Result<HashMap<Board, i32>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let mut result: HashMap<Board, i32> = HashMap::new();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<PositionEvaluation>(&line) {
            Ok(position_eval) => {
                if let Some(best_eval) = position_eval.evals.iter().max_by_key(|e| e.depth) {
                    if let Some(first_pv) = best_eval.pvs.get(0) {
                        if let Some(cp_value) = first_pv.cp {
                            match Board::from_str(position_eval.fen.as_str()) {
                                Ok(board) => {
                                    result.insert(board, cp_value);
                                }
                                Err(err) => eprintln!(
                                    "Failed to parse board {} Error: {})",
                                    position_eval.fen, err
                                ),
                            }
                        } else {
                            eprintln!(
                                "Skipping FEN {} because first PV has no cp value (mate maybe: {:?})",
                                position_eval.fen, first_pv.mate
                            );
                        }
                    } else {
                        eprintln!("No principal variation found for FEN {}", position_eval.fen);
                    }
                } else {
                    eprintln!("No evals found for FEN {}", position_eval.fen);
                }
            }
            Err(e) => {
                eprintln!("Failed to parse line: {}. Error: {}", line, e);
            }
        }
    }

    Ok(result)
}

fn time_fn<F, R>(func: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = func();
    let elapsed = start.elapsed();
    (result, elapsed)
}

// Benchmark evaluation with timing, eval & cache counts
fn compute_stats(data: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let p5 = sorted[(n as f64 * 0.05).floor() as usize];
    let p95 = sorted[(n as f64 * 0.95).floor().min((n - 1) as f64) as usize];

    (mean, median, min, max, std_dev, p5, p95)
}

fn benchmark_evaluation(fen_to_stockfish: &HashMap<Board, i32>) {
    let mut scores: Vec<(i32, i32)> = Vec::new();
    let mut times: Vec<Duration> = Vec::new();
    let mut eval_counts: Vec<usize> = Vec::new();
    let mut cache_counts: Vec<usize> = Vec::new();
    let mut depth_counts: Vec<usize> = Vec::new();
    let stop = Arc::new(AtomicBool::new(false));

    let max_depth = 5;
    for (key, val) in fen_to_stockfish.iter() {
        EVAL_COUNT.store(0, Ordering::Relaxed);
        CACHE_COUNT.store(0, Ordering::Relaxed);

        let (outcome, duration) = time_fn(|| {
            let mut repetition_table: RepetitionTable = HashMap::new();
            let mut transpo_table = TranspositionTable::new();
            best_move_interruptible(
                key,
                Duration::from_millis(100),
                1000,
                RepetitionTable::new(),
                &mut transpo_table,
                None,
            )
        });

        times.push(duration);
        eval_counts.push(EVAL_COUNT.load(Ordering::Relaxed));
        cache_counts.push(CACHE_COUNT.load(Ordering::Relaxed));
        depth_counts.push(DEPTH_COUNT.load(Ordering::Relaxed));

        if let Some(_mv) = outcome.best_move {
            scores.push((outcome.score, *val));
        }
    }

    // Convert durations to milliseconds
    let time_ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let evals_f64: Vec<f64> = eval_counts.iter().map(|&x| x as f64).collect();
    let caches_f64: Vec<f64> = cache_counts.iter().map(|&x| x as f64).collect();
    let diffs: Vec<f64> = scores.iter().map(|(a, b)| (*a - *b) as f64).collect();
    let depths_f64: Vec<f64> = depth_counts.iter().map(|&d| d as f64).collect();

    let (avg_time, med_time, min_time, max_time, std_time, p5_time, p95_time) =
        compute_stats(&time_ms);
    let (avg_e, med_e, min_e, max_e, std_e, p5_e, p95_e) = compute_stats(&evals_f64);
    let (avg_c, med_c, min_c, max_c, std_c, p5_c, p95_c) = compute_stats(&caches_f64);
    let (avg_d, med_d, min_d, max_d, std_d, p5_d, p95_d) = compute_stats(&diffs);
    let (avg_dpt, med_dpt, min_dpt, max_dpt, std_dpt, p5_dpt, p95_dpt) = compute_stats(&depths_f64);

    println!("⏱ Timing (ms)     : Avg {:.2}, Med {:.2}, Min {:.2}, Max {:.2}, Std {:.2}, P5 {:.2}, P95 {:.2}",
             avg_time, med_time, min_time, max_time, std_time, p5_time, p95_time);
    println!("📊 Eval Counts     : Avg {:.2}, Med {:.2}, Min {:.0}, Max {:.0}, Std {:.2}, P5 {:.0}, P95 {:.0}",
             avg_e, med_e, min_e, max_e, std_e, p5_e, p95_e);
    println!("🗃 Cache Counts    : Avg {:.2}, Med {:.2}, Min {:.0}, Max {:.0}, Std {:.2}, P5 {:.0}, P95 {:.0}",
             avg_c, med_c, min_c, max_c, std_c, p5_c, p95_c);
    println!("🎯 Score Diffs     : Avg {:.2}, Med {:.2}, Min {:.2}, Max {:.2}, Std {:.2}, P5 {:.2}, P95 {:.2}",
             avg_d, med_d, min_d, max_d, std_d, p5_d, p95_d);
    println!("🔍 Search Depth   : Avg {:.2}, Med {:.2}, Min {:.0}, Max {:.0}, Std {:.2}, P5 {:.0}, P95 {:.0}",
             avg_dpt, med_dpt, min_dpt, max_dpt, std_dpt, p5_dpt, p95_dpt
    );
}

const NEG_INFINITY: i32 = -100_000_000;
const POS_INFINITY: i32 = 100_000_000;
const PV_WIDTH: usize = 1; // keep up to X moves per position for ordering

const QUIESCE_REMAIN: usize = 15;

pub fn compute_best_from_fen(
    fen: &str,
    max_depth: usize,
) -> Result<(Option<ChessMove>, i32), String> {
    let board = Board::from_str(fen).map_err(|e| format!("Invalid FEN '{}': {}", fen, e))?;

    let mut transpo_table = TranspositionTable::new();
    let mut repetition_table: HashMap<u64, usize> = HashMap::new();
    repetition_table.insert(board.get_hash(), 1);

    let outcome = best_move_using_iterative_deepening(
        &board,
        max_depth,
        &mut transpo_table,
        &mut repetition_table,
    );

    Ok((outcome.best_move, outcome.score))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} [--uci] [--benchmark [FILE]] [--best <FEN> [DEPTH]]",
            args[0]
        );
        std::process::exit(1);
    }

    match args[1].as_str() {
        "--version" => {
            println!("Version: {}", VERSION);
        }

        "--uci" => {
            uci_loop();
        }

        "--benchmark" => {
            let filepath = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("/Users/ekazuki/Downloads/lichess_db_eval_1000.jsonl");
            match process_file(filepath) {
                Ok(hash_map) => benchmark_evaluation(&hash_map),
                Err(err) => eprintln!("Failed to open {}: {}", filepath, err),
            }
        }

        "--best" => {
            if args.len() < 3 {
                eprintln!("Usage: {} --best <FEN> [DEPTH]", args[0]);
                std::process::exit(1);
            }
            let fen = &args[2];
            let max_depth: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);

            match compute_best_from_fen(fen, max_depth) {
                Ok((Some(mv), score)) => {
                    let board_disp = Board::from_str(fen).unwrap();
                    let score_str = uci_score_string(score, board_disp.side_to_move());
                    println!("bestmove {} (score {})", mv, score_str)
                }
                Ok((None, _)) => println!("No legal move available"),
                Err(msg) => {
                    eprintln!("Error: {}", msg);
                    std::process::exit(1);
                }
            }
        }

        _ => {
            eprintln!("Unknown command: {}", args[1]);
            eprintln!(
                "Usage: {} [--version] [--uci] [--benchmark [FILE]] [--best <FEN> [DEPTH]]",
                args[0]
            );
            std::process::exit(1);
        }
    }
}

// 1rbqkb1r/pppp1ppp/2n1p3/8/3PP1n1/4BN1P/PPP2PP1/RN1QKB1R w KQk - 0 5
// Coup de tour bizarre

// WTF Is that ?
//1rbqkb1r/pppp1ppp/2n1p3/6N1/3PP1n1/4B2P/PPP2PP1/RN1QKB1R b KQk - 0 6

//[Event "*"]
// [Site "*"]
// [Date "2025.06.07"]
// [Time "20:40:25"]
// [Round "*"]
// [White "Rust v7"]
// [Black "hm::Human"]
// [Result "*"]
// [ECO "C50"]
// [Opening "Giuoco Piano"]
// [TimeControl "movetime: 10"]
// [PlyCount "27"]
//
// 1.Nc3 {+0.0/8 10029 1523284}  e5 2.Nf3 {+0.1/8 10032 2023131; A00: Dunst (Sleipner, Heinrichsen) opening}  Nc6
// 3.e4 {+0.4/7 10029 1888261}  Nf6 4.Bc4 {+0.2/6 10022 1044219}  Bc5
// 5.Ng5 {+0.6/7 10026 659867; C50: Giuoco Piano, four knights variation}  d5 6.exd5 {+0.8/8 10034 2793299}  b5
// 7.Bxb5 {+4.7/8 10033 4095069}  Nxd5 8.Bxc6+ {+8.4/9 10023 903031}  Ke7
// 9.Nxd5+ {+10.2/9 10027 815727}  Qxd5 10.Bxd5 {+14.2/8 10038 3199286}  Bf5
// 11.Bxa8 {+17.6/8 10034 2886168}  Rxa8 12.Qf3 {+18.9/9 10036 1075835}  g6
// 13.Qxa8 {+20.8/9 10037 1966843}  Kf6 14.Qh8+ {+24.9/9 10039 1753891}
