//! Bulk rescoring of bulletformat training data with the engine's own search.
//!
//! Reads a `bulletformat::ChessBoard` file (32-byte records), re-searches every
//! position at a fixed depth with the current net, and writes a copy of the
//! file with only the score field (bytes 24..26, i16 LE, stm-relative) patched.
//!
//! Record layout (see bulletformat): positions are stored side-to-move-relative
//! (vertically flipped + colours swapped when black is to move), so every
//! reconstructed position has white to move and the search score is written
//! back directly. Castling rights and en-passant are not stored by the format;
//! positions are rebuilt without them.
//!
//! Resumable: if the output file already exists, rescoring continues after the
//! last fully written record.

use crate::engine_core::Board;
use crate::search::{best_move, SearchScratch, QUIET};
use crate::RepetitionTable;
use crate::TranspositionTable;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const REC: usize = 32;
const CHUNK_RECORDS: usize = 100_000;
/// Scores beyond this (e.g. mate scores) are clamped before the i16 write.
const SCORE_CLAMP: i32 = 9999;

pub fn run_rescore(input: &str, output: &str, depth: usize, threads: usize) -> std::io::Result<()> {
    QUIET.store(true, Ordering::Relaxed);

    let in_len = std::fs::metadata(input)?.len();
    assert_eq!(
        in_len % REC as u64,
        0,
        "input is not a whole number of 32-byte records"
    );
    let total = in_len / REC as u64;

    // Resume after the last fully written record of a previous run.
    let done = std::fs::metadata(output)
        .map(|m| m.len() / REC as u64)
        .unwrap_or(0);
    let mut reader = BufReader::new(File::open(input)?);
    reader.seek(SeekFrom::Start(done * REC as u64))?;
    let mut writer = BufWriter::new(OpenOptions::new().create(true).append(true).open(output)?);

    println!(
        "rescore: {input} -> {output} depth={depth} threads={threads} total={total} resume_at={done}"
    );

    let processed = AtomicU64::new(0);
    let errors = AtomicU64::new(0);
    let clamped = AtomicU64::new(0);
    let t0 = Instant::now();

    // One TT per worker, reused across all of that worker's positions.
    let tts: Vec<Arc<TranspositionTable>> = (0..threads)
        .map(|_| Arc::new(TranspositionTable::new()))
        .collect();

    let mut buf = vec![0u8; CHUNK_RECORDS * REC];
    let mut remaining = total - done;
    while remaining > 0 {
        let n = remaining.min(CHUNK_RECORDS as u64) as usize;
        reader.read_exact(&mut buf[..n * REC])?;
        let chunk = &mut buf[..n * REC];

        let per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (wi, slice) in chunk.chunks_mut(per * REC).enumerate() {
                let tt = Arc::clone(&tts[wi]);
                let (processed, errors, clamped) = (&processed, &errors, &clamped);
                s.spawn(move || {
                    let mut scratch = SearchScratch::new(&Board::default());
                    for rec in slice.chunks_mut(REC) {
                        match rescore_record(rec, depth, &tt, &mut scratch) {
                            Ok(true) => {
                                clamped.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(false) => {}
                            Err(()) => {
                                errors.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        processed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });

        writer.write_all(chunk)?;
        writer.flush()?;
        remaining -= n as u64;

        let p = processed.load(Ordering::Relaxed);
        let rate = p as f64 / t0.elapsed().as_secs_f64();
        let eta_h = remaining as f64 / rate / 3600.0;
        println!(
            "progress: {}/{} ({:.2}%) | {:.0} pos/s | eta {:.1}h | errors={} clamped={}",
            done + p,
            total,
            100.0 * (done + p) as f64 / total as f64,
            rate,
            eta_h,
            errors.load(Ordering::Relaxed),
            clamped.load(Ordering::Relaxed),
        );
    }

    println!(
        "rescore done: {} positions in {:.1}h | errors={} clamped={}",
        processed.load(Ordering::Relaxed),
        t0.elapsed().as_secs_f64() / 3600.0,
        errors.load(Ordering::Relaxed),
        clamped.load(Ordering::Relaxed),
    );
    Ok(())
}

/// Re-search one record in place. Returns Ok(true) if the score was clamped
/// (mate-ish), Err(()) if the position could not be reconstructed (original
/// score kept).
fn rescore_record(
    rec: &mut [u8],
    depth: usize,
    tt: &Arc<TranspositionTable>,
    scratch: &mut SearchScratch,
) -> Result<bool, ()> {
    let fen = decode_fen(rec).ok_or(())?;
    let board = Board::from_str(&fen).map_err(|_| ())?;

    let repetition_table = RepetitionTable::new(board.get_hash());
    let outcome = best_move(
        &board,
        depth,
        Arc::clone(tt),
        repetition_table,
        None,
        None,
        1,
        Some(scratch),
    );

    let clamped = outcome.score.abs() > SCORE_CLAMP;
    let score = outcome.score.clamp(-SCORE_CLAMP, SCORE_CLAMP) as i16;
    rec[24..26].copy_from_slice(&score.to_le_bytes());
    Ok(clamped)
}

/// Rebuild a FEN (white to move, no castling/ep) from a bulletformat record.
fn decode_fen(rec: &[u8]) -> Option<String> {
    let occ = u64::from_le_bytes(rec[0..8].try_into().ok()?);
    let mut board64 = [0u8; 64]; // 0 = empty, else nibble | 0x10
    let mut idx = 0usize;
    let mut o = occ;
    while o != 0 {
        let sq = o.trailing_zeros() as usize;
        let nib = (rec[8 + idx / 2] >> (4 * (idx & 1))) & 0xF;
        board64[sq] = nib | 0x10;
        idx += 1;
        o &= o - 1;
    }

    const PIECES: [char; 6] = ['P', 'N', 'B', 'R', 'Q', 'K'];
    let mut fen = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            let v = board64[rank * 8 + file];
            if v == 0 {
                empty += 1;
                continue;
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
                empty = 0;
            }
            let p = (v & 0x7) as usize;
            if p > 5 {
                return None;
            }
            let c = PIECES[p];
            fen.push(if v & 0x8 != 0 {
                c.to_ascii_lowercase()
            } else {
                c
            });
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    fen.push_str(" w - - 0 1");
    Some(fen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_fen_startpos_roundtrip() {
        // Build a record for the startpos via the same nibble convention.
        let mut rec = [0u8; 32];
        let fen_pieces: [(usize, u8); 32] = {
            let mut v = vec![];
            // white back rank + pawns
            for (sq, p) in [
                (0, 3),
                (1, 1),
                (2, 2),
                (3, 4),
                (4, 5),
                (5, 2),
                (6, 1),
                (7, 3),
            ] {
                v.push((sq, p)); // RNBQKBNR
            }
            for f in 0..8 {
                v.push((8 + f, 0u8)); // P
            }
            for f in 0..8 {
                v.push((48 + f, 0x8)); // p
            }
            for (sq, p) in [
                (56, 3),
                (57, 1),
                (58, 2),
                (59, 4),
                (60, 5),
                (61, 2),
                (62, 1),
                (63, 3),
            ] {
                v.push((sq, p | 0x8));
            }
            v.sort_by_key(|x| x.0);
            v.try_into().unwrap()
        };
        let occ: u64 = fen_pieces.iter().fold(0, |acc, (sq, _)| acc | 1u64 << sq);
        rec[0..8].copy_from_slice(&occ.to_le_bytes());
        for (idx, (_, nib)) in fen_pieces.iter().enumerate() {
            rec[8 + idx / 2] |= nib << (4 * (idx & 1));
        }
        let fen = decode_fen(&rec).unwrap();
        assert_eq!(fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    }
}
