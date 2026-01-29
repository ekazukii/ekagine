use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use chess::{Board, Color};
use crate::{estimate_moves_to_go, parse_go_tokens, EngineState};

/// Time plan for a search, specifying soft and hard limits
#[derive(Debug, Clone)]
pub struct TimePlan {
    pub soft_limit: Option<Duration>,
    pub hard_limit: Option<Duration>,
    pub infinite: bool,
}

impl TimePlan {
    pub fn fixed(duration: Duration) -> Self {
        let guard_soft = Duration::from_millis(10);
        let guard_hard = Duration::from_millis(2);
        let soft = if duration > guard_soft {
            duration.saturating_sub(guard_soft)
        } else {
            duration
        };
        let hard = if duration > guard_hard {
            duration.saturating_sub(guard_hard)
        } else {
            duration
        };

        Self {
            soft_limit: Some(soft),
            hard_limit: Some(hard.max(soft)),
            infinite: false,
        }
    }

    pub fn infinite() -> Self {
        Self {
            soft_limit: None,
            hard_limit: None,
            infinite: true,
        }
    }

    pub fn has_deadlines(&self) -> bool {
        self.soft_limit.is_some() || self.hard_limit.is_some()
    }
}

pub type StopFlag = Arc<AtomicBool>;

/// Handle for managing time during search with soft and hard deadlines
pub struct TimeManagerHandle {
    start: Instant,
    soft_limit: Option<Duration>,
    soft_deadline: Option<Instant>,
    cancel: Arc<AtomicBool>,
}

impl TimeManagerHandle {
    pub fn new(stop: &StopFlag, plan: &TimePlan, start: Instant) -> Option<Self> {
        if !plan.has_deadlines() {
            return None;
        }

        let soft_deadline = plan.soft_limit.map(|d| start + d);
        let hard_deadline = plan.hard_limit.map(|d| start + d);
        let cancel = Arc::new(AtomicBool::new(false));

        if let Some(deadline) = hard_deadline {
            let cancel_clone = Arc::clone(&cancel);
            let stop_clone = stop.clone();
            thread::spawn(move || loop {
                if cancel_clone.load(Ordering::Relaxed) {
                    break;
                }

                let now = Instant::now();
                if now >= deadline {
                    stop_clone.store(true, Ordering::Relaxed);
                    break;
                }

                let remaining = deadline.saturating_duration_since(now);
                if remaining.is_zero() {
                    thread::yield_now();
                } else {
                    let sleep_for = remaining.min(Duration::from_millis(5));
                    thread::sleep(sleep_for);
                }
            });
        }

        Some(Self {
            start,
            soft_limit: plan.soft_limit,
            soft_deadline,
            cancel,
        })
    }

    pub fn soft_limit_reached(&self, now: Instant) -> bool {
        if let Some(deadline) = self.soft_deadline {
            return now >= deadline;
        }
        false
    }

    pub fn soft_limit_reached_scaled(&self, now: Instant, scale: f64) -> bool {
        if let Some(limit) = self.soft_limit {
            let scaled_limit = Duration::from_secs_f64(limit.as_secs_f64() * scale);
            let scaled_deadline = self.start + scaled_limit;
            return now >= scaled_deadline;
        }
        false
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for TimeManagerHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Time scaling factors for intelligent time management
pub struct TimeScaleFactors {
    stability: f64,          // Based on best move changes
    node_fraction: f64,      // Based on nodes spent on best move
    score_trend: f64,        // Based on score improvement/drop
    fail_high_early: bool,   // First move caused beta cutoff
}

impl TimeScaleFactors {
    pub fn new() -> Self {
        Self {
            stability: 1.0,
            node_fraction: 1.0,
            score_trend: 1.0,
            fail_high_early: false,
        }
    }

    pub fn calculate_stability_factor(best_move_changes: usize, depth: usize) -> f64 {
        if best_move_changes == 0 && depth >= 6 {
            0.7 // Very stable - can stop earlier
        } else if best_move_changes == 1 {
            0.85 // Stable
        } else if best_move_changes == 2 {
            1.0 // Normal
        } else if best_move_changes == 3 {
            1.3 // Unstable - search longer
        } else {
            1.5 // Very unstable
        }
    }

    pub fn calculate_node_fraction_factor(best_move_nodes: u64, total_nodes: u64) -> f64 {
        if total_nodes == 0 {
            return 1.0;
        }

        let fraction = best_move_nodes as f64 / total_nodes as f64;

        if fraction < 0.2 {
            0.7 // Very clear best move - stop earlier
        } else if fraction < 0.3 {
            0.85 // Clear
        } else if fraction < 0.5 {
            1.0 // Normal
        } else if fraction < 0.6 {
            1.2 // Unclear - search longer
        } else {
            1.5 // Very unclear position
        }
    }

    pub fn calculate_score_trend_factor(prev_score: Option<i32>, current_score: i32) -> f64 {
        let Some(prev) = prev_score else {
            return 1.0;
        };

        // Ignore score changes in mate positions (unreliable)
        const MATE_THRESHOLD: i32 = 10000;
        if prev.abs() > MATE_THRESHOLD || current_score.abs() > MATE_THRESHOLD {
            return 1.0;
        }

        let score_change = current_score - prev;

        if score_change < -100 {
            1.8 // Huge drop (>1 pawn) - possible blunder, search longer
        } else if score_change < -50 {
            1.4 // Significant drop
        } else if score_change < -20 {
            1.1 // Small drop
        } else if score_change > 50 {
            0.8 // Big improvement - can stop earlier
        } else {
            1.0 // Normal variation
        }
    }

    pub fn set_stability(&mut self, value: f64) {
        self.stability = value;
    }

    pub fn set_node_fraction(&mut self, value: f64) {
        self.node_fraction = value;
    }

    pub fn set_score_trend(&mut self, value: f64) {
        self.score_trend = value;
    }

    pub fn set_fail_high_early(&mut self, value: bool) {
        self.fail_high_early = value;
    }

    pub fn compute_scale(&self) -> f64 {
        let mut scale = self.stability * self.node_fraction * self.score_trend;

        // Fail-high on first move means position is likely good for us
        if self.fail_high_early {
            scale *= 0.7;
        }

        // Clamp to reasonable range to avoid extreme time allocations
        scale.clamp(0.5, 2.5)
    }
}

pub fn plan_time_for_move(
    tokens: &[&str],
    side: Color,
    board: &Board,
    state: &EngineState,
) -> Option<TimePlan> {
    let parsed = parse_go_tokens(tokens);

    if parsed.infinite || parsed.ponder {
        return Some(TimePlan::infinite());
    }

    if let Some(ms) = parsed.movetime {
        let capped = ms.max(1);
        return Some(TimePlan::fixed(Duration::from_millis(capped)));
    }

    let (time_left_opt, increment_opt) = match side {
        Color::White => (parsed.wtime, parsed.winc),
        Color::Black => (parsed.btime, parsed.binc),
    };

    let time_left = time_left_opt.unwrap_or(0);
    let increment = increment_opt.unwrap_or(0);

    if time_left == 0 && increment == 0 {
        return Some(TimePlan::fixed(Duration::from_millis(5)));
    }

    let moves_remaining = parsed
        .movestogo
        .filter(|m| *m > 0)
        .unwrap_or_else(|| estimate_moves_to_go(board));

    const MIN_RESERVE_MS: u64 = 40;
    const RESERVE_DIVISOR: u64 = 20;
    const MIN_THINK_MS: u64 = 5;
    const SOFT_GUARD_MS: u64 = 8;
    const HARD_GUARD_MS: u64 = 2;

    let mut reserve = time_left / RESERVE_DIVISOR;
    if reserve < MIN_RESERVE_MS {
        reserve = MIN_RESERVE_MS;
    }
    if reserve > time_left / 2 {
        reserve = time_left / 2;
    }

    let mut usable = time_left.saturating_sub(reserve);
    if usable < MIN_THINK_MS {
        usable = time_left.max(increment);
    }
    if usable < MIN_THINK_MS {
        usable = MIN_THINK_MS;
    }

    let moves = moves_remaining.max(1);
    let per_move = usable / moves;
    let inc_share = (increment * 7) / 10;

    let mut soft_ms = per_move.saturating_add(inc_share).max(MIN_THINK_MS);
    if soft_ms + SOFT_GUARD_MS > usable {
        soft_ms = usable.saturating_sub(SOFT_GUARD_MS).max(MIN_THINK_MS);
    }

    if let Some(prev) = state.last_think_time {
        let prev_ms = prev.as_millis() as u64;
        // Smooth towards recent usage without exceeding usable budget.
        soft_ms = ((soft_ms + prev_ms) / 2).clamp(MIN_THINK_MS, usable);
    }

    let mut hard_ms = soft_ms
        .saturating_add(per_move / 2)
        .saturating_add(increment / 2)
        .saturating_add(5);
    if hard_ms > usable {
        hard_ms = usable.saturating_sub(HARD_GUARD_MS).max(soft_ms);
    }

    Some(TimePlan {
        soft_limit: Some(Duration::from_millis(soft_ms)),
        hard_limit: Some(Duration::from_millis(hard_ms.max(soft_ms))),
        infinite: false,
    })
}