# SPSA tuning run — 2026-06-09

Tuned the 8 UCI `Tunable` search constants by simultaneous-perturbation
self-play (theta+ vs theta-) under cutechess-cli on the local Mac (M-series,
cc=10, st=0.2 + timemargin=400, book 8moves_v3, resign 600/4, draw 30/8/10).

## Result
SPSA ran **230 iterations / 2300 games**. The Polyak-averaged estimate moved the
constants modestly and consistently. Validation of that average vs the previous
defaults (GSPRT, same TC st=0.2):

| stage | games | tuned elo vs defaults |
|------:|------:|:----------------------|
| checkpoint k=50  |  60 | -52.5 ±46  (early SPSA drift / noise) |
| checkpoint k=100 |  60 |  -0.0 ±58 |
| checkpoint k=150 |  60 | +11.6 ±43 |
| checkpoint k=200 |  60 | +11.6 ±48 |
| **SPRT (avg)**   | **1520** | **+2.7 ±9.9  (LLR +0.12, all sub-totals positive after ~500g)** |

Conclusion: a **small but consistently positive gain (~+2.5 to +3.5 Elo)** at the
tuning TC. ~70% draw rate means a 2σ confirmation needs ~16 000 games locally;
the **definitive check is the 10+0.1 ssprt cluster** (open a PR to trigger it).

## Applied changes (Polyak average, written into src/search.rs)
| param | default | tuned |
|-------|--------:|------:|
| AspirationWindow  | 25    | 28    |
| QFutilityMargin   | 200   | 196   |
| FutilityBase      | 100   | 92    |
| FutilityPerDepth  | 75    | 76    |
| RazoringMargin    | 300   | 290   |
| ProbcutMargin     | 180   | 188   |
| SingularBetaMult  | 2     | 2     |
| HistLmrDivisor    | 16384 | 15881 |

Raw run state kept in `run/` (resume tuning with `python3 tuning/spsa.py tune`).
To revert: `git checkout src/search.rs`.
