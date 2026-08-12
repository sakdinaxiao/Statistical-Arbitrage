# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

Statistical arbitrage (cointegrated pairs trading) bot that trades perpetual futures
on **Bybit** (linear / USDT-settled contracts). Written in Python 3.12 with asyncio.

The bot trades a pair of coins `coin_x` and `coin_y`: it models the log-price spread
`log(y) - (beta * log(x) + alpha)` and enters a market-neutral position when the
spread's z-score deviates beyond an entry threshold, exiting on mean reversion or
stoploss. Multiple instances (`bot1` … `bot4`) run different pairs in parallel.

## Layout

```
src/
  main.py                # PairTrading class: wires everything together, main async loop
  bot1.py / bot2.py / bot3.py   # thin per-pair runners (gitignored — machine-specific)
  scanner.py             # offline screener: tests intra-category pairs (1mo structure ADF + 8h ADF + half-life gate)
  data/
    bybit_data.py        # BybitService: REST wrapper (tickers, klines, positions, instruments)
    enums.py             # State (NoPosition/short_y/long_y), Action (HOLD/SY_LX/...)
    dataclasses.py       # Signal dataclass
  models/
    cointegration.py     # dual-window ADF: structure (1mo) + trade (8h), dynamic retest cadence
    kalman.py            # Kalman_2D: online alpha/beta estimation with startup-tuned process noise
    welford.py           # WelfordZScore: rolling-window z-score with frozen reference while holding
    expected_value.py    # ExpectedValueCalculator: half-life (AR(1)) + EV-vs-fees gate
    beta_spike.py        # BetaChecker: MAD-based spike detector on delta-beta (two-heap rolling median)
  execution/
    strategy.py          # StatArbStrategy: z-score thresholds -> Action signals
    executor.py          # OrderExecutor: 2-leg market orders via TaskGroup, qty rounding, leg-failure unwind
    trade_logger.py      # TradeLogger: append-only CSV of orders + blocked entries
    dashboard.py         # LiveDashboard: one-line-per-candle console status
logs/                    # trade CSVs (gitignored), one file per run
backtest_data/           # kline CSVs for offline analysis
venv/                    # local virtualenv (gitignored)
.env                     # KEY / SECRET for Bybit API (gitignored, never commit)
```

## Tech stack & dependencies

- Python **3.12** (see `venv/pyvenv.cfg`); uses `asyncio.TaskGroup`, so 3.11+ is required.
- Runtime deps (installed in `venv/`): `pybit` (Bybit V5 REST), `numpy`, `pandas`,
  `statsmodels` (+ `scipy`, `patsy`), `python-dotenv`, `rich`, `requests`, `websocket-client`.
- **There is no `requirements.txt` / `pyproject.toml`.** Dependencies are only present
  in `venv/`. If you add a dependency, install it into `venv` and mention it to the user.
- No test suite, no linter/formatter config, no CI, no Dockerfile. "Tests" are run by
  hand; keep changes easy to sanity-check via `python -m py_compile` or short scripts.

## Running

All runners import with bare module paths (e.g. `from main import PairTrading`,
`from data.bybit_data import BybitService`), so **run from `src/`**:

```bash
cd src
../venv/bin/python main.py          # default pair: ETHUSDT/BTCUSDT
../venv/bin/python bot1.py          # per-pair runners (UNI/AVAX, SUI/SOL, AVAX/BCH)
../venv/bin/python scanner.py         # screener: prints cointegrated, tradeable pairs
```

`.env` must define `KEY` and `SECRET` (Bybit API credentials). `main.py` currently
connects with `demo=True, testnet=False` — i.e. **Bybit demo trading**, not live, but
treat any change here as safety-critical. The screener uses dummy credentials and
public endpoints only.

Note: `bot*.py` and `logs/` are gitignored on purpose (see `.gitignore`); they are
per-machine configs. The `BOT_ID` env var set in the bot runners is currently unused
by the rest of the code.

## Architecture / runtime flow

`PairTrading.main()` (`src/main.py`) runs one loop iteration per candle
(`timeframe = 180` seconds, i.e. 3m candles, `max_bars = 160` ≈ 8 hours of history):

1. Startup: fetch 3 months of 3m klines, run OLS for initial alpha/beta, derive
   dynamic entry/stoploss z-thresholds from the 95th percentile of historical
   |z|, compute spread half-life to size the rolling window (`window = 2 * half_life`),
   and require ADF stationarity (p < 0.05) on both the 1-month structure window and
   the 8-hour trade window before trading. The Kalman process-noise scale (`q_frac`)
   is then tuned over five candidates using a Ljung-Box test on out-of-warmup
   residuals; the smallest scale with p > 0.05 is selected, falling back to the
   candidate with the highest p-value.
2. Per tick: fetch mid prices → Kalman update for (alpha, beta, residual `et`) →
   beta-spike check → Welford z-score → fresh half-life → cointegration tracker →
   strategy signal. The cointegration retest cadence is dynamic: every 1 hour of
   candles while flat, every `3 * half_life` bars while holding (invalid half-life
   falls back to 1h). The two ADF tests are gated separately: the 8-hour trade
   test short-circuits its timer while non-stationary (rechecked every candle so
   a healed spread is caught instantly), while the 1-month structure test is
   timer-only (its short-circuit is removed — a 1-month baseline cannot heal in
   minutes, and skipping it saves the expensive ADF fit). Their counters
   (`counter_trade` / `counter_structure` in `models/cointegration.py`) reset
   independently. On top of the timers, a "danger zone" guard fires while flat:
   when `abs(z_score) >= entry * 0.85` and no stationarity test has run in the
   last `DANGER_ZONE_COOLDOWN` bars (5 = 15min, tracked by `danger_counter`,
   which resets on any test), `force_retest` is called so entries never run on
   stale flags.
3. Entries are gated: blocked when either stationarity window is non-stationary,
   when a beta spike was detected, or when expected value net of fees
   (`FEERATE = 0.001`) is not positive. Blocked entries are logged via
   `TradeLogger.log_blocked`. While holding, a structure-window failure halves
   the stop-loss (restored if it recovers); a trade-window failure only blocks
   new entries.
4. `OrderExecutor` fires both legs as IOC market orders concurrently via
   `asyncio.TaskGroup` (pybit calls are blocking, so wrapped in `asyncio.to_thread`).
   Quantities are rounded to the exchange's `qtyStep`/`minOrderQty` and scaled up to
   the ~5.5 USDT minimum notional. If one leg fails on entry, the filled leg is
   unwound; failure to unwind calls `sys.exit(1)` to avoid a naked position.
5. State is recovered at startup from open positions (`BybitService.sync_state`), so
   a restart resumes the correct long_y/short_y state.

Conventions baked into the design (respect these when editing):

- **Y is always the more expensive coin** (`symbol_list[1]`); X is the cheap one.
  Sizing is done as fixed `qty_y` units of Y, hedged by `qty_x = |beta| * qty_y *
  price_y / price_x`. The screener enforces this ordering when proposing pairs.
- z-score > 0 → short Y / long X (`Action.SY_LX`); z < 0 → `Action.SX_LY`.
- While a position is open, the z-score reference mean/std are **frozen at entry**
  (`WelfordZScore.frozen_*`); the rolling window still updates underneath.
- Cointegration is force-retested after any executed trade, after a beta spike,
  and proactively by the danger-zone guard while flat (z within 15% of entry and
  no test in the last 5 bars).
- The trade logger records, per closed round-trip, a win/lose judgement and a
  `pnl` column: direction-aware `(close - open) * qty` summed over both legs,
  minus taker fees (`FEERATE`) on entry and exit notionals of both legs. Log
  files are one per pair per day (`logs/YYYY-MM-DD/trades_<x>_<y>.csv`, appended
  across restarts). Full PnL plumbing beyond the logger was deliberately
  scrubbed (see git history); do not re-add it without being asked.

## Coding conventions

- Plain classes with snake_case-ish methods, but the codebase is inconsistent
  (e.g. `Kalman_2D`, `spread_stationaryTest`, `z_score_cal`) — **match the local
  style of the file you are editing**; do not reformat or rename existing identifiers.
- Comments are lowercase, sparse, and explain the *why* of non-obvious math or safety
  behavior. Keep that tone; don't add docstring boilerplate.
- Defensive guards everywhere: NaN checks before math, `try/except` around every
  Bybit call returning `None` on failure, log-and-continue in the main loop.
- Logging is plain `print` with a module prefix (`main: ...`, `api: ...`,
  `executor: ...`); the CSV `TradeLogger` must never raise into the trading loop.
- Regions added incrementally are wrapped in marker comments like
  `# --- DASHBOARD START ---` / `# --- LOGGER WIRE START ---`.

## Security considerations

- Never read, print, or commit `.env` (holds live Bybit `KEY`/`SECRET`).
- The bot places real orders on Bybit demo by default; flipping `demo`/`testnet`
  flags in `main.py` changes that — flag it loudly if a change touches those.
- Order execution safety rules (reduceOnly closes, leg-failure unwind, hard exit on
  unwind failure, min-notional scaling) are load-bearing; preserve them when
  refactoring `executor.py`.
