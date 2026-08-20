import asyncio
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.bybit_data import BybitService
from models.expected_value import ExpectedValueCalculator
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# note: tickers verified against Bybit linear perps (2026-08). MEME small-caps
# trade as 1000x-unit contracts (SHIB1000, 1000PEPE, ...). MKR trades as SKYUSDT.
# delisted/never-listed: TON, FET, TRU, OM.
CRYPTO_UNIVERSES = {
    "C1_MAJORS": [
        "BTCUSDT", "ETHUSDT", "XRPUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT"
    ],
    "C2_L1": [
        "ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "NEARUSDT", "DOTUSDT"
    ],
    "C3_L2_BETA": [
        "ETHUSDT", "ARBUSDT", "OPUSDT", "POLUSDT"
    ],
    "C4_DEFI": [
        "ETHUSDT", "UNIUSDT", "AAVEUSDT", "LINKUSDT"
    ],
    "C5_STABLES": [
        "USDCUSDT", "DAIUSDT", "FDUSDUSDT"
    ],
    "C6_WRAPPED_LST": [
        "BTCUSDT", "WBTCUSDT", "ETHUSDT", "STETHUSDT"
    ],
    "C7_AI": [
        "FETUSDT", "RNDRUSDT", "TAOUSDT"
    ],
    "C8_GAMING": [
        "IMXUSDT", "GALAUSDT", "SANDUSDT", "MANAUSDT"
    ],
    "C9_MEMES": [
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT"
    ]
}

FAST_INTERVAL = 3
FAST_DAYS = 30
SLOW_INTERVAL = 60
SLOW_DAYS = 180
TRAIN_FRACTION = 0.70
MAX_HALF_LIFE_BARS = 160
TRADEABLE_FDR_ALPHA = 0.05
CANDIDATE_FDR_ALPHA = 0.10
RESULT_VALID_HOURS = 24
CONCURRENCY = 8
MIN_COVERAGE = 0.90

expected_fast = FAST_DAYS * 24 * 60 // FAST_INTERVAL
expected_slow = SLOW_DAYS * 24 * 60 // SLOW_INTERVAL

# Legacy aliases for compatibility prior to full pipeline refactor
WINDOW = MAX_HALF_LIFE_BARS
INTERVAL = FAST_INTERVAL
MAX_BARS = MAX_HALF_LIFE_BARS
DAYS = FAST_DAYS
STRUCTURE_BARS = (FAST_DAYS * 24 * 60) // FAST_INTERVAL


@dataclass
class EvaluationResult:
    x_sym: str
    y_sym: str
    category: str
    is_eligible: bool
    rejection_reason: str | None = None
    p_fast: float | None = None
    p_slow: float | None = None
    p_joint: float | None = None
    p_joint_adj: float | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    alpha_fast: float | None = None
    alpha_slow: float | None = None
    half_life: float | None = None
    n_common_fast: int = 0
    n_common_slow: int = 0
    price_y: float = 0.0
    tradeable: bool = False
    candidate: bool = False
    categories: list[str] = field(default_factory=list)


def check_dataset_eligibility(n_common: int, expected_count: int, timeframe_name: str) -> tuple[bool, str | None]:
    """
    Checks whether an aligned X/Y timestamp intersection dataset meets coverage
    and validation segment size requirements.

    Returns (is_eligible, rejection_reason).
    """
    min_required = math.ceil(round(expected_count * MIN_COVERAGE, 6))
    if n_common < min_required:
        return False, f"insufficient {timeframe_name} history"

    split = int(n_common * TRAIN_FRACTION)
    val_len = n_common - split
    if val_len < 21:
        return False, f"insufficient {timeframe_name} validation sample"

    return True, None


async def _fetch_dataset(api, sym, interval, days, sem):
    async with sem:
        try:
            m = await api.get_past_price(sym, str(interval), days=days)
            if not m:
                print(f"  skip {sym} ({interval}m): unavailable history")
                return sym, interval, None
            return sym, interval, m
        except Exception as e:
            print(f"  skip {sym} ({interval}m): fetch error - {e}")
            return sym, interval, None


def validate_split(log_x, log_y):
    """
    Fits OLS on the first TRAIN_FRACTION (70%) of log prices, then evaluates
    stationarity of the resulting spread on the remaining (30%) validation slice.

    Returns (p_value, alpha, beta, validation_spread).
    """
    x = np.asarray(log_x, dtype=float)
    y = np.asarray(log_y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("log_x and log_y must be 1-dimensional")
    if len(x) != len(y):
        raise ValueError("log_x and log_y must have equal lengths")
    if len(x) == 0:
        raise ValueError("log_x and log_y cannot be empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("log_x and log_y must contain only finite float values")

    split = int(len(x) * TRAIN_FRACTION)
    if split < 2 or (len(x) - split) < 2:
        raise ValueError("Insufficient sample size for split validation")

    x_train = x[:split]
    y_train = y[:split]

    ols_model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
    alpha = float(ols_model.params[0])
    beta = float(ols_model.params[1])

    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("OLS fit produced non-finite parameters")

    x_val = x[split:]
    y_val = y[split:]
    validation_spread = y_val - (beta * x_val + alpha)

    if not np.all(np.isfinite(validation_spread)):
        raise ValueError("Validation spread contains non-finite values")

    adf_res = adfuller(validation_spread)
    p_value = float(adf_res[1])

    return p_value, alpha, beta, validation_spread


def evaluate_pair(
    sym1: str,
    sym2: str,
    map1_fast: dict,
    map2_fast: dict,
    map1_slow: dict,
    map2_slow: dict,
    category: str | list[str],
    ev_calculator: ExpectedValueCalculator,
) -> EvaluationResult:
    """
    Evaluates cointegration and trading eligibility for a pair of coins across fast
    and slow timeframes.
    """
    if isinstance(category, list):
        categories = list(category)
        cat_str = ", ".join(categories)
    else:
        cat_str = str(category)
        categories = [c.strip() for c in cat_str.split(",") if c.strip()]

    common_fast = sorted(set(map1_fast) & set(map2_fast))
    common_slow = sorted(set(map1_slow) & set(map2_slow))

    ok_fast, reason_fast = check_dataset_eligibility(len(common_fast), expected_fast, "fast")
    if not ok_fast:
        return EvaluationResult(
            x_sym=sym1, y_sym=sym2, category=cat_str, categories=categories, is_eligible=False, rejection_reason=reason_fast
        )

    ok_slow, reason_slow = check_dataset_eligibility(len(common_slow), expected_slow, "slow")
    if not ok_slow:
        return EvaluationResult(
            x_sym=sym1, y_sym=sym2, category=cat_str, categories=categories, is_eligible=False, rejection_reason=reason_slow
        )

    # Force Y to be the expensive coin based on latest common fast timestamp
    latest_t_fast = common_fast[-1]
    p1 = map1_fast[latest_t_fast]
    p2 = map2_fast[latest_t_fast]

    if p1 is None or p2 is None or not math.isfinite(p1) or not math.isfinite(p2) or p1 <= 0 or p2 <= 0:
        return EvaluationResult(
            x_sym=sym1, y_sym=sym2, category=cat_str, categories=categories, is_eligible=False, rejection_reason="non-positive or non-finite price"
        )

    if p1 > p2:
        x_sym, y_sym = sym2, sym1
        mapx_fast, mapy_fast = map2_fast, map1_fast
        mapx_slow, mapy_slow = map2_slow, map1_slow
    else:
        x_sym, y_sym = sym1, sym2
        mapx_fast, mapy_fast = map1_fast, map2_fast
        mapx_slow, mapy_slow = map1_slow, map2_slow

    # Validate price positivity and finiteness across all timestamps
    for t in common_fast:
        px, py = mapx_fast[t], mapy_fast[t]
        if px is None or py is None or not math.isfinite(px) or not math.isfinite(py) or px <= 0 or py <= 0:
            return EvaluationResult(
                x_sym=x_sym, y_sym=y_sym, category=cat_str, categories=categories, is_eligible=False, rejection_reason="non-positive or non-finite price in fast data"
            )

    for t in common_slow:
        px, py = mapx_slow[t], mapy_slow[t]
        if px is None or py is None or not math.isfinite(px) or not math.isfinite(py) or px <= 0 or py <= 0:
            return EvaluationResult(
                x_sym=x_sym, y_sym=y_sym, category=cat_str, categories=categories, is_eligible=False, rejection_reason="non-positive or non-finite price in slow data"
            )

    log_x_fast = [math.log(mapx_fast[t]) for t in common_fast]
    log_y_fast = [math.log(mapy_fast[t]) for t in common_fast]
    log_x_slow = [math.log(mapx_slow[t]) for t in common_slow]
    log_y_slow = [math.log(mapy_slow[t]) for t in common_slow]

    try:
        p_fast, alpha_fast, beta_fast, val_spread_fast = validate_split(log_x_fast, log_y_fast)
        p_slow, alpha_slow, beta_slow, val_spread_slow = validate_split(log_x_slow, log_y_slow)
        
        if p_fast is None or not math.isfinite(p_fast) or p_slow is None or not math.isfinite(p_slow):
            return EvaluationResult(
                x_sym=x_sym, y_sym=y_sym, category=cat_str, categories=categories, is_eligible=False, rejection_reason="non-finite ADF p-value"
            )
            
        p_joint = max(p_fast, p_slow)
        hl = ev_calculator.cal_half_life(val_spread_fast)
    except Exception as e:
        return EvaluationResult(
            x_sym=x_sym, y_sym=y_sym, category=cat_str, categories=categories, is_eligible=False, rejection_reason=f"validation/ADF fit error: {e}"
        )

    price_y = mapy_fast[latest_t_fast]

    return EvaluationResult(
        x_sym=x_sym,
        y_sym=y_sym,
        category=cat_str,
        categories=categories,
        is_eligible=True,
        p_fast=p_fast,
        p_slow=p_slow,
        p_joint=p_joint,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        alpha_fast=alpha_fast,
        alpha_slow=alpha_slow,
        half_life=hl,
        n_common_fast=len(common_fast),
        n_common_slow=len(common_slow),
        price_y=price_y,
        tradeable=False,
    )


def apply_bh_correction(results_list: list[EvaluationResult]) -> None:
    """
    Applies the Benjamini-Hochberg (BH) False Discovery Rate (FDR) procedure
    to the joint p-values across all eligible pairs.
    """
    m = len(results_list)
    if m == 0:
        return

    valid_pairs = [(i, r) for i, r in enumerate(results_list) if r.p_joint is not None and math.isfinite(r.p_joint)]
    m_valid = len(valid_pairs)
    
    if m_valid == 0:
        return

    valid_pairs.sort(key=lambda x: x[1].p_joint)
    
    adjusted_values = [0.0] * m_valid
    
    for rank, (orig_i, r) in enumerate(valid_pairs, start=1):
        adjusted_values[rank-1] = r.p_joint * m_valid / rank
        
    adjusted_values[-1] = min(adjusted_values[-1], 1.0)
    for rank in range(m_valid - 2, -1, -1):
        adjusted_values[rank] = min(adjusted_values[rank + 1], adjusted_values[rank], 1.0)
        
    for rank, (orig_i, r) in enumerate(valid_pairs):
        r.p_joint_adj = adjusted_values[rank]
        
        hl_ok = (
            r.half_life is not None
            and math.isfinite(r.half_life)
            and 0 < r.half_life <= MAX_HALF_LIFE_BARS
        )
        
        passes_tradeable_fdr = (
            r.p_joint_adj is not None
            and r.p_joint_adj <= TRADEABLE_FDR_ALPHA
        )
        passes_candidate_fdr = (
            r.p_joint_adj is not None
            and r.p_joint_adj <= CANDIDATE_FDR_ALPHA
        )
        
        r.tradeable = passes_tradeable_fdr and hl_ok
        r.candidate = passes_candidate_fdr and not r.tradeable


async def main():
    load_dotenv()
    key = "dummy"
    secret = "dummy"

    # Assume 0.05% fee rate and 0.1 qty for screening
    ev_calculator = ExpectedValueCalculator(qty_y=0.1, fee_rate=0.0005, max_bars=MAX_BARS)
    api = BybitService(key, secret, testnet=False, demo=True)

    # Get all unique symbols across categories
    all_symbols = sorted(set(s for cat in CRYPTO_UNIVERSES.values() for s in cat))

    print(
        f"Fetching {len(all_symbols)} unique symbols "
        f"(fast: {FAST_INTERVAL}m {FAST_DAYS}d, slow: {SLOW_INTERVAL}m {SLOW_DAYS}d, "
        f"concurrency: {CONCURRENCY})..."
    )
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for sym in all_symbols:
        tasks.append(_fetch_dataset(api, sym, FAST_INTERVAL, FAST_DAYS, sem))
        tasks.append(_fetch_dataset(api, sym, SLOW_INTERVAL, SLOW_DAYS, sem))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    fast_data = {}
    slow_data = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        sym, interval, m = res
        if m:
            if interval == FAST_INTERVAL or str(interval) == str(FAST_INTERVAL):
                fast_data[sym] = m
            elif interval == SLOW_INTERVAL or str(interval) == str(SLOW_INTERVAL):
                slow_data[sym] = m

    # Build unique pair canonical keys and map all associated categories
    pair_categories: dict[tuple[str, str], list[str]] = {}
    for cat_name, coins in CRYPTO_UNIVERSES.items():
        valid_coins = [c for c in coins if c in fast_data and c in slow_data]
        for sym1, sym2 in combinations(sorted(valid_coins), 2):
            pair_key = tuple(sorted((sym1, sym2)))
            if pair_key not in pair_categories:
                pair_categories[pair_key] = []
            if cat_name not in pair_categories[pair_key]:
                pair_categories[pair_key].append(cat_name)

    print(f"Evaluating {len(pair_categories)} unique intra-category pairs...")
    results_list = []
    ineligible_list = []
    for (sym1, sym2), cat_list in pair_categories.items():
        try:
            res = evaluate_pair(
                sym1,
                sym2,
                fast_data[sym1],
                fast_data[sym2],
                slow_data[sym1],
                slow_data[sym2],
                cat_list,
                ev_calculator,
            )
            if res.is_eligible:
                results_list.append(res)
            else:
                ineligible_list.append(res)
        except Exception as e:
            print(f"scanner: error evaluating pair {sym1}/{sym2}: {e}")

    if ineligible_list:
        print(f"\nIneligible pair(s) skipped ({len(ineligible_list)} total):")
        for r in ineligible_list:
            reason = r.rejection_reason if r.rejection_reason else "ineligible dataset"
            print(f"  skip {r.x_sym}/{r.y_sym} ({r.category}): {reason}")

    # Apply Benjamini-Hochberg FDR correction across all eligible pairs
    apply_bh_correction(results_list)

    winners = [r for r in results_list if r.tradeable]
    winners.sort(
        key=lambda r: (
            r.p_joint_adj if r.p_joint_adj is not None else 1.0,
            r.p_joint if r.p_joint is not None else 1.0,
        )
    )
    
    candidates = [r for r in results_list if r.candidate]
    candidates.sort(
        key=lambda r: (
            r.p_joint_adj if r.p_joint_adj is not None else 1.0,
            r.p_joint if r.p_joint is not None else 1.0,
        )
    )

    if winners:
        cat_width = max([len(r.category) for r in winners] + [len("category"), 12])
        table_width = cat_width + 92

        print("\n" + "=" * table_width)
        print(
            f"{'category':<{cat_width}}{'pair':<20}{'p_slow':>9}{'p_fast':>9}{'p_joint':>9}{'p_adj':>9}{'beta':>9}{'half_life':>11}{'bars':>7}  trade"
        )
        print("=" * table_width)
        for r in winners:
            hl_s = f"{r.half_life:.1f}" if r.half_life is not None else "drift"
            print(
                f"{r.category:<{cat_width}}{r.x_sym+'/'+r.y_sym:<20}{r.p_slow:>9.4f}{r.p_fast:>9.4f}{r.p_joint:>9.4f}{r.p_joint_adj:>9.4f}"
                f"{r.beta_fast:>9.3f}{hl_s:>11}{r.n_common_fast:>7}  YES"
            )
    else:
        print(f"\nNo pair passes joint BH FDR <= {TRADEABLE_FDR_ALPHA} and the half-life gate.")
        
    print("\nCandidate pairs passing joint BH FDR <= 0.10\nADVISORY ONLY - NOT APPROVED FOR BOT DEPLOYMENT")
    if candidates:
        cat_width = max([len(r.category) for r in candidates] + [len("category"), 12])
        table_width = cat_width + 92
        print("=" * table_width)
        print(
            f"{'category':<{cat_width}}{'pair':<20}{'p_slow':>9}{'p_fast':>9}{'p_joint':>9}{'p_adj':>9}{'beta':>9}{'half_life':>11}{'bars':>7}"
        )
        print("=" * table_width)
        for r in candidates:
            hl_s = f"{r.half_life:.1f}" if r.half_life is not None and math.isfinite(r.half_life) else "drift"
            hl_status = "ok" if (r.half_life is not None and math.isfinite(r.half_life) and 0 < r.half_life <= MAX_HALF_LIFE_BARS) else "EXCESS" if (r.half_life is not None and r.half_life > MAX_HALF_LIFE_BARS) else "INVALID"
            print(
                f"{r.category:<{cat_width}}{r.x_sym+'/'+r.y_sym:<20}{r.p_slow:>9.4f}{r.p_fast:>9.4f}{r.p_joint:>9.4f}{r.p_joint_adj:>9.4f}"
                f"{r.beta_fast:>9.3f}{hl_s:>11} ({hl_status:<7}) {r.n_common_fast:>7}"
            )
    else:
        print("No candidate pair passes joint BH FDR <= 0.10.")

    print("\n" + "=" * 105)
    if winners:
        print(
            f"{len(winners)} tradeable pair(s) passing all gates (joint BH FDR <= {TRADEABLE_FDR_ALPHA} and "
            f"0<half_life<={MAX_HALF_LIFE_BARS} bars), no overlapping coins:"
        )
        used_coins = set()
        count = 1
        for r in winners:
            if r.x_sym in used_coins or r.y_sym in used_coins:
                continue
            used_coins.add(r.x_sym)
            used_coins.add(r.y_sym)
            print(
                f"  Bot {count} ({r.category}): coinlist = [\"{r.x_sym}\", \"{r.y_sym}\"]   "
                f"# p_joint={r.p_joint:.4f}, p_adj={r.p_joint_adj:.4f}, half_life={r.half_life:.1f} bars, price_y=${r.price_y:.3f}"
            )
            count += 1
    else:
        print("No pair currently clears all gates.")


if __name__ == "__main__":
    asyncio.run(main())
