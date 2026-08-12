import asyncio
import math
import os
import sys
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
    ]
}

FAST_INTERVAL = 3
FAST_DAYS = 30
SLOW_INTERVAL = 60
SLOW_DAYS = 180
TRAIN_FRACTION = 0.70
MAX_HALF_LIFE_BARS = 160
FDR_ALPHA = 0.05
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


async def _fetch_symbol(api, sym, sem):
    async with sem:
        # fetch extra history so the 1-month structure slice and 8h trade window are always covered
        return sym, await api.get_past_price(sym, str(INTERVAL), days=DAYS)

def evaluate(log_x, log_y, ev_calculator):
    def get_p_val(datax, datay):
        ols_model = sm.OLS(datay, sm.add_constant(datax)).fit()
        alpha = ols_model.params[0]
        beta = ols_model.params[1]
        spread = np.array(datay) - ((beta * np.array(datax)) + alpha)
        return adfuller(spread)[1], beta, spread

    struct_x = log_x[-STRUCTURE_BARS:]
    struct_y = log_y[-STRUCTURE_BARS:]
    p_struct, _, _ = get_p_val(struct_x, struct_y)

    win_x = log_x[-WINDOW:]
    win_y = log_y[-WINDOW:]
    p_win, beta, spread_win = get_p_val(win_x, win_y)

    hl = ev_calculator.cal_half_life(spread_win)

    return p_struct, p_win, beta, hl

async def main():
    load_dotenv()
    key = "dummy"
    secret = "dummy"

    # Assume 0.05% fee rate and 0.1 qty for screening
    ev_calculator = ExpectedValueCalculator(qty_y=0.1, fee_rate=0.0005, max_bars=MAX_BARS)
    api = BybitService(key, secret, testnet=False, demo=True)

    # Get all unique symbols across categories
    all_symbols = set()
    for cat in CRYPTO_UNIVERSES.values():
        all_symbols.update(cat)

    print(f"Fetching {len(all_symbols)} unique symbols ({INTERVAL}m, {DAYS}d, {CONCURRENCY} at a time)...")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(
        *(_fetch_symbol(api, sym, sem) for sym in all_symbols)
    )
    data = {}
    for sym, m in results:
        if m and len(m) >= WINDOW:
            data[sym] = m
        else:
            print(f"  skip {sym}: insufficient data")

    rows = []
    # Only test intra-category combinations
    for cat_name, coins in CRYPTO_UNIVERSES.items():
        valid_coins = [c for c in coins if c in data]
        print(f"Evaluating {cat_name} ({len(valid_coins)} valid coins)")
        for sym1, sym2 in combinations(sorted(valid_coins), 2):
            map1, map2 = data[sym1], data[sym2]
            common = sorted(set(map1) & set(map2))
            if len(common) < WINDOW:
                continue

            # Force Y to be the expensive coin
            price1 = map1[common[-1]]
            price2 = map2[common[-1]]
            if price1 > price2:
                x_sym, y_sym = sym2, sym1
                mapx, mapy = map2, map1
                price_y = price1
            else:
                x_sym, y_sym = sym1, sym2
                mapx, mapy = map1, map2
                price_y = price2

            log_x = [math.log(mapx[t]) for t in common]
            log_y = [math.log(mapy[t]) for t in common]

            try:
                p_struct, p_win, beta, hl = evaluate(log_x, log_y, ev_calculator)
            except Exception:
                continue

            hl_ok = hl is not None and 0 < hl <= MAX_BARS
            # Strict 0.05 p-value threshold
            tradeable = p_struct < 0.05 and p_win < 0.05 and hl_ok
            rows.append((x_sym, y_sym, p_struct, p_win, beta, hl, tradeable, len(common), price_y, cat_name))

    # Sort by p_win, then half_life
    rows.sort(key=lambda r: (r[3], r[5] if r[5] is not None else 1e9))

    print("\n" + "=" * 98)
    print(f"{'category':<12}{'pair':<20}{'p_struct':>9}{'p_win':>9}{'beta':>9}{'half_life':>11}{'bars':>7}  trade")
    print("=" * 98)
    for x, y, pf, pw, beta, hl, ok, n, price_y, cat in rows:
        hl_s = f"{hl:.1f}" if hl is not None else "drift"
        flag = "  YES" if ok else ""
        print(f"{cat:<12}{x+'/'+y:<20}{pf:>9.4f}{pw:>9.4f}{beta:>9.3f}{hl_s:>11}{n:>7}{flag}")

    winners = [r for r in rows if r[6]]
    print("\n" + "=" * 98)
    if winners:
        print(f"{len(winners)} tradeable pair(s) (p_struct<0.05 AND p_win<0.05 AND 0<half_life<={MAX_BARS} bars), no overlapping coins:")
        used_coins = set()
        count = 1
        for x, y, pf, pw, beta, hl, ok, n, price_y, cat in winners:
            if x in used_coins or y in used_coins:
                continue
            used_coins.add(x)
            used_coins.add(y)
            print(f"  Bot {count} ({cat}): coinlist = [\"{x}\", \"{y}\"]   # p_win={pw:.4f}, half_life={hl:.1f} bars, price_y=${price_y:.3f}")
            count += 1
    else:
        print("No pair currently clears both gates.")

if __name__ == "__main__":
    asyncio.run(main())
