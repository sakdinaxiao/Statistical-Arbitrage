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

UNIVERSE = [
    "BTCUSDT", "ETHUSDT",            # majors
    "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT",   # L1s
    "ARBUSDT", "OPUSDT",             # eth L2s
    "APTUSDT", "SUIUSDT",            # move-lang L1s
    "DOGEUSDT", "SHIBUSDT", "WIFUSDT",  # memes
    "LINKUSDT", "UNIUSDT", "AAVEUSDT",  # defi
    "LTCUSDT", "BCHUSDT",            # btc forks
    "MATICUSDT", "NEARUSDT",
]

WINDOW = 160        # main.py retests on 160 (8 hours)
INTERVAL = 3       # candle minutes
MAX_BARS = 160       # default: reject half-life slower than this
CONCURRENCY = 8     # parallel symbol fetches

async def _fetch_symbol(api, sym, sem):
    async with sem:
        # 14 days to match the live bot's structure window
        return sym, await api.get_past_price(sym, str(INTERVAL), days=14)

def evaluate(log_x, log_y, ev_calculator):
    def get_p_val(datax, datay):
        ols_model = sm.OLS(datay, sm.add_constant(datax)).fit()
        alpha = ols_model.params[0]
        beta = ols_model.params[1]
        spread = np.array(datay) - ((beta * np.array(datax)) + alpha)
        return adfuller(spread)[1], beta, spread

    p_full, _, _ = get_p_val(log_x, log_y)
    
    win_x = log_x[-WINDOW:]
    win_y = log_y[-WINDOW:]
    p_win, beta, spread_win = get_p_val(win_x, win_y)

    hl = ev_calculator.half_life(spread_win)

    return p_full, p_win, beta, hl

async def main():
    load_dotenv()
    key = "dummy"
    secret = "dummy"

    # Assume 0.05% fee rate and 0.1 qty for screening
    ev_calculator = ExpectedValueCalculator(qty_y=0.1, fee_rate=0.0005, max_bars=MAX_BARS)
    api = BybitService(key, secret, testnet=False, demo=True)

    print(f"Fetching {len(UNIVERSE)} symbols ({INTERVAL}m, 14d, {CONCURRENCY} at a time)...")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(
        *(_fetch_symbol(api, sym, sem) for sym in UNIVERSE)
    )
    data = {}
    for sym, m in results:
        if m and len(m) >= WINDOW:
            data[sym] = m
        else:
            print(f"  skip {sym}: insufficient data")

    rows = []
    for sym1, sym2 in combinations(sorted(data), 2):
        map1, map2 = data[sym1], data[sym2]
        common = sorted(set(map1) & set(map2))
        if len(common) < WINDOW:
            continue

        # Force Y to be the expensive coin to solve the decimal stepSize issue!
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
            p_full, p_win, beta, hl = evaluate(log_x, log_y, ev_calculator)
        except Exception as e:
            continue

        hl_ok = hl is not None and 0 < hl <= MAX_BARS
        tradeable = p_full < 0.05 and p_win < 0.05 and hl_ok
        rows.append((x_sym, y_sym, p_full, p_win, beta, hl, tradeable, len(common), price_y))

    rows.sort(key=lambda r: (r[3], r[5] if r[5] is not None else 1e9))

    print("\n" + "=" * 86)
    print(f"{'pair':<20}{'p_full':>9}{'p_win':>9}{'beta':>9}{'half_life':>11}{'bars':>7}  trade")
    print("=" * 86)
    for x, y, pf, pw, beta, hl, ok, n, price_y in rows:
        hl_s = f"{hl:.1f}" if hl is not None else "drift"
        flag = "  YES" if ok else ""
        print(f"{x+'/'+y:<20}{pf:>9.4f}{pw:>9.4f}{beta:>9.3f}{hl_s:>11}{n:>7}{flag}")

    winners = [r for r in rows if r[6]]
    print("\n" + "=" * 86)
    if winners:
        print(f"{len(winners)} tradeable pair(s) (p_full<0.05 AND p_win<0.05 AND 0<half_life<={MAX_BARS} bars):")
        for x, y, pf, pw, beta, hl, ok, n, price_y in winners:
            print(f"  coinlist = [\"{x}\", \"{y}\"]   # p_win={pw:.4f}, half_life={hl:.1f} bars, price_y=${price_y:.3f}")
        
        print("\n" + "=" * 86)
        print("Mutually Exclusive Tradeable Pairs:")
        used_coins = set()
        exclusive_pairs = []
        for x, y, pf, pw, beta, hl, ok, n, price_y in winners:
            if x not in used_coins and y not in used_coins:
                exclusive_pairs.append((x, y, pw, hl, price_y))
                used_coins.add(x)
                used_coins.add(y)
        for i, (x, y, pw, hl, price_y) in enumerate(exclusive_pairs):
            print(f"  Bot {i+1}: coinlist = [\"{x}\", \"{y}\"]   # p_win={pw:.4f}, half_life={hl:.1f} bars, price_y=${price_y:.3f}")
    else:
        print("No pair currently clears both gates.")

if __name__ == "__main__":
    asyncio.run(main())
