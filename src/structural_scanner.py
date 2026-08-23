import asyncio
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from data.bybit_data import BybitService
from scanner import CRYPTO_UNIVERSES


INTERVAL = "60"
DAYS = 180
EXPECTED_BARS = DAYS * 24
MIN_COVERAGE = 0.90
CONCURRENCY = 2
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 1
OUTPUT_PATH = Path(__file__).with_name("approved_pairs.json")


@dataclass
class StructuralPair:
    x_sym: str
    y_sym: str
    beta: float
    half_life: float
    categories: list[str]


def cal_half_life(spread) -> float | None:
    s = np.asarray(spread, dtype=float)
    if len(s) < 2 or not np.all(np.isfinite(s)):
        return None

    lag = s[:-1]
    delta = s[1:] - lag
    lam, _ = np.polyfit(lag, delta, 1)
    if not np.isfinite(lam) or lam >= 0:
        return None

    half_life = -math.log(2) / lam
    return float(half_life) if math.isfinite(half_life) and half_life > 0 else None


def evaluate_pair(sym1: str, sym2: str, map1: dict, map2: dict, categories: list[str]) -> StructuralPair | None:
    common_times = sorted(set(map1) & set(map2))
    if len(common_times) < math.ceil(EXPECTED_BARS * MIN_COVERAGE):
        return None

    try:
        price1 = float(map1[common_times[-1]])
        price2 = float(map2[common_times[-1]])
        if not math.isfinite(price1) or not math.isfinite(price2) or price1 <= 0 or price2 <= 0:
            return None

        if price1 > price2:
            x_sym, y_sym, map_x, map_y = sym2, sym1, map2, map1
        else:
            x_sym, y_sym, map_x, map_y = sym1, sym2, map1, map2

        x = np.asarray([map_x[t] for t in common_times], dtype=float)
        y = np.asarray([map_y[t] for t in common_times], dtype=float)
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(x <= 0) or np.any(y <= 0):
            return None

        log_x = np.log(x)
        log_y = np.log(y)
        johansen = coint_johansen(np.column_stack((log_x, log_y)), det_order=0, k_ar_diff=1)
        if not math.isfinite(johansen.lr1[0]) or johansen.lr1[0] <= johansen.cvt[0, 1]:
            return None

        vector = np.real_if_close(johansen.evec[:, 0])
        if np.iscomplexobj(vector) or not np.all(np.isfinite(vector)) or abs(vector[1]) < np.finfo(float).eps:
            return None

        beta = float(-vector[0] / vector[1])
        if not math.isfinite(beta):
            return None

        half_life = cal_half_life(log_y - beta * log_x)
        if half_life is None:
            return None
    except Exception as e:
        print(f"structural_scanner: skip {sym1}/{sym2}: {e}")
        return None

    return StructuralPair(x_sym, y_sym, beta, half_life, categories)


async def fetch_history(api, symbol: str, sem: asyncio.Semaphore) -> tuple[str, dict | None]:
    async with sem:
        for attempt in range(FETCH_ATTEMPTS):
            try:
                data = await api.get_past_price(symbol, INTERVAL, days=DAYS)
            except Exception as e:
                print(f"structural_scanner: request failed for {symbol}: {e}")
                data = None
            if data:
                return symbol, data

            if attempt < FETCH_ATTEMPTS - 1:
                delay = FETCH_BACKOFF_SECONDS * (2 ** attempt)
                print(f"structural_scanner: retrying {symbol} in {delay}s after unavailable/rate-limited response")
                await asyncio.sleep(delay)

    print(f"structural_scanner: skip {symbol}: unavailable after {FETCH_ATTEMPTS} attempts")
    return symbol, None


def write_approved_pairs(pairs: list[StructuralPair]) -> None:
    with OUTPUT_PATH.open("w") as f:
        json.dump([asdict(pair) for pair in pairs], f, indent=2)
        f.write("\n")


async def main():
    api = BybitService("dummy", "dummy", testnet=False, demo=True)
    symbols = sorted({symbol for category in CRYPTO_UNIVERSES.values() for symbol in category})

    print(f"structural_scanner: fetching {len(symbols)} symbols ({DAYS}d, 1h candles, concurrency {CONCURRENCY})")
    sem = asyncio.Semaphore(CONCURRENCY)
    fetched = await asyncio.gather(*(fetch_history(api, symbol, sem) for symbol in symbols))
    data = {symbol: candles for symbol, candles in fetched if candles}

    pair_categories: dict[tuple[str, str], list[str]] = {}
    for category, symbols in CRYPTO_UNIVERSES.items():
        for sym1, sym2 in combinations(sorted(symbol for symbol in symbols if symbol in data), 2):
            pair_categories.setdefault((sym1, sym2), []).append(category)

    approved = []
    for (sym1, sym2), categories in pair_categories.items():
        pair = evaluate_pair(sym1, sym2, data[sym1], data[sym2], categories)
        if pair:
            approved.append(pair)

    approved.sort(key=lambda pair: pair.half_life)
    write_approved_pairs(approved)
    print(f"structural_scanner: wrote {len(approved)} approved pair(s) to {OUTPUT_PATH}")
    for pair in approved:
        print(f"  {pair.x_sym}/{pair.y_sym}: beta={pair.beta:.4f}, half_life={pair.half_life:.1f}h ({', '.join(pair.categories)})")


if __name__ == "__main__":
    asyncio.run(main())
