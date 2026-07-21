"""
Sandbox: qty pump-up + exact-amount close test.

Runs on Bybit demo trading, entry/exit once on a chosen pair, then asserts
no position is left open.  Useful for verifying:
  - tiny qty_y gets pumped up past the ~$6 minimum-notional gate,
  - the x-leg is scaled and rounded to lot rules,
  - exit closes the exact filled size (not qty=0 "close all"),
  - both legs are flat after the round-trip.

Run from src/:
    cd src && ../venv/bin/python sandbox/test_qty_close.py [coin_x] [coin_y] [qty_y]

Defaults: ETHUSDT BTCUSDT 0.0001
"""
import asyncio
import os
import sys
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv

# Running as src/sandbox/<file>.py -> add src/ to import path.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.bybit_data import BybitService
from execution.trade_logger import TradeLogger


class QtyCloseSandbox:
    def __init__(self, key, secret, coin_x, coin_y, qty_y, beta=0.5):
        self.coin_x = coin_x
        self.coin_y = coin_y
        self.qty_y = qty_y
        self.beta = beta
        self.bybit = BybitService(key=key, secret=secret, testnet=False, demo=True)
        self.rules = {}
        self.logger = TradeLogger()

    async def setup(self):
        self.rules = await self.bybit.get_instruments_info([self.coin_x, self.coin_y])
        if not self.rules:
            raise RuntimeError("sandbox: failed to fetch instrument rules")
        print(f"sandbox: rules -> {self.rules}")

    def _round_qty(self, qty, symbol):
        step, min_qty = self.rules.get(symbol, ("0.001", "0.001"))
        step_d = Decimal(str(step))
        min_qty_d = Decimal(str(min_qty))
        q = (Decimal(str(qty)) / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
        if q < min_qty_d:
            q = min_qty_d
        return float(q.normalize())

    async def _positions(self):
        status = self.bybit.get_account_status()
        positions = status.get("positions", []) if status else []
        return {
            p["symbol"]: {
                "side": p["side"],
                "size": float(p["size"]),
                "avgPrice": float(p["avgPrice"]) if p.get("avgPrice") else 0.0,
            }
            for p in positions
            if float(p["size"]) > 0
        }

    async def _market(self, symbol, side, qty, reduce_only=False):
        print(f"sandbox: {side} {qty} {symbol} reduce_only={reduce_only}")
        try:
            res = await asyncio.to_thread(
                self.bybit.session.place_order,
                category="linear",
                symbol=symbol,
                side=side,
                qty=str(qty),
                timeInForce="IOC",
                orderType="Market",
                reduceOnly=reduce_only,
            )
            print(f"sandbox: order response -> {res}")
            return res
        except Exception as e:
            print(f"sandbox: order error {e}")
            return None

    async def _flatten(self, positions):
        """Close any existing positions before the test starts."""
        if not positions:
            return True
        print(f"sandbox: flattening pre-existing positions -> {positions}")
        tasks = []
        for sym, p in positions.items():
            close_side = "Sell" if p["side"] == "Buy" else "Buy"
            tasks.append(self._market(sym, close_side, p["size"], reduce_only=True))
        await asyncio.gather(*tasks)
        await asyncio.sleep(2)
        remaining = await self._positions()
        if remaining:
            print(f"sandbox: CRITICAL could not flatten positions -> {remaining}")
            return False
        print("sandbox: pre-test flatten ok")
        return True

    async def run(self):
        await self.setup()

        prices = await self.bybit.get_current_price([self.coin_x, self.coin_y])
        if prices is None:
            raise RuntimeError("sandbox: failed to fetch prices")
        px = prices["prices"][self.coin_x]
        py = prices["prices"][self.coin_y]
        print(f"sandbox: prices {self.coin_x}={px:.4f} {self.coin_y}={py:.4f}")

        # ---- compute sizes, mirroring executor.py ----
        raw_qty_x = abs(self.beta) * self.qty_y * (py / px)
        qty_x = self._round_qty(raw_qty_x, self.coin_x)
        qty_y = self._round_qty(self.qty_y, self.coin_y)

        val_x = qty_x * px
        val_y = qty_y * py
        print(f"sandbox: raw qty_x={qty_x} (${val_x:.2f}) qty_y={qty_y} (${val_y:.2f})")

        if val_x < 5.5 or val_y < 5.5:
            raw_val_x = raw_qty_x * px
            raw_val_y = self.qty_y * py
            scale = max(
                6.0 / raw_val_x if raw_val_x > 0 else 1.0,
                6.0 / raw_val_y if raw_val_y > 0 else 1.0,
            )
            qty_x = self._round_qty(raw_qty_x * scale, self.coin_x)
            qty_y = self._round_qty(self.qty_y * scale, self.coin_y)
            val_x = qty_x * px
            val_y = qty_y * py
            print(f"sandbox: pumped qty_x={qty_x} (${val_x:.2f}) qty_y={qty_y} (${val_y:.2f})")

        # ---- start flat ----
        before = await self._positions()
        if not await self._flatten(before):
            raise RuntimeError("sandbox: cannot start flat")

        # ---- entry: short y, long x ----
        print("sandbox: --- ENTRY ---")
        y_order, x_order = await asyncio.gather(
            self._market(self.coin_y, "Sell", qty_y, reduce_only=False),
            self._market(self.coin_x, "Buy", qty_x, reduce_only=False),
        )
        await asyncio.sleep(3)

        entry_pos = await self._positions()
        print(f"sandbox: positions after entry -> {entry_pos}")

        y_filled = entry_pos.get(self.coin_y, {}).get("size", 0.0)
        x_filled = entry_pos.get(self.coin_x, {}).get("size", 0.0)

        if y_filled == 0.0 or x_filled == 0.0:
            print("sandbox: CRITICAL one leg did not fill; unwinding filled leg(s)")
            unwind_tasks = []
            if y_filled:
                unwind_tasks.append(self._market(self.coin_y, "Buy", y_filled, reduce_only=True))
            if x_filled:
                unwind_tasks.append(self._market(self.coin_x, "Sell", x_filled, reduce_only=True))
            if unwind_tasks:
                await asyncio.gather(*unwind_tasks)
            raise RuntimeError("sandbox: entry incomplete, unwound and aborted")

        # ---- exit exact filled size ----
        print("sandbox: --- EXIT ---")
        await asyncio.gather(
            self._market(self.coin_y, "Buy", y_filled, reduce_only=True),
            self._market(self.coin_x, "Sell", x_filled, reduce_only=True),
        )
        await asyncio.sleep(3)

        after = await self._positions()
        print(f"sandbox: positions after exit -> {after}")

        if after:
            print(f"sandbox: FAIL - leftover positions -> {after}")
            return False

        print("sandbox: PASS - flat after round-trip")
        return True


async def main():
    load_dotenv()
    key = os.getenv("KEY")
    secret = os.getenv("SECRET")
    if not key or not secret:
        print("sandbox: set KEY and SECRET in .env")
        sys.exit(1)

    coin_x = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    coin_y = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
    qty_y = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0001

    print(f"sandbox: testing pair {coin_x}/{coin_y} qty_y={qty_y}")
    ok = await QtyCloseSandbox(key, secret, coin_x, coin_y, qty_y).run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
