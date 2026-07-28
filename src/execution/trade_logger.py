import csv
import os
from datetime import datetime

from data.enums import Action

HEADER = [
    "timestamp", "action", "coin_x", "coin_y", "z_score", "beta",
    "price_x", "price_y", "qty_x", "qty_y", "y_ok", "x_ok", "status", "reason", "result"
]

EXITS = (Action.EXIT_PROFIT, Action.EXIT_LOSS, Action.EXIT_REGIME)


class TradeLogger:
    """Append-only CSV of placed orders + gate-blocked entries. One file per run."""

    def __init__(self, coin_x, coin_y):
        now = datetime.now()
        log_dir = f"logs/{now:%Y-%m-%d}"
        os.makedirs(log_dir, exist_ok=True)
        self.path = f"{log_dir}/trades_{coin_x}_{coin_y}_{now:%H%M%S}.csv"
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
        self._entry = None  # (action, price_x, price_y, qty_x, qty_y) of last filled entry

    def _row(self, payload, qty_x, qty_y, y_ok, x_ok, status, reason, result=""):
        # logging must never break the trading loop: swallow any write error.
        try:
            # Format as clean human-readable local time (matches the dashboard)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts, payload.action.value, payload.coin_x, payload.coin_y,
                    payload.z_score, payload.beta, payload.price_x, payload.price_y,
                    qty_x, qty_y, y_ok, x_ok, status, reason, result
                ])
        except Exception as e:
            print(f"trade_logger: failed to write row: {e}")

    def _judge(self, payload):
        # combined pnl of both legs: long leg gains when price rises, short when it falls
        action, price_x, price_y, qty_x, qty_y = self._entry
        x_sign = -1 if action == Action.SX_LY else 1  # SX_LY = short X long Y
        pnl = (x_sign * qty_x * (payload.price_x - price_x)
               - x_sign * qty_y * (payload.price_y - price_y))
        return "win" if pnl > 0 else "lose"

    def log_order(self, payload, qty_x, qty_y, y_ok, x_ok):
        status = "filled" if (y_ok and x_ok) else "failed"
        result = ""
        if status == "filled":
            if payload.action in (Action.SY_LX, Action.SX_LY):
                self._entry = (payload.action, payload.price_x, payload.price_y, qty_x, qty_y)
            elif payload.action in EXITS and self._entry is not None:
                result = self._judge(payload)
                self._entry = None
        self._row(payload, qty_x, qty_y, y_ok, x_ok, status, payload.reason, result)

    def log_blocked(self, payload, reason):
        self._row(payload, "", "", "", "", "blocked", reason)

    def log_candle(self, payload, risk):
        # one row per candle the risk gate ran on, incl. when it denies entry.
        status = "risk_ok" if risk.allow else "risk_block"
        self._row(payload, "", "", "", "", status, risk.reason)
