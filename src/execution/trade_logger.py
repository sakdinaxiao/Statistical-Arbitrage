import csv
import os
from datetime import datetime

from data.enums import Action

HEADER = [
    "timestamp", "action", "coin_x", "coin_y", "z_score", "beta",
    "price_x", "price_y", "qty_x", "qty_y", "y_ok", "x_ok", "status", "reason", "result", "pnl"
]

EXITS = (Action.EXIT_PROFIT, Action.EXIT_LOSS, Action.EXIT_REGIME)


class TradeLogger:
    """Append-only CSV of placed orders + gate-blocked entries. One file per pair per day."""

    def __init__(self, coin_x, coin_y, feerate):
        self.coin_x = coin_x
        self.coin_y = coin_y
        self.feerate = feerate
        self._entry = None  # (action, price_x, price_y, qty_x, qty_y) of last filled entry

    def _path(self):
        # resolved per write so a run crossing midnight rolls into the new day's file
        now = datetime.now()
        log_dir = f"logs/{now:%Y-%m-%d}"
        os.makedirs(log_dir, exist_ok=True)
        return f"{log_dir}/trades_{self.coin_x}_{self.coin_y}.csv"

    def _row(self, payload, qty_x, qty_y, y_ok, x_ok, status, reason, result="", pnl=""):
        # logging must never break the trading loop: swallow any write error.
        try:
            path = self._path()
            # Format as clean human-readable local time (matches the dashboard)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(HEADER)
                w.writerow([
                    ts, payload.action.value, payload.coin_x, payload.coin_y,
                    payload.z_score, payload.beta, payload.price_x, payload.price_y,
                    qty_x, qty_y, y_ok, x_ok, status, reason, result, pnl
                ])
        except Exception as e:
            print(f"trade_logger: failed to write row: {e}")

    def _close_pnl(self, payload):
        # combined pnl of both legs: long leg gains when price rises, short when it falls
        action, price_x, price_y, qty_x, qty_y = self._entry
        x_sign = -1 if action == Action.SX_LY else 1  # SX_LY = short X long Y
        gross = (x_sign * qty_x * (payload.price_x - price_x)
                 - x_sign * qty_y * (payload.price_y - price_y))
        # taker fee on both legs, entry and exit
        fees = self.feerate * (qty_x * (price_x + payload.price_x)
                               + qty_y * (price_y + payload.price_y))
        pnl = gross - fees
        return ("win" if pnl > 0 else "lose"), round(pnl, 6)

    def log_order(self, payload, qty_x, qty_y, y_ok, x_ok):
        status = "filled" if (y_ok and x_ok) else "failed"
        result = ""
        pnl = ""
        if status == "filled":
            if payload.action in (Action.SY_LX, Action.SX_LY):
                self._entry = (payload.action, payload.price_x, payload.price_y, qty_x, qty_y)
            elif payload.action in EXITS and self._entry is not None:
                result, pnl = self._close_pnl(payload)
                self._entry = None
        self._row(payload, qty_x, qty_y, y_ok, x_ok, status, payload.reason, result, pnl)

    def log_blocked(self, payload, reason):
        self._row(payload, "", "", "", "", "blocked", reason)

    def log_candle(self, payload, risk):
        # one row per candle the risk gate ran on, incl. when it denies entry.
        status = "risk_ok" if risk.allow else "risk_block"
        self._row(payload, "", "", "", "", status, risk.reason)
