import csv
import os
from datetime import datetime

HEADER = [
    "timestamp", "action", "coin_x", "coin_y", "z_score", "beta",
    "price_x", "price_y", "qty_x", "qty_y", "y_ok", "x_ok", "status", "reason",
    "realized_pnl"
]


class TradeLogger:
    """Append-only CSV of placed orders + gate-blocked entries. One file per run."""

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.path = f"logs/trades_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

    def _row(self, payload, qty_x, qty_y, y_ok, x_ok, status, reason, pnl=""):
        # logging must never break the trading loop: swallow any write error.
        try:
            ts = datetime.fromtimestamp(payload.timestamp_ns / 1e9).isoformat()
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts, payload.action.value, payload.coin_x, payload.coin_y,
                    payload.z_score, payload.beta, payload.price_x, payload.price_y,
                    qty_x, qty_y, y_ok, x_ok, status, reason, pnl
                ])
        except Exception as e:
            print(f"trade_logger: failed to write row: {e}")

    def log_order(self, payload, qty_x, qty_y, y_ok, x_ok, pnl=""):
        status = "filled" if (y_ok and x_ok) else "failed"
        self._row(payload, qty_x, qty_y, y_ok, x_ok, status, payload.reason, pnl)

    def log_blocked(self, payload, reason):
        self._row(payload, "", "", "", "", "blocked", reason)

    def log_candle(self, payload, risk):
        # one row per candle the risk gate ran on, incl. when it denies entry.
        status = "risk_ok" if risk.allow else "risk_block"
        self._row(payload, "", "", "", "", status, risk.reason)
