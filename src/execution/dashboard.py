CLEAR = "\033[2J\033[H"
import sys
from collections import deque
from datetime import datetime

class LiveDashboard:
    def __init__(self, title="StatArb Trading Bot", history_size=15):
        self.title = title
        self.last_state = None

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, state_name, z_score, beta, is_stat, target_pair, raw_x, raw_y, ev, pnl=None):
        stat_icon = "✅ Yes" if is_stat else "❌ No"
        ev_str = f"{ev:.4f} USDT" if ev is not None else "N/A"
        pnl_str = f"{pnl:+.4f} USDT" if pnl is not None else "N/A"
        
        now = datetime.now().strftime("%H:%M:%S")
        history_line = f"[{now}] X: {raw_x:<8.4f} | Y: {raw_y:<8.4f} | Z: {z_score:<6.3f} | EV: {ev_str}"
        
        if self.last_state != state_name:
            lines = []
            lines.append("=" * 60)
            lines.append(f" {self.title}")
            lines.append("-" * 60)
            lines.append(f" Target Pair    : {target_pair}")
            lines.append(f" Position State : {state_name}")
            lines.append(f" Realized PnL   : {pnl_str}")
            lines.append(f" Z-Score        : {z_score:.3f}")
            lines.append(f" Beta           : {beta:.3f}")
            lines.append(f" Stationary     : {stat_icon}")
            lines.append(f" Expected Value : {ev_str}")
            lines.append("-" * 60)
            lines.append(" Price & EV Stream (Most Recent Last):")
            print(CLEAR + "\n".join(lines))
            self.last_state = state_name
            
        print(f"  {history_line}")
