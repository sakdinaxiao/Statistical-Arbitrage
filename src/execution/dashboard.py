from rich.live import Live
from rich.table import Table
from datetime import datetime

class LiveDashboard:
    def __init__(self, title="StatArb Trading Bot"):
        self.title = title
        self.live = None

    def start(self):
        self.live = Live(refresh_per_second=4)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def update(self, state_name, z_score, beta, is_stat, target_pair, raw_x, raw_y, ev, realized_pnl=None, unrealized_pnl=None, dynamic_entry=1.5, dynamic_stoploss=4.0):
        if not self.live:
            return

        stat_icon = "✅ Yes" if is_stat else "❌ No"
        
        # Safely format numbers that might be None
        ev_str = f"{ev:.4f} USDT" if ev is not None else "N/A"
        z_str = f"{z_score:.3f}" if z_score is not None else "-"
        p_x = f"{raw_x:.4f}" if raw_x is not None else "-"
        p_y = f"{raw_y:.4f}" if raw_y is not None else "-"
        beta_str = f"{beta:.4f}" if beta is not None else "-"
        rpnl_str = f"{realized_pnl:.4f} USDT" if realized_pnl is not None else "-"
        upnl_str = f"{unrealized_pnl:.4f} USDT" if unrealized_pnl is not None else "-"
        
        target_z = dynamic_entry if state_name in ("NoPosition", None) else dynamic_stoploss
        target_z_str = f"{target_z:.3f}" if target_z is not None else "-"

        table = Table(title=f"{self.title} - {datetime.now().strftime('%H:%M:%S')}")
        table.add_column("Pair", style="cyan")
        table.add_column("Prices (X/Y)", style="magenta")
        table.add_column("State", style="yellow")
        table.add_column("Stationary")
        table.add_column("Beta", justify="right")
        table.add_column("Current Z", justify="right")
        table.add_column("Target Z", justify="right")
        table.add_column("EV", justify="right")
        table.add_column("uPnL", justify="right", style="blue")
        table.add_column("rPnL", justify="right", style="green")

        table.add_row(
            target_pair,
            f"{p_x} / {p_y}",
            str(state_name),
            stat_icon,
            beta_str,
            z_str,
            target_z_str,
            ev_str,
            upnl_str,
            rpnl_str
        )

        self.live.update(table)
