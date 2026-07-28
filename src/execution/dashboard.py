from datetime import datetime

class LiveDashboard:
    def __init__(self, title="StatArb Trading Bot"):
        self.title = title

    def start(self):
        print(f"=== Started {self.title} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    def stop(self):
        print(f"=== Stopped {self.title} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    def update(self, state_name, z_score, beta, is_trade_stat, is_structure_stat, target_pair, raw_x, raw_y, ev, dynamic_entry=1.5, dynamic_stoploss=4.0):
        stat_t_icon = "✅" if is_trade_stat else "❌"
        stat_s_icon = "✅" if is_structure_stat else "❌"
        
        # Safely format numbers that might be None
        ev_str = f"{ev:.4f}" if ev is not None else "N/A"
        z_str = f"{z_score:.3f}" if z_score is not None else "-"
        p_x = f"{raw_x:.4f}" if raw_x is not None else "-"
        p_y = f"{raw_y:.4f}" if raw_y is not None else "-"
        beta_str = f"{beta:.4f}" if beta is not None else "-"
        
        target_z = dynamic_entry if state_name in ("NoPosition", None) else dynamic_stoploss
        target_z_str = f"{target_z:.3f}" if target_z is not None else "-"

        timestamp = datetime.now().strftime('%H:%M:%S')
        
        output = (
            f"[{timestamp}] {target_pair} | P(X/Y): {p_x}/{p_y} | "
            f"State: {state_name} | Stat(T/S): {stat_t_icon}/{stat_s_icon} | Beta: {beta_str} | "
            f"Z: {z_str} (Target: {target_z_str}) | EV: {ev_str} USDT"
        )
        print(output)
