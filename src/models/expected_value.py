import numpy as np


class ExpectedValueCalculator:
    def __init__(self, qty_y, fee_rate, max_bars=160):
        # qty_y is the SAME base size the executor trades, so the EV is real
        self.qty_y = qty_y
        self.fee_rate = fee_rate
        self.max_bars = max_bars   # 60 bars * 5min = 5h, skip trades slower than this

    def half_life(self, spread_series):
        # AR(1) on the spread: delta = lam*level + c. lam < 0 means mean reverting.
        s = np.asarray(spread_series, dtype=float)
        if len(s) < 2:
            return None

        lag = s[:-1]
        delta = s[1:] - lag

        lam, c = np.polyfit(lag, delta, 1)
        if lam >= 0:
            return None   # spread is drifting, not coming back

        return -np.log(2) / lam

    def assess(self, z_score, beta, price_y, spread_series) -> bool:
        # gate 0: missing data guards
        if z_score is None or beta is None or price_y is None:
            return False
        if np.isnan(z_score) or np.isnan(beta):
            return False

        hl = self.half_life(spread_series)
        
        # gate 1: must revert, and revert fast enough (checked early to avoid np.std warnings)
        if hl is None or np.isnan(hl) or hl <= 0 or hl > self.max_bars:
            return False

        # everything in dollars so profit and fees are comparable.
        # position is dollar-hedged, so a spread move of (z*sigma) earns notional_y*z*sigma.
        sigma = float(np.std(spread_series))
        notional_y = self.qty_y * price_y
        expected_profit = abs(z_score) * sigma * notional_y

        # 2 legs (notional_x = beta*notional_y), entry + exit = the *2
        cost = self.fee_rate * notional_y * (1 + abs(beta)) * 2
        ev = expected_profit - cost

        # gate 2: profit must beat the fees
        if ev <= 0:
            return False, ev

        return True , ev

        