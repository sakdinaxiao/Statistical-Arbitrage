import numpy as np


class ExpectedValueCalculator:
    def __init__(self, qty_y, fee_rate, max_bars=160):
        # qty_y is the SAME base size the executor trades, so the EV is real
        self.qty_y = qty_y
        self.fee_rate = fee_rate
        self.max_bars = max_bars   # 60 bars * 5min = 5h, skip trades slower than this
        self.halflife = None

    def cal_half_life(self, spread_series):
        # AR(1) on the spread: delta = lam*level + c. lam < 0 means mean reverting.
        self.halflife = None
        s = np.asarray(spread_series, dtype=float)
        if len(s) < 3 or not np.all(np.isfinite(s)):
            return None

        lag = s[:-1]
        delta = s[1:] - lag
        if np.ptp(lag) == 0:
            return None

        lam, c = np.polyfit(lag, delta, 1)
        if not np.isfinite(lam) or lam >= 0:
            return None   # spread is drifting, not coming back

        self.halflife = -np.log(2) / lam
        return self.halflife

    def assess(self, z_score, beta, price_y, spread_series) -> bool:
        # gate 0: missing data guards
        if z_score is None or beta is None or price_y is None:
            return False, 0
        if not np.all(np.isfinite([z_score, beta, price_y])) or price_y <= 0:
            return False, 0

        hl = self.halflife
        
        # gate 1: must revert, and revert fast enough (checked early to avoid np.std warnings)
        if hl is None or np.isnan(hl) or hl <= 0 or hl > self.max_bars:
            return False, 0

        # everything in dollars so profit and fees are comparable.
        # position is dollar-hedged, so a spread move of (z*sigma) earns notional_y*z*sigma.
        sigma = float(np.std(spread_series))
        notional_y = self.qty_y * price_y
        expected_profit = abs(z_score) * sigma * notional_y

        # 2 legs (notional_x = beta*notional_y), entry + exit = the *2
        cost = self.fee_rate * notional_y * (1 + abs(beta)) * 2
        ev = expected_profit - cost

        # gate 2: profit must beat the fees
        if not np.isfinite(ev) or ev <= 0:
            return False, ev

        return True , ev
