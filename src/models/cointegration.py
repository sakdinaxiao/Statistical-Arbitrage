from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
from collections import deque

class Cointegrate:
    def __init__(self,timeFrame,past_x,past_y):
        # structure = long-term relationship (1mo), trade = short-term stability (8h)
        self.structure_stationary_flag = False
        self.trade_stationary_flag = False

        self.past_x = past_x
        self.past_y = past_y

        self.restest_x = deque(past_x, maxlen=len(past_x))
        self.restest_y = deque(past_y,maxlen=len(past_y))

        per_hr = 60/(timeFrame/60)
        # flat-state cadence: once per hour; while holding, update() targets 3 * half-life instead
        self.retest_ticks_flat = per_hr

        self.counter_trade = 0
        self.counter_structure = 0
        # bars since the last stationarity test of either window; drives the danger-zone retest in main
        self.danger_counter = 0

        
    def update(self, new_x, new_y, max_bars_trade, max_bars_structure, is_holding=False, half_life_bars=None):
        self.restest_x.append(new_x)
        self.restest_y.append(new_y)
        self.counter_trade += 1
        self.counter_structure += 1
        self.danger_counter += 1

        # dynamic cadence: 1h flat, 3 * half-life while holding; bad half-life falls back to 1h
        retest_ticks = self.retest_ticks_flat
        if is_holding and half_life_bars is not None and np.isfinite(half_life_bars) and half_life_bars > 0:
            retest_ticks = 3 * half_life_bars

        list_x = list(self.restest_x)
        list_y = list(self.restest_y)

        # trade test short-circuits on a False flag so a healed spread is caught instantly
        if not self.trade_stationary_flag or self.counter_trade >= retest_ticks:
            self.trade_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_trade:],list_y[-max_bars_trade:])
            self.counter_trade = 0
            self.danger_counter = 0

        # structure test is locked to the timer; a 1-month baseline cannot heal in minutes
        if self.counter_structure >= retest_ticks:
            self.structure_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_structure:],list_y[-max_bars_structure:])
            self.counter_structure = 0
            self.danger_counter = 0

    def force_retest(self, max_bars_trade, max_bars_structure):
        list_x = list(self.restest_x)
        list_y = list(self.restest_y)

        self.trade_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_trade:],list_y[-max_bars_trade:])
        self.counter_trade = 0

        self.structure_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_structure:],list_y[-max_bars_structure:])
        self.counter_structure = 0
        self.danger_counter = 0

    def spread_stationaryTest(self,datax,datay):
        # returns True when the ADF p-value on the spread is < 0.05; caller assigns the flag
        if len(datax) != len(datay):
            print("data amout is not the same")
            return False

        ols_model = sm.OLS(datay, sm.add_constant(datax)).fit()

        alpha = ols_model.params[0]
        beta = ols_model.params[1]

        spread = np.array(datay) - ((beta * np.array(datax)) + alpha)

        result = adfuller(spread)
        p_value = result[1]
        

        return p_value < 0.05
