from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
from collections import deque

class Cointegrate:
    def __init__(self,timeFrame,past_x,past_y,retest_trade_hours=2,retest_structure_hours=2):
        # structure = long-term relationship (14d), trade = short-term stability (8h)
        self.structure_stationary_flag = False
        self.trade_stationary_flag = False

        self.past_x = past_x
        self.past_y = past_y

        self.restest_x = deque(past_x, maxlen=len(past_x))
        self.restest_y = deque(past_y,maxlen=len(past_y))

        per_hr = 60/(timeFrame/60)
        self.retest_ticks_trade = per_hr * retest_trade_hours
        self.retest_ticks_structure = per_hr * retest_structure_hours

        self.counter_trade = 0
        self.counter_structure = 0

        
    def update(self, new_x, new_y, max_bars_trade, max_bars_structure):
        self.restest_x.append(new_x)
        self.restest_y.append(new_y)
        self.counter_trade += 1
        self.counter_structure += 1
        
        if not self.trade_stationary_flag or self.counter_trade >= self.retest_ticks_trade:
            list_x = list(self.restest_x)
            list_y = list(self.restest_y)
            self.trade_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_trade:],list_y[-max_bars_trade:])
            self.counter_trade = 0

        if not self.structure_stationary_flag or self.counter_structure >= self.retest_ticks_structure:
            list_x = list(self.restest_x)
            list_y = list(self.restest_y)
            self.structure_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_structure:],list_y[-max_bars_structure:])
            self.counter_structure = 0

    def force_retest(self, max_bars_trade, max_bars_structure):
        list_x = list(self.restest_x)
        list_y = list(self.restest_y)

        self.trade_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_trade:],list_y[-max_bars_trade:])
        self.counter_trade = 0

        self.structure_stationary_flag = self.spread_stationaryTest(list_x[-max_bars_structure:],list_y[-max_bars_structure:])
        self.counter_structure = 0

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
