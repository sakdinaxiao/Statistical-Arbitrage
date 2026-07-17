from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
from collections import deque

class Cointegrate:
    def __init__(self,timeFrame,past_x,past_y,retest_interval_hours=2):
        self.stationary_flag = False

        self.past_x = past_x
        self.past_y = past_y

        self.restest_x = deque(past_x, maxlen=len(past_x))
        self.restest_y = deque(past_y,maxlen=len(past_y))

        per_hr = 60/(timeFrame/60)
        self.retest_ticks = per_hr * retest_interval_hours

        self.counter = 0

        
    def update(self, new_x, new_y,max_bars):
        self.restest_x.append(new_x)
        self.restest_y.append(new_y)
        self.counter += 1
        
        if not self.stationary_flag or self.counter >= self.retest_ticks:
            list_x = list(self.restest_x)
            list_y = list(self.restest_y)
            self.spread_stationaryTest(list_x[-max_bars:],list_y[-max_bars:])

    def force_retest(self,max_bars):
        list_x = list(self.restest_x)
        list_y = list(self.restest_y)

        self.spread_stationaryTest(list_x[-max_bars:],list_y[-max_bars:])

    def spread_stationaryTest(self,datax,datay):
        self.counter = 0

        # returns the ADF p-value on the spread; caller bands it (e.g. <0.05 healthy)
        if len(datax) != len(datay):
            print("data amout is not the same")
            return 1.0

        ols_model = sm.OLS(datay, sm.add_constant(datax)).fit()

        alpha = ols_model.params[0]
        beta = ols_model.params[1]

        spread = np.array(datay) - ((beta * np.array(datax)) + alpha)

        result = adfuller(spread)
        p_value = result[1]

        if p_value < 0.05:
            self.stationary_flag = True
        else:
            self.stationary_flag = False
