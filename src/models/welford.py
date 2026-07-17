import numpy as np


class WelfordZScore:
    def __init__(self,window,initx,inity,initalpha,initbeta):
        self.window = window
        self.s_array = np.zeros(window)
        self.index = 0
        self.mean = 0
        self.M2 = 0
        self.z_score = 0.0

        self.frozen_mean = 0.0
        self.frozen_std = 0.0

        if len(initx) != self.window or len(inity) != self.window:
            print(f"Error: Initialization arrays must be of length {self.window}")
            return None #decide what to return to bvreak initialize obj in main
        
        for i in range(len(self.s_array)):
            spread = inity[i] - ((initbeta*initx[i]) + initalpha)
            self.s_array[i] = spread  

        self.mean = np.mean(self.s_array)
        distance = (self.s_array - self.mean) ** 2
        self.M2 = np.sum(distance)   


        variance = max(0.0, self.M2 / self.window)
        std = variance ** 0.5
        if std == 0.0:
            self.z_score = 0.0
        else:
            self.z_score = (spread - self.mean) / std

        self.frozen_mean = self.mean
        self.frozen_std = std     

    def z_score_cal(self,x,y,alpha,beta,hold=False):
        if np.isnan(x) or np.isnan(y) or np.isnan(alpha) or np.isnan(beta):
            return self.z_score, np.nan
        else:

            #WelFord online -- keep the rolling window live so half-life stays fresh
            spread = y - ((beta*x) + alpha)
            old_value = self.s_array[self.index]

            self.s_array[self.index] = spread

            old_mean = self.mean
            self.mean = old_mean + (spread - old_value) / self.window

            new_distance = (spread - old_mean) * (spread - self.mean)
            old_distance = (old_value - old_mean) * (old_value - self.mean)

            self.M2 = self.M2 + new_distance - old_distance 

            variance = max(0.0, self.M2 / self.window)
            std = variance ** 0.5 

            self.index = (self.index + 1) %self.window

            if hold:
                # position open: measure z against the reference frozen at entry,
                # so the rolling mean can't chase the spread into a false z=0 exit
                ref_mean, ref_std = self.frozen_mean, self.frozen_std
            else:
                # flat: live reference, snapshot it so it's right the instant we enter
                ref_mean, ref_std = self.mean, std
                self.frozen_mean, self.frozen_std = self.mean, std

            if ref_std == 0.0: 
                self.z_score = 0.0
            else:
                self.z_score = (spread - ref_mean) / ref_std

        return self.z_score
    
    def get_spread_series(self):
        # ring buffer  so half-life can regress it
        return np.concatenate([self.s_array[self.index:], self.s_array[:self.index]])




    