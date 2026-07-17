from data.dataclasses import Signal
from data.enums import Action, State
import numpy as np 

class StatArbStrategy:
    def __init__(self,coin_x,coin_y):
        self.entry = 1.5
        self.stoploss = 4.0 #tune these 3 for real market
        self.profit = 0.0
        
        self.coin_x = coin_x
        self.coin_y = coin_y
    
        
    
    def create_signal(self,z_score,beta,state,price_x=0.0,price_y=0.0):
        if np.isnan(z_score) or np.isnan(beta):
            return Signal(
                action=Action.INVALID,
                z_score=z_score,
                beta=beta,
                coin_x=self.coin_x,
                coin_y=self.coin_y,
                price_x=price_x,
                price_y=price_y,
                reason= "Invalid number"
            )
        
        #hold
        action = Action.HOLD
        reason = f"holding: z={z_score:.3f}, state={state.name}"

        if state == State.NoPosition: #look for entry
            if self.stoploss > z_score >= self.entry: 
                action = Action.SY_LX
                reason = f"z_score: {z_score} is more than {self.entry}"
            elif -self.stoploss < z_score <= -self.entry:
                action = Action.SX_LY
                reason = f"z_score: {z_score} is less than {-self.entry}"
            
        elif state == State.short_y: #on short postion
            if z_score >= self.stoploss: #stoploss case
                action = Action.EXIT_LOSS
                reason = f"z_score: {z_score} is hitting {self.stoploss}"

            elif z_score <= self.profit: # z hit profit
                action = Action.EXIT_PROFIT
                reason = f"z_score: {z_score} is hitting {self.profit}"

        elif state == State.long_y: # on long position
            if -self.stoploss >= z_score:
                action = Action.EXIT_LOSS
                reason = f"z_score: {z_score} is hitting {-self.stoploss}"

            elif z_score >= self.profit:
                action = Action.EXIT_PROFIT
                reason = f"z_score: {z_score} is hitting {self.profit}"

        return Signal(
            action=action,
            z_score=z_score,
            beta=beta,
            coin_x=self.coin_x,
            coin_y=self.coin_y,
            price_x=price_x,
            price_y=price_y,
            reason=reason
        )