from enum import Enum

class Action(Enum):
    HOLD = "HOLD" #0
    SY_LX = "SHORT Y LONG X" #0 to 1 zScore is positive
    SX_LY = "SHORT X LONG Y" # 0 to -1 zScore is negative
    EXIT_PROFIT = "EXIT PROFIT" # 1 or -1 to 0
    EXIT_LOSS = "EXIT BY STOPLOSS"
    EXIT_REGIME = "EXIT BY REGIME BREAK"   # pair stopped cointegrating, force-close
    INVALID = "INVALID"

class State(Enum):
    NoPosition = 0
    short_y = 1
    long_y = -1
