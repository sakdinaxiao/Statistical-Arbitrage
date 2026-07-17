from dataclasses import dataclass, field
import time
from data.enums import Action

@dataclass
class Signal:
    action : Action
    z_score : float
    beta: float
    coin_x : str
    coin_y : str
    price_x : float = 0.0          # raw (un-logged) price of coin_x, for hedge sizing
    price_y : float = 0.0          # raw (un-logged) price of coin_y, for hedge sizing
    reason : str = ""
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())