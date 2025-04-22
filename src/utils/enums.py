from enum import Enum

class WeightsApplication(str, Enum):
    NUMBER = "number"
    WEALTH = "wealth"

class WeightsMethod(str, Enum):
    DOLLAR = "dollar"
    SHARE = "share"

class NaNHandling(str, Enum):
    DROP = "drop"
    KEEP = "keep"
    FILL = "fill"
    ZERO = "zero"