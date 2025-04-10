from enum import Enum

class WeightsMethod(str, Enum):
    STOCKS = "stocks"
    WEALTH = "wealth"

class NaNHandling(str, Enum):
    DROP = "drop"
    KEEP = "keep"
    FILL = "fill"