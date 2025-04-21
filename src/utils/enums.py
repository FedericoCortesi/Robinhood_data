from enum import Enum

class WeightsMethod(str, Enum):
    NUMBER = "number"
    WEALTH = "wealth"

class NaNHandling(str, Enum):
    DROP = "drop"
    KEEP = "keep"
    FILL = "fill"
    ZERO = "zero"