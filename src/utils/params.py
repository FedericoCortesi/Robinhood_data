from dataclasses import dataclass, field
from typing import Optional, Set

@dataclass
class ReturnParams:
    horizons: Set[int] = field(default_factory=lambda: {5, 15, 30, 60, 120})
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cumulative: bool = True
    append_start: bool = False
