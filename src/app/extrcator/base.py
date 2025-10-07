from __future__ import annotations
from typing import Iterable, Literal, Dict, Any, List
from dataclasses import dataclass
from datetime import date

Interval = Literal["daily"]

@dataclass
class Candle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class DataProvider:
    name: str = "base"

    def historical_prices(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        interval: Interval = "daily",
    ) -> Iterable[Candle]:
        raise NotImplementedError

def to_records(candles: Iterable[Candle]) -> List[Dict[str, Any]]:
    return [c.__dict__ for c in candles]

