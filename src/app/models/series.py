from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Tuple
from datetime import datetime
from statistics import mean, pstdev
import math

from app.extractor.base import Candle

# -----------------------------
# Helpers internos
# -----------------------------
def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")

def _sorted_unique_candles(candles: Iterable[Candle]) -> List[Candle]:
    by_date: Dict[str, Candle] = {}
    for c in candles:
        ds = str(c.date)[:10]
        by_date[ds] = Candle(
            date=ds,
            open=float(c.open),
            high=float(c.high),
            low=float(c.low),
            close=float(c.close),
            volume=float(c.volume),
        )
    return [by_date[ds] for ds in sorted(by_date.keys(), key=_parse_date)]

def _ffill_closes(candles: List[Candle]) -> List[Candle]:
    prev_close: Optional[float] = None
    out: List[Candle] = []
    for c in candles:
        close = c.close if c.close is not None else prev_close
        if close is None:
            out.append(c)
        else:
            out.append(Candle(c.date, c.open, c.high, c.low, close, c.volume))
            prev_close = close
        prev_close = out[-1].close
    return out

def _drop_na_head(candles: List[Candle]) -> List[Candle]:
    out: List[Candle] = []
    seen_valid = False
    for c in candles:
        if c.close is not None and not math.isnan(c.close):
            seen_valid = True
        if seen_valid:
            out.append(c)
    return out

# -----------------------------
# Serie de precios
# -----------------------------
@dataclass
class PriceSeries:
    provider: str
    datatype: str   # "prices" o "fx"
    symbol: str
    candles: List[Candle] = field(default_factory=list)

    # Métricas calculadas automáticamente
    close_mean: Optional[float] = field(init=False, default=None)
    close_std: Optional[float] = field(init=False, default=None)
    ret_mean: Optional[float] = field(init=False, default=None)
    ret_std: Optional[float] = field(init=False, default=None)
    n_obs: int = field(init=False, default=0)

    # Config limpieza
    forward_fill: bool = True
    use_log_returns: bool = True  # ✅ retornos log por defecto

    def __post_init__(self):
        self.candles = _sorted_unique_candles(self.candles)
        if self.forward_fill:
            self.candles = _ffill_closes(self.candles)
        self.candles = _drop_na_head(self.candles)
        self.n_obs = len(self.candles)
        self._recompute_stats()

    # --------------------------
    # Core accessors
    # --------------------------
    def closes(self) -> List[float]:
        return [float(c.close) for c in self.candles if c.close is not None]

    def dates(self) -> List[str]:
        return [c.date for c in self.candles]

    def returns(self) -> List[float]:
        cls = self.closes()
        if len(cls) < 2:
            return []
        rets: List[float] = []
        prev = cls[0]
        for x in cls[1:]:
            if prev == 0:
                rets.append(0.0)
            else:
                if self.use_log_returns:
                    rets.append(math.log(x / prev))
                else:
                    rets.append(x / prev - 1.0)
            prev = x
        return rets

    def returns_with_dates(self) -> List[Tuple[str, float]]:
        ds = self.dates()
        rs = self.returns()
        return list(zip(ds[1:], rs))

    # --------------------------
    # Stats auto
    # --------------------------
    def _recompute_stats(self):
        cls = self.closes()
        if len(cls) >= 1:
            self.close_mean = mean(cls)
            self.close_std = pstdev(cls) if len(cls) > 1 else 0.0
        else:
            self.close_mean = None
            self.close_std = None

        rs = self.returns()
        if len(rs) >= 1:
            self.ret_mean = mean(rs)
            self.ret_std = pstdev(rs) if len(rs) > 1 else 0.0
        else:
            self.ret_mean = None
            self.ret_std = None

    # --------------------------
    # Utilidades
    # --------------------------
    def add_candle(self, c: Candle):
        self.candles.append(c)
        self.__post_init__()

    def to_records(self) -> List[Dict]:
        rows: List[Dict] = []
        for c in self.candles:
            rows.append({
                "provider": self.provider,
                "datatype": self.datatype,
                "symbol": self.symbol,
                "date": c.date,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
        return rows

    def summary(self) -> Dict[str, Optional[float]]:
        return {
            "provider": self.provider,
            "datatype": self.datatype,
            "symbol": self.symbol,
            "n_obs": self.n_obs,
            "close_mean": self.close_mean,
            "close_std": self.close_std,
            "ret_mean": self.ret_mean,
            "ret_std": self.ret_std,
        }

    def to_dataframe(self):
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError("pandas no está instalado (requerido para to_dataframe).")
        df = pd.DataFrame(self.to_records()).sort_values("date").reset_index(drop=True)
        return df
