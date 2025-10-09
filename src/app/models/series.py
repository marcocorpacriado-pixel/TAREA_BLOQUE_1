from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Tuple
from datetime import datetime
from statistics import mean, pstdev
import math
import os

from app.extractor.base import Candle

# -----------------------------
# Helpers internos
# -----------------------------
def _parse_date(d: str) -> datetime:
    return datetime.strptime(str(d)[:10], "%Y-%m-%d")

def _sorted_unique_candles(candles: Iterable[Candle]) -> List[Candle]:
    by_date: Dict[str, Candle] = {}
    for c in candles:
        ds = str(c.date)[:10]
        by_date[ds] = Candle(
            date=ds,
            open=float(c.open) if c.open is not None else float("nan"),
            high=float(c.high) if c.high is not None else float("nan"),
            low=float(c.low) if c.low is not None else float("nan"),
            close=float(c.close) if c.close is not None else float("nan"),
            volume=float(c.volume) if c.volume is not None else 0.0,
        )
    return [by_date[ds] for ds in sorted(by_date.keys(), key=_parse_date)]

def _ffill_closes(candles: List[Candle]) -> List[Candle]:
    prev_close: Optional[float] = None
    out: List[Candle] = []
    for c in candles:
        close = c.close if (c.close is not None and not math.isnan(c.close)) else prev_close
        if close is None:
            out.append(c)
        else:
            out.append(Candle(c.date, c.open, c.high, c.low, close, c.volume))
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

def _clip_outliers(values: List[float], z: float = 6.0) -> List[float]:
    # recorte winsor simple por z-score
    if len(values) < 3:
        return values
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    if s == 0.0:
        return values
    lo, hi = m - z*s, m + z*s
    return [min(max(v, lo), hi) for v in values]

# -----------------------------
# Serie de precios
# -----------------------------
@dataclass
class PriceSeries:
    """
    Representa UNA serie temporal (un símbolo con un proveedor y un tipo de datos).
    Acepta input flexible (DataFrame/CSV/records) y expone herramientas de limpieza.
    """
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

    # Config limpieza / retornos
    forward_fill: bool = True
    use_log_returns: bool = True  # retornos log por defecto

    # ---------- Constructores flexibles ----------
    @classmethod
    def from_dataframe(cls,
                       df,
                       symbol: str,
                       provider: str = "custom",
                       datatype: str = "prices",
                       column_map: Optional[Dict[str, str]] = None) -> "PriceSeries":
        """
        Acepta cualquier DataFrame que tenga 'date' y 'close' (y opcionalmente open/high/low/volume).
        column_map permite mapear nombres arbitrarios -> estándar: {"fecha":"date","Adj Close":"close",...}
        """
        if column_map:
            df = df.rename(columns=column_map)

        required = {"date", "close"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"El DataFrame debe tener columnas al menos {required}. Recibido: {list(df.columns)}")

        # columnas opcionales
        open_col  = "open" if "open" in df.columns else None
        high_col  = "high" if "high" in df.columns else None
        low_col   = "low"  if "low"  in df.columns else None
        vol_col   = "volume" if "volume" in df.columns else None

        candles: List[Candle] = []
        for _, row in df.iterrows():
            candles.append(Candle(
                date=str(row["date"])[:10],
                open=float(row[open_col]) if open_col  else float("nan"),
                high=float(row[high_col]) if high_col  else float("nan"),
                low=float(row[low_col])  if low_col   else float("nan"),
                close=float(row["close"]),
                volume=float(row[vol_col]) if vol_col else 0.0
            ))
        return cls(provider=provider, datatype=datatype, symbol=symbol, candles=candles)

    @classmethod
    def from_csv(cls, path: str, symbol: str, provider: str = "custom", datatype: str = "prices",
                 column_map: Optional[Dict[str, str]] = None) -> "PriceSeries":
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError("pandas es necesario para PriceSeries.from_csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        return cls.from_dataframe(df, symbol=symbol, provider=provider, datatype=datatype, column_map=column_map)

    @classmethod
    def from_records(cls, records: List[Dict], symbol: str, provider: str = "custom", datatype: str = "prices") -> "PriceSeries":
        candles = []
        for r in records:
            candles.append(Candle(
                date=str(r["date"])[:10],
                open=float(r.get("open", float("nan"))),
                high=float(r.get("high", float("nan"))),
                low=float(r.get("low", float("nan"))),
                close=float(r["close"]),
                volume=float(r.get("volume", 0.0)),
            ))
        return cls(provider=provider, datatype=datatype, symbol=symbol, candles=candles)

    # ---------- Inicialización / limpieza ----------
    def __post_init__(self):
        # ordenar + deduplicar + limpiar missing inicial y FFill si procede
        self.candles = _sorted_unique_candles(self.candles)
        if self.forward_fill:
            self.candles = _ffill_closes(self.candles)
        self.candles = _drop_na_head(self.candles)
        self.n_obs = len(self.candles)
        self._recompute_stats()

    # ---------- Limpieza / preproceso ----------
    def clean(self,
              forward_fill: Optional[bool] = None,
              clip_outliers_z: Optional[float] = None) -> "PriceSeries":
        """
        - forward_fill: cambia dinámicamente el modo ffill y re-calcula
        - clip_outliers_z: aplica winsor sobre cierres antes de stats (no muta datos raw OHLC)
        """
        if forward_fill is not None:
            self.forward_fill = forward_fill
        # re-aplica pipeline base
        self.__post_init__()

        if clip_outliers_z:
            cls = self.closes()
            clipped = _clip_outliers(cls, z=clip_outliers_z)
            # reemplazar solo el close (no tocamos open/high/low)
            for i, c in enumerate(self.candles):
                self.candles[i] = Candle(c.date, c.open, c.high, c.low, clipped[i], c.volume)
            self._recompute_stats()
        return self

    def validate(self) -> List[str]:
        issues = []
        if self.n_obs < 3:
            issues.append("Muy pocas observaciones (<3).")
        if any(c.close is None or math.isnan(c.close) for c in self.candles):
            issues.append("Existen cierres NaN tras limpieza.")
        if self.close_std is not None and self.close_std == 0.0:
            issues.append("Desviación de cierre = 0 (serie constante).")
        return issues

    # ---------- Accesores ----------
    def closes(self) -> List[float]:
        return [float(c.close) for c in self.candles if c.close is not None and not math.isnan(c.close)]

    def dates(self) -> List[str]:
        return [c.date for c in self.candles]

    def span_dates(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.candles:
            return None, None
        return self.candles[0].date, self.candles[-1].date

    # ---------- Retornos ----------
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

    # ---------- Stats automáticas ----------
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

    # ---------- Utilidades ----------
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
        start, end = self.span_dates()
        return {
            "provider": self.provider,
            "datatype": self.datatype,
            "symbol": self.symbol,
            "span": f"{start} → {end}" if start and end else None,
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
