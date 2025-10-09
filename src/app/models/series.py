from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Tuple
from datetime import datetime
from statistics import mean, pstdev

# Reutilizamos Candle del proyecto (formato estandarizado)
from app.extractor.base import Candle

def _sorted_candles(candles: Iterable[Candle]) -> List[Candle]:
    # Ordenamos por fecha ascendente para que los retornos tengan sentido
    def _parse(d: str) -> datetime:
        # "YYYY-MM-DD"
        return datetime.strptime(d, "%Y-%m-%d")
    return sorted(list(candles), key=lambda c: _parse(c.date))

@dataclass
class PriceSeries:
    """
    Representa UNA serie temporal (un símbolo con un proveedor y un tipo de datos).
    'candles' es la lista de observaciones OHLCV ya estandarizadas.
    """
    provider: str
    datatype: str   # "prices" o "fx"
    symbol: str
    candles: List[Candle] = field(default_factory=list)

    # Métricas calculadas automáticamente en __post_init__
    close_mean: Optional[float] = field(init=False, default=None)
    close_std: Optional[float] = field(init=False, default=None)
    ret_mean: Optional[float] = field(init=False, default=None)
    ret_std: Optional[float] = field(init=False, default=None)
    n_obs: int = field(init=False, default=0)

    def __post_init__(self):
        # Asegurar orden temporal
        self.candles = _sorted_candles(self.candles)
        self.n_obs = len(self.candles)
        # Calcula métricas básicas automáticamente
        self._recompute_stats()

    # --------------------------
    # Métodos "core"
    # --------------------------
    def closes(self) -> List[float]:
        return [c.close for c in self.candles]

    def dates(self) -> List[str]:
        return [c.date for c in self.candles]

    def returns(self) -> List[float]:
        """
        Retornos simples diarios a partir de 'close':
        r_t = close_t / close_{t-1} - 1
        """
        closes = self.closes()
        if len(closes) < 2:
            return []
        rets: List[float] = []
        prev = closes[0]
        for x in closes[1:]:
            if prev == 0:
                rets.append(0.0)
            else:
                rets.append(x / prev - 1.0)
            prev = x
        return rets

    def returns_with_dates(self) -> List[Tuple[str, float]]:
        ds = self.dates()
        rs = self.returns()
        # alinear fechas con retornos (desde la 2ª fecha)
        return list(zip(ds[1:], rs))

    # --------------------------
    # Estadística (auto + on-demand)
    # --------------------------
    def _recompute_stats(self):
        # Media y desviación del 'close'
        cls = self.closes()
        if len(cls) >= 1:
            self.close_mean = mean(cls)
            self.close_std = pstdev(cls) if len(cls) > 1 else 0.0
        else:
            self.close_mean = None
            self.close_std = None

        # Media y desviación de retornos
        rs = self.returns()
        if len(rs) >= 1:
            self.ret_mean = mean(rs)
            self.ret_std = pstdev(rs) if len(rs) > 1 else 0.0
        else:
            self.ret_mean = None
            self.ret_std = None

    def add_candle(self, c: Candle):
        """Añadir una nueva observación y actualizar stats automáticamente."""
        self.candles.append(c)
        self.candles = _sorted_candles(self.candles)
        self.n_obs = len(self.candles)
        self._recompute_stats()

    # --------------------------
    # Conversión utilitaria
    # --------------------------
    def to_records(self) -> List[Dict]:
        """La misma estructura estandarizada que guardamos en CSV/Parquet con metadatos."""
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
        """Resumen rápido para imprimir o logging."""
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

    # (Opcional) integración suave con pandas si está disponible
    def to_dataframe(self):
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError("pandas no está instalado (requerido para to_dataframe).")
        df = pd.DataFrame(self.to_records())
        # garantizar orden temporal
        df = df.sort_values("date").reset_index(drop=True)
        return df
