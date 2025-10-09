from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from statistics import mean, pstdev

from app.models.series import PriceSeries

@dataclass
class Portfolio:
    """
    Cartera de series con pesos (suma de pesos ~ 1.0 recomendado).
    Las series deben ser 'PriceSeries' con retornos diarios.
    """
    series: Dict[str, PriceSeries] = field(default_factory=dict)  # symbol -> PriceSeries
    weights: Dict[str, float] = field(default_factory=dict)       # symbol -> weight

    def add(self, ps: PriceSeries, weight: float):
        self.series[ps.symbol] = ps
        self.weights[ps.symbol] = weight

    def _aligned_returns(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Devuelve una lista [(fecha, {symbol: ret, ...}), ...] con la intersección de fechas.
        """
        # Construir mapa symbol -> dict(fecha -> retorno)
        per_symbol: Dict[str, Dict[str, float]] = {}
        for sym, ps in self.series.items():
            per_symbol[sym] = dict(ps.returns_with_dates())  # fecha -> retorno

        # Intersección de fechas entre todos los símbolos
        symbols = list(per_symbol.keys())
        if not symbols:
            return []

        # conjunto de fechas de intersección
        common_dates = None
        for sym in symbols:
            dates = set(per_symbol[sym].keys())
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates & dates

        if not common_dates:
            return []

        dates_sorted = sorted(list(common_dates))  # orden ascendente
        aligned: List[Tuple[str, Dict[str, float]]] = []
        for d in dates_sorted:
            row: Dict[str, float] = {}
            for sym in symbols:
                row[sym] = per_symbol[sym][d]
            aligned.append((d, row))
        return aligned

    def portfolio_returns(self) -> List[Tuple[str, float]]:
        """
        Retornos diarios de la cartera = suma de w_i * r_i alineados por fecha.
        """
        aligned = self._aligned_returns()
        out: List[Tuple[str, float]] = []
        for d, row in aligned:
            pr = 0.0
            for sym, r in row.items():
                w = self.weights.get(sym, 0.0)
                pr += w * r
            out.append((d, pr))
        return out

    def stats(self) -> Dict[str, Optional[float]]:
        """
        Estadística básica de la cartera (media y desviación de retornos).
        """
        rets = [x for _, x in self.portfolio_returns()]
        if len(rets) == 0:
            return {"ret_mean": None, "ret_std": None, "n_obs": 0}
        elif len(rets) == 1:
            return {"ret_mean": rets[0], "ret_std": 0.0, "n_obs": 1}
        else:
            return {"ret_mean": mean(rets), "ret_std": pstdev(rets), "n_obs": len(rets)}
