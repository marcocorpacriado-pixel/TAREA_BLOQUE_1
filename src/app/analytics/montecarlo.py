from __future__ import annotations
from typing import Dict, List, Optional
import random

from app.models.series import PriceSeries
from app.analytics.metrics import align_series_returns, portfolio_return_series, metrics_for_portfolio

def _random_weights(symbols: List[str]) -> Dict[str, float]:
    # Dirichlet simple con alphas=1 (equivalente a pesos aleatorios uniformes que suman 1)
    raw = [random.random() for _ in symbols]
    s = sum(raw) or 1.0
    return {sym: x / s for sym, x in zip(symbols, raw)}

def simulate_random_portfolios(series: Dict[str, PriceSeries],
                               n: int = 1000,
                               criterion: str = "sharpe",
                               rf: float = 0.0,
                               freq: int = 252,
                               alpha: float = 0.95) -> List[dict]:
    """
    Genera n carteras aleatorias, calcula métricas y devuelve una lista de dicts.
    criterion: 'sharpe' | 'var' | 'cvar' | 'ret_mean' | 'ret_std'
    """
    symbols = list(series.keys())
    aligned = align_series_returns(series)  # [(fecha, {sym: ret, ...}), ...]
    if not aligned:
        raise ValueError("No hay intersección de fechas entre las series.")

    results: List[dict] = []
    for _ in range(n):
        w = _random_weights(symbols)
        pr = portfolio_return_series(w, aligned)
        ret_series = [x for _, x in pr]
        m = metrics_for_portfolio(ret_series, rf=rf, freq=freq, alpha=alpha)
        row = {"weights": w, **m}
        results.append(row)

    # ordena por criterio (Sharpe descendente por defecto; VaR/CVaR: orden ascendente — más alto = peor)
    reverse = True
    key = criterion
    if criterion in ("var", "cvar", "ret_std"):
        reverse = False  # menor riesgo mejor
    results.sort(key=lambda r: (r[key] is None, r[key]), reverse=reverse)
    return results
