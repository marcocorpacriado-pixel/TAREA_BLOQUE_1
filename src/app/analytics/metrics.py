from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from statistics import mean, pstdev
import math

from app.models.series import PriceSeries

def align_series_returns(series: Dict[str, PriceSeries]) -> List[Tuple[str, Dict[str, float]]]:
    """
    Devuelve [(fecha, {symbol: ret, ...}), ...] alineando la intersección de fechas.
    Usa returns() de PriceSeries (ya log o simples según config).
    """
    per_symbol: Dict[str, Dict[str, float]] = {sym: dict(ps.returns_with_dates()) for sym, ps in series.items()}
    symbols = list(per_symbol.keys())
    if not symbols:
        return []

    common = None
    for sym in symbols:
        dates = set(per_symbol[sym].keys())
        common = dates if common is None else (common & dates)
    if not common:
        return []

    dates_sorted = sorted(list(common))
    aligned: List[Tuple[str, Dict[str, float]]] = []
    for d in dates_sorted:
        row = {sym: per_symbol[sym][d] for sym in symbols}
        aligned.append((d, row))
    return aligned

def portfolio_return_series(weights: Dict[str, float],
                            aligned_returns: List[Tuple[str, Dict[str, float]]]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for d, row in aligned_returns:
        r = sum(weights.get(sym, 0.0) * row[sym] for sym in row.keys())
        out.append((d, r))
    return out

def sharpe_ratio(returns: List[float], rf: float = 0.0, freq: int = 252) -> Optional[float]:
    """
    Sharpe anualizado sobre retornos (log o simples).
    rf es el retorno por periodo (no anual); si rf es anual, ajusta antes.
    """
    if len(returns) < 2:
        return None
    excess = [r - rf for r in returns]
    mu = mean(excess)
    sd = pstdev(excess) if len(excess) > 1 else 0.0
    if sd == 0:
        return None
    return (mu / sd) * math.sqrt(freq)

def var_cvar(returns: List[float], alpha: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """
    VaR y CVaR históricos (por defecto 95%), en el espacio de retornos.
    Convención: pérdidas como números negativos; VaR es el cuantil inferior.
    """
    if not returns:
        return None, None
    xs = sorted(returns)
    # índice del cuantil (p.ej. 5% para el lado de pérdidas si alpha=0.95)
    q = 1.0 - alpha
    idx = max(0, min(len(xs) - 1, int(q * len(xs))))
    var_value = xs[idx]
    tail = xs[:idx + 1]  # pérdidas peores o iguales que el VaR
    cvar_value = sum(tail) / len(tail) if tail else var_value
    return var_value, cvar_value

def metrics_for_portfolio(returns: List[float],
                          rf: float = 0.0, freq: int = 252, alpha: float = 0.95) -> Dict[str, Optional[float]]:
    if len(returns) == 0:
        return {"sharpe": None, "var": None, "cvar": None, "ret_mean": None, "ret_std": None, "n_obs": 0}
    mu = mean(returns)
    sd = pstdev(returns) if len(returns) > 1 else 0.0
    s = sharpe_ratio(returns, rf=rf, freq=freq)
    v, cv = var_cvar(returns, alpha=alpha)
    return {"sharpe": s, "var": v, "cvar": cv, "ret_mean": mu, "ret_std": sd, "n_obs": len(returns)}
