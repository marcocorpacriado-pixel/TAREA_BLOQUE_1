from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import math

from app.models.series import PriceSeries

# ---------------------------
# Utilidades internas
# ---------------------------
def _daily_mu_sigma(ps: PriceSeries) -> Tuple[float, float]:
    """
    Estima mu y sigma diarios a partir de los retornos log (ps.returns()).
    Si ps.use_log_returns=False, seguirán siendo retornos simples; para GBM
    recomendamos PriceSeries.use_log_returns=True (por defecto ya lo está).
    """
    rets = ps.returns()
    if len(rets) < 2:
        return 0.0, 0.0
    arr = np.array(rets, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mu, sd

def _last_close(ps: PriceSeries) -> float:
    cls = ps.closes()
    if not cls:
        raise ValueError(f"No hay cierres en la serie {ps.symbol}.")
    return float(cls[-1])

# ---------------------------
# Simulación a nivel ACTIVO
# ---------------------------
def simulate_asset_return_paths(
    ps: PriceSeries,
    days: int = 252,
    n_paths: int = 1000,
    model: str = "gbm",           # "gbm" o "bootstrap"
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Devuelve matriz de retornos diarios simulados shape = (n_paths, days).
    - GBM: r_t ~ N(mu, sigma) (mu, sigma diarios estimados de la serie)
    - Bootstrap: r_t ~ sample con reemplazo de los retornos históricos
    Nota: si ps.use_log_returns=True, estos retornos serán log-returns.
    """
    rng = np.random.default_rng(seed)
    if days <= 0 or n_paths <= 0:
        raise ValueError("days y n_paths deben ser > 0")

    if model not in ("gbm", "bootstrap"):
        raise ValueError("model debe ser 'gbm' o 'bootstrap'")

    if model == "gbm":
        mu, sd = _daily_mu_sigma(ps)
        # r_t (log) ~ N(mu, sd)
        R = rng.normal(loc=mu, scale=sd, size=(n_paths, days))
    else:
        hist = np.array(ps.returns(), dtype=float)
        if hist.size == 0:
            # sin retornos: todo cero
            R = np.zeros((n_paths, days), dtype=float)
        else:
            idx = rng.integers(0, hist.size, size=(n_paths, days))
            R = hist[idx]

    return R

def simulate_asset_price_paths(
    ps: PriceSeries,
    days: int = 252,
    n_paths: int = 1000,
    model: str = "gbm",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Devuelve matriz de precios shape = (n_paths, days+1).
    - Columna 0 es el precio inicial S0 (último close observado).
    - Si los retornos son log: S_t = S0 * exp( cumsum(r) ).
      Si fueran simples: S_t = S0 * cumprod(1 + r).
    """
    S0 = _last_close(ps)
    R = simulate_asset_return_paths(ps, days=days, n_paths=n_paths, model=model, seed=seed)

    if ps.use_log_returns:
        # log-returns → sumas y exponencias
        # shape (n_paths, days)
        cum = np.cumsum(R, axis=1)
        paths = np.column_stack([np.full((n_paths, 1), S0), S0 * np.exp(cum)])
    else:
        # retornos simples → producto acumulado
        cum = np.cumprod(1.0 + R, axis=1)
        paths = np.column_stack([np.full((n_paths, 1), S0), S0 * cum])

    return paths

# ---------------------------
# Simulación a nivel CARTERA
# ---------------------------
def simulate_portfolio_paths(
    series: Dict[str, PriceSeries],
    weights: Dict[str, float],
    days: int = 252,
    n_paths: int = 1000,
    model: str = "gbm",
    seed: Optional[int] = None,
    rebalance_daily: bool = True,
    initial_value: float = 1.0,
) -> np.ndarray:
    """
    Simula la evolución de una cartera.
    - series: {symbol: PriceSeries}
    - weights: {symbol: peso}, sum(weights)=1 recomendado
    - rebalance_daily=True: cada día el retorno de cartera es sum(w_i * r_i_t) (re-balanceo)
      False: buy&hold aproximado usando pesos iniciales y precios simulados.

    Devuelve matriz shape (n_paths, days+1) con el valor de la cartera (V0=initial_value).
    """
    if not series:
        raise ValueError("No hay series en la cartera.")
    syms = list(series.keys())
    if any(w < 0 for w in weights.values()):
        raise ValueError("Pesos negativos no soportados en esta función.")
    # Normaliza pesos si no suman 1
    total_w = sum(weights.get(s, 0.0) for s in syms)
    if total_w <= 0:
        raise ValueError("La suma de pesos debe ser > 0.")
    w = np.array([weights.get(s, 0.0) for s in syms], dtype=float) / total_w

    rng = np.random.default_rng(seed)
    # Simula retornos por activo
    R_by_sym = []
    for s in syms:
        R = simulate_asset_return_paths(series[s], days=days, n_paths=n_paths, model=model, seed=rng.integers(0, 2**31-1))
        R_by_sym.append(R)  # (n_paths, days)

    # Stacking: (n_assets, n_paths, days)
    R_stack = np.stack(R_by_sym, axis=0)

    if rebalance_daily:
        # Retorno de cartera día t = sum_i w_i * r_{i,t}
        # Si series usan log-returns (por defecto), la suma es log-return de cartera (exacto si rebalancing).
        # Si fueran simples, es aproximación lineal.
        # (n_paths, days)
        if series[syms[0]].use_log_returns:
            Rc = np.tensordot(w, R_stack, axes=(0, 0))  # (n_paths, days)
            # log-returns → precio
            cum = np.cumsum(Rc, axis=1)
            V = np.column_stack([np.full((n_paths, 1), initial_value), initial_value * np.exp(cum)])
        else:
            Rc = np.tensordot(w, R_stack, axes=(0, 0))
            cum = np.cumprod(1.0 + Rc, axis=1)
            V = np.column_stack([np.full((n_paths, 1), initial_value), initial_value * cum])
        return V
    else:
        # Buy & Hold aproximado:
        # 1) Simula precios por activo, 2) invierte initial_value*w_i en cada activo a S0,
        # 3) valor de cartera = suma del valor de cada posición.
        paths_by_sym = []
        for s in syms:
            P = simulate_asset_price_paths(series[s], days=days, n_paths=n_paths, model=model, seed=rng.integers(0, 2**31-1))
            paths_by_sym.append(P)  # (n_paths, days+1)

        # Inversión inicial por activo
        S0s = np.array([_last_close(series[s]) for s in syms], dtype=float)  # precios iniciales
        alloc_value = initial_value * w                                # valor asignado por activo
        shares = alloc_value / S0s                                     # nº de participaciones por activo

        # Valor de cartera = suma_i shares_i * Price_i(t)
        # Construimos V con shape (n_paths, days+1)
        V = np.zeros_like(paths_by_sym[0])
        for i, P in enumerate(paths_by_sym):
            V += shares[i] * P
        return V

# ---------------------------
# Resúmenes útiles
# ---------------------------
def summarize_paths(paths: np.ndarray, q: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Devuelve estadísticas por tiempo:
      - mean: media por día
      - p_low / p_high: cuantiles simétricos (p.ej. 5% y 95% si q=0.05)
    paths: (n_paths, days+1)
    """
    if paths.ndim != 2:
        raise ValueError("paths debe ser una matriz 2D (n_paths, days+1).")
    mean = paths.mean(axis=0)
    low = np.quantile(paths, q, axis=0)
    high = np.quantile(paths, 1.0 - q, axis=0)
    return {"mean": mean, "p_low": low, "p_high": high}
