from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import io

from app.models.series import PriceSeries
from app.analytics.metrics import (
    align_series_returns, portfolio_return_series,
    metrics_for_portfolio
)
from app.analytics.simulate import simulate_portfolio_paths, summarize_paths

@dataclass
class Portfolio:
    series: Dict[str, PriceSeries]                # {ticker: PriceSeries}
    weights: Dict[str, float]                     # {ticker: peso}
    initial_value: float = 1.0
    name: str = "MyPortfolio"

    # Resultados de simulación (opcionales)
    sim_paths: Optional[np.ndarray] = field(init=False, default=None)
    sim_summary: Optional[Dict[str, np.ndarray]] = field(init=False, default=None)

    def __post_init__(self):
        ssum = sum(self.weights.values())
        if ssum <= 0:
            raise ValueError("Suma de pesos debe ser > 0")
        if abs(ssum - 1.0) > 1e-9:
            self.weights = {k: v/ssum for k, v in self.weights.items()}

    # ---------- Métricas históricas sobre retornos alineados ----------
    def _historical_portfolio_returns(self) -> List[Tuple[str, float]]:
        aligned = align_series_returns(self.series)  # [(fecha, {sym:r,...}), ...]
        return portfolio_return_series(self.weights, aligned)

    def historical_metrics(self, rf: float = 0.0, freq: int = 252, alpha: float = 0.95) -> Dict[str, Optional[float]]:
        rets = [x for _, x in self._historical_portfolio_returns()]
        return metrics_for_portfolio(rets, rf=rf, freq=freq, alpha=alpha)

    # ---------- Monte Carlo integrado ----------
    def montecarlo_simulate(self,
                            days: int = 252,
                            n_paths: int = 1000,
                            model: str = "gbm",
                            rebalance_daily: bool = True,
                            seed: Optional[int] = None,
                            q: float = 0.05):
        self.sim_paths = simulate_portfolio_paths(
            self.series, self.weights,
            days=days, n_paths=n_paths, model=model,
            seed=seed, rebalance_daily=rebalance_daily,
            initial_value=self.initial_value
        )
        self.sim_summary = summarize_paths(self.sim_paths, q=q)
        return self

    # ---------- Reporte en Markdown ----------
    def report(self,
               title: Optional[str] = None,
               rf: float = 0.0,
               freq: int = 252,
               alpha: float = 0.95,
               include_simulation: bool = True) -> str:
        """
        Devuelve Markdown con:
          - pesos, componentes, span, n_obs
          - métricas históricas (Sharpe, VaR, CVaR, media, vol)
          - advertencias de calidad de datos
          - (opcional) métricas de simulación si existen
        """
        title = title or f"Informe de Cartera — {self.name}"
        lines: List[str] = [f"# {title}", ""]

        # Pesos
        lines.append("## Composición")
        for k, v in self.weights.items():
            lines.append(f"- **{k}**: {v:.2%}")
        lines.append("")

        # Calidad de las series
        lines.append("## Calidad de datos (series)")
        any_warn = False
        for sym, ps in self.series.items():
            start, end = ps.span_dates()
            warns = ps.validate()
            span = f"{start} → {end}" if start and end else "N/A"
            lines.append(f"- **{sym}** · span: {span} · n={ps.n_obs}")
            if warns:
                any_warn = True
                for w in warns:
                    lines.append(f"  - ⚠️ {w}")
        if not any_warn:
            lines.append("- ✅ Sin incidencias relevantes en limpieza.")
        lines.append("")

        # Métricas históricas de la cartera
        lines.append("## Métricas históricas de la cartera")
        hist = self.historical_metrics(rf=rf, freq=freq, alpha=alpha)
        lines.append(f"- Retorno medio (por periodo): **{(hist['ret_mean'] or 0)*100:.3f}%**")
        lines.append(f"- Volatilidad (por periodo): **{(hist['ret_std'] or 0)*100:.3f}%**")
        lines.append(f"- Sharpe (anualizado): **{hist['sharpe'] if hist['sharpe'] is not None else 'N/A'}**")
        lines.append(f"- VaR {int(alpha*100)}% (histórico): **{hist['var'] if hist['var'] is not None else 'N/A'}**")
        lines.append(f"- CVaR {int(alpha*100)}% (histórico): **{hist['cvar'] if hist['cvar'] is not None else 'N/A'}**")
        lines.append(f"- Observaciones: **{hist['n_obs']}**")
        lines.append("")

        # Simulación si existe
        if include_simulation and self.sim_summary is not None:
            lines.append("## Simulación Monte Carlo (resumen)")
            mean_end = float(self.sim_summary["mean"][-1])
            p5_end   = float(self.sim_summary["p_low"][-1])
            p95_end  = float(self.sim_summary["p_high"][-1])
            lines.append(f"- Valor final esperado: **{mean_end:.4f}** (V0={self.initial_value})")
            lines.append(f"- Banda {5}–{95}% al final: **[{p5_end:.4f}, {p95_end:.4f}]**")
            lines.append("")

        # Advertencias generales
        lines.append("## Notas y advertencias")
        if hist["n_obs"] < 30:
            lines.append("- ⚠️ Pocas observaciones históricas (<30); las métricas pueden ser inestables.")
        if any(ps.ret_std == 0.0 for ps in self.series.values() if ps.ret_std is not None):
            lines.append("- ⚠️ Alguna serie parece prácticamente constante; revise los datos.")
        if not include_simulation:
            lines.append("- ℹ️ Simulación no incluida en este informe. Ejecuta `montecarlo_simulate()` para agregarla.")

        return "\n".join(lines)

    # ---------- Visualizaciones útiles ----------
    def plots_report(self,
                     normalize: bool = True,
                     show_paths: bool = True,
                     max_paths: int = 10):
        """
        Muestra:
          1) Cierres normalizados por componente (comparativa)
          2) Histograma de retornos de cartera (histórico)
          3) Si hay simulación: banda [p5,p95], media y hasta N trayectorias
        """
        import pandas as pd

        # 1) Cierres normalizados (si procede)
        plt.figure(figsize=(9,4))
        for sym, ps in self.series.items():
            cls = np.array(ps.closes(), dtype=float)
            if cls.size == 0:
                continue
            y = cls/cls[0] if normalize else cls
            x = np.arange(len(y))
            plt.plot(x, y, label=sym)
        plt.title(f"Cierres {'normalizados' if normalize else ''} — {self.name}")
        plt.xlabel("Tiempo (índice)")
        plt.ylabel("Precio" + (" normalizado" if normalize else ""))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

        # 2) Histograma de retornos de cartera (histórico)
        rets = [r for _, r in self._historical_portfolio_returns()]
        if rets:
            plt.figure(figsize=(7,4))
            plt.hist(rets, bins=40, alpha=0.8)
            plt.title(f"Histograma de retornos (histórico) — {self.name}")
            plt.xlabel("Retorno")
            plt.ylabel("Frecuencia")
            plt.grid(alpha=0.3)
            plt.show()

        # 3) Simulación (si existe)
        if self.sim_summary is not None and self.sim_paths is not None:
            mean = self.sim_summary["mean"]
            p_low = self.sim_summary["p_low"]
            p_high = self.sim_summary["p_high"]
            t = np.arange(len(mean))
            plt.figure(figsize=(9,5))
            plt.fill_between(t, p_low, p_high, alpha=0.2, label="5%-95%")
            plt.plot(t, mean, color="black", lw=2, label="Media")
            if show_paths:
                n_draw = min(max_paths, self.sim_paths.shape[0])
                for i in range(n_draw):
                    plt.plot(t, self.sim_paths[i, :], alpha=0.35, lw=0.8)
            plt.title(f"Simulación Monte Carlo — {self.name}")
            plt.xlabel("Días")
            plt.ylabel("Valor cartera")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.show()
