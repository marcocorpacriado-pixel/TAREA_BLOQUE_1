path = "/content/TAREA_BLOQUE_1/src/app/models/portfolio.py"
code = r'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from app.models.series import PriceSeries
from app.analytics.metrics import (
    align_series_returns, portfolio_return_series,
    metrics_for_portfolio
)
from app.analytics.simulate import simulate_portfolio_paths, summarize_paths

@dataclass
class Portfolio:
    series: Dict[str, PriceSeries]
    weights: Dict[str, float]
    initial_value: float = 1.0
    name: str = "MyPortfolio"

    sim_paths: Optional[np.ndarray] = field(init=False, default=None)
    sim_summary: Optional[Dict[str, np.ndarray]] = field(init=False, default=None)

    def __post_init__(self):
        ssum = sum(self.weights.values())
        if ssum <= 0:
            raise ValueError("Suma de pesos debe ser > 0")
        if abs(ssum - 1.0) > 1e-9:
            self.weights = {k: v/ssum for k, v in self.weights.items()}

    def _historical_portfolio_returns(self) -> List[Tuple[str, float]]:
        aligned = align_series_returns(self.series)
        return portfolio_return_series(self.weights, aligned)

    def historical_metrics(self, rf: float = 0.0, freq: int = 252, alpha: float = 0.95) -> Dict[str, Optional[float]]:
        rets = [x for _, x in self._historical_portfolio_returns()]
        return metrics_for_portfolio(rets, rf=rf, freq=freq, alpha=alpha)

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

    def report(self,
               title: Optional[str] = None,
               rf: float = 0.0,
               freq: int = 252,
               alpha: float = 0.95,
               include_simulation: bool = True) -> str:
        title = title or f"Informe de Cartera — {self.name}"
        lines: List[str] = [f"# {title}", ""]
        lines.append("## Composición")
        for k, v in self.weights.items():
            lines.append(f"- **{k}**: {v:.2%}")
        lines.append("")
        lines.append("## Métricas históricas de la cartera")
        hist = self.historical_metrics(rf=rf, freq=freq, alpha=alpha)
        lines.append(f"- Retorno medio: {(hist['ret_mean'] or 0)*100:.3f}%")
        lines.append(f"- Volatilidad: {(hist['ret_std'] or 0)*100:.3f}%")
        lines.append(f"- Sharpe: {hist['sharpe'] if hist['sharpe'] else 'N/A'}")
        lines.append("")
        if include_simulation and self.sim_summary is not None:
            lines.append("## Simulación Monte Carlo")
            mean_end = float(self.sim_summary["mean"][-1])
            p5_end   = float(self.sim_summary["p_low"][-1])
            p95_end  = float(self.sim_summary["p_high"][-1])
            lines.append(f"- Valor final esperado: {mean_end:.4f}")
            lines.append(f"- Banda 5–95%: [{p5_end:.4f}, {p95_end:.4f}]")
        return "\n".join(lines)

    def plots_report(self,
                     which: list[str] = ("components", "hist", "sim"),
                     normalize: bool = True,
                     show_paths: bool = True,
                     max_paths: int = 10,
                     figsize: tuple[float, float] = (12, 6),
                     dpi: int = 160,
                     save_dir: str | None = None):
        import os
        os.makedirs(save_dir, exist_ok=True) if save_dir else None

        # 1) Componentes
        if "components" in which:
            plt.figure(figsize=figsize, dpi=dpi)
            for sym, ps in self.series.items():
                cls = np.array(ps.closes(), dtype=float)
                if cls.size == 0:
                    continue
                y = cls/cls[0] if normalize else cls
                plt.plot(y, label=sym)
            plt.title(f"Cierres {'normalizados' if normalize else ''} — {self.name}")
            plt.grid(alpha=0.3)
            plt.legend()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "components.png"), bbox_inches="tight")
            plt.show()

        # 2) Simulación
        if "sim" in which and self.sim_summary is not None:
            mean = self.sim_summary["mean"]
            p_low = self.sim_summary["p_low"]
            p_high = self.sim_summary["p_high"]
            t = np.arange(len(mean))
            plt.figure(figsize=figsize, dpi=dpi)
            plt.fill_between(t, p_low, p_high, alpha=0.2, label="5%-95%")
            plt.plot(t, mean, lw=2, label="Media")
            if show_paths and self.sim_paths is not None:
                for i in range(min(max_paths, self.sim_paths.shape[0])):
                    plt.plot(t, self.sim_paths[i, :], alpha=0.3, lw=0.7)
            plt.title(f"Simulación Monte Carlo — {self.name}")
            plt.grid(alpha=0.3)
            plt.legend()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "simulation.png"), bbox_inches="tight")
            plt.show()
'''
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

import importlib, app.models.portfolio as pf_mod
importlib.reload(pf_mod)
print("✅ portfolio.py actualizado y recargado correctamente")
from app.models.portfolio import Portfolio
print("Métodos disponibles:", [m for m in dir(Portfolio) if not m.startswith('_')])
