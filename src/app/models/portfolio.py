from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from app.models.series import PriceSeries
from app.analytics.simulate import (
    simulate_portfolio_paths,
    summarize_paths,
)

@dataclass
class Portfolio:
    series: Dict[str, PriceSeries]                # {ticker: PriceSeries}
    weights: Dict[str, float]                    # {ticker: peso}
    initial_value: float = 1.0
    name: str = "MyPortfolio"

    # Estos se completarán tras la simulación
    sim_paths: Optional[np.ndarray] = field(init=False, default=None)
    sim_summary: Optional[Dict[str, np.ndarray]] = field(init=False, default=None)

    def __post_init__(self):
        ssum = sum(self.weights.values())
        if abs(ssum - 1.0) > 1e-6:
            self.weights = {k: v/ssum for k, v in self.weights.items()}

    # ============================================================
    # MÉTODO PRINCIPAL: SIMULACIÓN DE MONTE CARLO
    # ============================================================
    def montecarlo_simulate(
        self,
        days: int = 252,
        n_paths: int = 1000,
        model: str = "gbm",
        rebalance_daily: bool = True,
        seed: Optional[int] = None,
        q: float = 0.05,
    ):
        """
        Ejecuta una simulación Monte Carlo de la cartera.
        Guarda internamente las trayectorias y su resumen.
        """
        self.sim_paths = simulate_portfolio_paths(
            self.series,
            self.weights,
            days=days,
            n_paths=n_paths,
            model=model,
            seed=seed,
            rebalance_daily=rebalance_daily,
            initial_value=self.initial_value,
        )

        self.sim_summary = summarize_paths(self.sim_paths, q=q)
        print(f"✅ Simulación Monte Carlo completada ({n_paths} paths, {days} días).")

    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    def plot_simulation(self, show_paths: bool = False):
        """
        Grafica el resultado de la simulación.
        - Si show_paths=True, dibuja varias trayectorias.
        - Siempre muestra la media y las bandas de confianza.
        """
        if self.sim_summary is None:
            raise RuntimeError("No hay resultados de simulación. Ejecuta primero montecarlo_simulate().")

        mean = self.sim_summary["mean"]
        p_low = self.sim_summary["p_low"]
        p_high = self.sim_summary["p_high"]

        plt.figure(figsize=(9, 5))
        t = np.arange(len(mean))
        plt.fill_between(t, p_low, p_high, alpha=0.2, color="dodgerblue", label="5%-95%")
        plt.plot(t, mean, color="black", lw=2, label="Media")

        if show_paths and self.sim_paths is not None:
            for i in range(min(10, self.sim_paths.shape[0])):  # hasta 10 trayectorias
                plt.plot(t, self.sim_paths[i, :], color="gray", alpha=0.3, lw=0.8)

        plt.title(f"Simulación Monte Carlo — {self.name}")
        plt.xlabel("Días")
        plt.ylabel("Valor cartera")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

