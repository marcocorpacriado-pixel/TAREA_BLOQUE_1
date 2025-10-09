from __future__ import annotations
import os, csv, argparse, json
from pathlib import Path
from datetime import date
from typing import Optional, Iterable, List, Dict

from dotenv import load_dotenv

# === Extractor (descarga) ===
from app.extractor.base import to_records, Candle
from app.extractor.alphavantage import AlphaVantage
from app.extractor.marketstack import MarketStack
from app.extractor.twelvedata import TwelveData

# === Series, cartera y simulación ===
from app.models.series import PriceSeries
from app.analytics.simulate import (
    simulate_asset_price_paths,
    simulate_portfolio_paths,
    summarize_paths,
)
from app.analytics.montecarlo import simulate_random_portfolios

try:
    import pandas as pd
except Exception:
    pd = None

load_dotenv()

PROVIDERS = {
    "alphavantage": AlphaVantage,
    "marketstack": MarketStack,
    "twelvedata": TwelveData,
}

def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))

def ensure_records(records: List[Dict], provider: str, symbol: str, datatype: str):
    for r in records:
        r["provider"] = provider
        r["symbol"] = symbol
        r["datatype"] = datatype
    cols = ["provider", "datatype", "symbol", "date", "open", "high", "low", "close", "volume"]
    return [{c: row.get(c) for c in cols} for row in records]

def save_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No hay datos para guardar: {path}")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Guardado CSV: {path}")

def save_parquet(path: Path, rows: List[Dict]):
    if pd is None:
        print("pandas no instalado; guardando CSV en su lugar.")
        return save_csv(path.with_suffix(".csv"), rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No hay datos para guardar: {path}")
        return
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"Guardado Parquet: {path}")

def load_symbols(symbols: Optional[str], symbols_file: Optional[str]) -> List[str]:
    items: List[str] = []
    if symbols:
        items += [s.strip() for s in symbols.split(",") if s.strip()]
    if symbols_file:
        p = Path(symbols_file)
        if not p.exists():
            raise FileNotFoundError(f"No existe symbols_file: {symbols_file}")
        with p.open("r", encoding="utf-8") as f:
            items += [line.strip() for line in f if line.strip()]
    seen = set()
    uniq = []
    for s in items:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    if not uniq:
        raise ValueError("No se han indicado símbolos (usa --symbol, --symbols o --symbols-file).")
    return uniq

def fetch_prices(provider, symbol: str, start: Optional[date], end: Optional[date]) -> List[Dict]:
    candles: Iterable[Candle] = provider.historical_prices(symbol=symbol, start=start, end=end, interval="daily")
    return to_records(candles)

def fetch_fx(provider, pair: str, start: Optional[date], end: Optional[date]) -> List[Dict]:
    pair = pair.replace(" ", "")
    if "/" in pair:
        frm, to = pair.split("/", 1)
    else:
        frm, to = pair[:3], pair[3:]
    if not hasattr(provider, "fx_daily"):
        raise NotImplementedError(f"{provider.name} no implementa fx_daily().")
    candles: Iterable[Candle] = provider.fx_daily(frm, to, start=start, end=end)
    return to_records(candles)

# === Helpers para SIMULATE ===
def load_priceseries_from_csv(base_dir: str, provider: str, datatype: str, symbol: str) -> PriceSeries:
    """
    Lee el CSV estandarizado creado por 'fetch' y devuelve PriceSeries.
    Ruta esperada: {base_dir}/{provider}/{datatype}/{symbol}.csv
    """
    csv_path = Path(base_dir) / provider / datatype / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}. Ejecuta primero 'fetch' para {symbol}.")
    if pd is None:
        raise RuntimeError("pandas es necesario para cargar CSV en PriceSeries.")
    df = pd.read_csv(csv_path)
    candles = [
        Candle(
            date=str(row["date"])[:10],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in df.iterrows()
    ]
    return PriceSeries(provider=provider, datatype=datatype, symbol=symbol, candles=candles)

def run_fetch(args: argparse.Namespace):
    Provider = PROVIDERS[args.provider]
    provider = Provider()
    start = parse_date(args.start)
    end = parse_date(args.end)

    series: List[str] = []
    if args.symbol:
        series = [args.symbol]
    series = load_symbols(args.symbols, args.symbols_file) if (args.symbols or args.symbols_file) else series
    if not series:
        raise ValueError("Debes indicar símbolos.")

    for sym in series:
        if args.datatype == "prices":
            rows = fetch_prices(provider, sym, start, end)
        else:
            rows = fetch_fx(provider, sym, start, end)
        rows = ensure_records(rows, provider=provider.name, symbol=sym, datatype=args.datatype)
        outdir = Path(args.outdir)
        ext = "parquet" if args.format == "parquet" else "csv"
        out = outdir / provider.name / args.datatype / f"{sym}.{ext}"
        if args.format == "parquet":
            save_parquet(out, rows)
        else:
            save_csv(out, rows)

def run_simulate(args: argparse.Namespace):
    """
    Simula activos o cartera leyendo CSVs previamente descargados.
    """
    # 1) Cargar series
    symbols = load_symbols(args.symbols, args.symbols_file) if args.symbols or args.symbols_file else ([args.symbol] if args.symbol else [])
    if not symbols:
        raise ValueError("Debes indicar símbolos (asset o componentes de cartera).")
    # provider/datatype nos dicen en qué carpeta buscar los CSV
    provider = args.provider
    datatype = args.datatype  # 'prices' o 'fx'
    base_dir = args.outdir_data  # carpeta donde están los CSV (normalmente 'data')

    if args.level == "asset":
        if len(symbols) != 1:
            raise ValueError("Para level=asset indica exactamente UN símbolo (usa --symbol o --symbols 1 elemento).")
        sym = symbols[0]
        ps = load_priceseries_from_csv(base_dir, provider, datatype, sym)
        # 2) Simular
        paths = simulate_asset_price_paths(ps, days=args.days, n_paths=args.n_paths, model=args.model,
                                           seed=args.seed)
        summ = summarize_paths(paths, q=args.q)
        # 3) Guardar resumen (media/p_low/p_high por día)
        outdir = Path(args.outdir_sim) / "asset" / provider / datatype / sym
        outdir.mkdir(parents=True, exist_ok=True)
        # Guardar arrays
        if pd is None:
            # Guardar como CSVs simples
            rows = [{"t": i, "mean": float(summ["mean"][i]), "p_low": float(summ["p_low"][i]), "p_high": float(summ["p_high"][i])}
                    for i in range(len(summ["mean"]))]
            save_csv(outdir / "summary.csv", rows)
        else:
            import numpy as np
            df = pd.DataFrame({
                "t": np.arange(paths.shape[1]),
                "mean": summ["mean"],
                "p_low": summ["p_low"],
                "p_high": summ["p_high"],
            })
            df.to_csv(outdir / "summary.csv", index=False)
        print(f"✅ Simulación asset guardada en: {outdir}")

    else:  # level == "portfolio"
        # 1) cargar todas las series componentes
        series: Dict[str, PriceSeries] = {}
        for s in symbols:
            series[s] = load_priceseries_from_csv(base_dir, provider, datatype, s)

        # 2) pesos
        if args.weights:
            ws = [float(x) for x in args.weights.split(",")]
            if len(ws) != len(symbols):
                raise ValueError("El número de pesos debe coincidir con el número de símbolos.")
            ssum = sum(ws)
            if ssum <= 0:
                raise ValueError("La suma de pesos debe ser > 0.")
            weights = {sym: w/ssum for sym, w in zip(symbols, ws)}
        else:
            # iguales si no se pasan
            w = 1.0 / len(symbols)
            weights = {sym: w for sym in symbols}

        # 3) simular
        paths = simulate_portfolio_paths(series, weights, days=args.days, n_paths=args.n_paths, model=args.model,
                                         seed=args.seed, rebalance_daily=not args.buy_and_hold,
                                         initial_value=args.initial_value)
        summ = summarize_paths(paths, q=args.q)

        # 4) guardar
        outdir = Path(args.outdir_sim) / "portfolio" / provider / datatype / ("_".join(symbols))
        outdir.mkdir(parents=True, exist_ok=True)
        if pd is None:
            rows = [{"t": i, "mean": float(summ["mean"][i]), "p_low": float(summ["p_low"][i]), "p_high": float(summ["p_high"][i])}
                    for i in range(len(summ["mean"]))]
            save_csv(outdir / "summary.csv", rows)
        else:
            import numpy as np
            df = pd.DataFrame({
                "t": np.arange(paths.shape[1]),
                "mean": summ["mean"],
                "p_low": summ["p_low"],
                "p_high": summ["p_high"],
            })
            df.to_csv(outdir / "summary.csv", index=False)

        # guardar pesos usados
        with open(outdir / "weights.json", "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)
        print(f"✅ Simulación cartera guardada en: {outdir}")

def main():
    p = argparse.ArgumentParser(description="Extractor y simulador financiero multi-API.")
    sub = p.add_subparsers(dest="action", required=True)

    # ---- fetch (ya lo tenías)
    p_fetch = sub.add_parser("fetch", help="Descargar series desde proveedores (CSV/Parquet).")
    p_fetch.add_argument("--provider", choices=list(PROVIDERS.keys()), default="alphavantage")
    p_fetch.add_argument("--datatype", choices=["prices", "fx"], default="prices")
    p_fetch.add_argument("--symbol")
    p_fetch.add_argument("--symbols")
    p_fetch.add_argument("--symbols-file")
    p_fetch.add_argument("--start", help="YYYY-MM-DD")
    p_fetch.add_argument("--end", help="YYYY-MM-DD")
    p_fetch.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p_fetch.add_argument("--outdir", default="data")
    p_fetch.set_defaults(func=run_fetch)

    # ---- simulate (nuevo)
    p_sim = sub.add_parser("simulate", help="Simular activo o cartera a partir de CSVs.")
    p_sim.add_argument("--level", choices=["asset", "portfolio"], default="asset", help="Simular un activo o una cartera.")
    p_sim.add_argument("--provider", choices=list(PROVIDERS.keys()), default="alphavantage", help="Dónde están los CSV.")
    p_sim.add_argument("--datatype", choices=["prices", "fx"], default="prices", help="Tipo de series a leer (carpeta).")
    # Input de símbolos
    p_sim.add_argument("--symbol", help="Un único símbolo (para level=asset).")
    p_sim.add_argument("--symbols", help="Lista separada por comas (para portfolio o varios assets).")
    p_sim.add_argument("--symbols-file", help="Fichero con símbolos (uno por línea).")
    # Pesos (solo portfolio)
    p_sim.add_argument("--weights", help="Pesos coma-separados (mismo nº que símbolos). Si no, iguales.")
    # Parámetros de simulación
    p_sim.add_argument("--model", choices=["gbm", "bootstrap"], default="gbm")
    p_sim.add_argument("--days", type=int, default=252)
    p_sim.add_argument("--n-paths", type=int, default=1000)
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.add_argument("--q", type=float, default=0.05, help="Cuantil para bandas (ej 0.05 → 5%/95%).")
    p_sim.add_argument("--buy-and-hold", action="store_true", help="Simular cartera Buy&Hold (por defecto rebalance diario).")
    p_sim.add_argument("--initial-value", type=float, default=1.0)
    # Dónde leer CSV y dónde guardar simulación
    p_sim.add_argument("--outdir-data", default="data", help="Carpeta base donde están los CSV de datos.")
    p_sim.add_argument("--outdir-sim", dest="outdir_sim", default="data/simulations", help="Carpeta donde guardar resultados.")
    p_sim.set_defaults(func=run_simulate)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

