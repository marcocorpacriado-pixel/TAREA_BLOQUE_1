from __future__ import annotations
import os, csv, argparse
from pathlib import Path
from datetime import date
from typing import Optional, Iterable, List, Dict
from dotenv import load_dotenv

from app.extractor.base import to_records, Candle
from app.extractor.alphavantage import AlphaVantage
from app.extractor.marketstack import MarketStack
from app.extractor.twelvedata import TwelveData

try:
    import pandas as pd
except Exception:
    pd = None  # parquet requerirá pandas si se usa

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
    """Añade columnas estándar a cada fila."""
    for r in records:
        r["provider"] = provider
        r["symbol"] = symbol
        r["datatype"] = datatype
    # Orden de columnas coherente:
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
    # quitar duplicados conservando orden
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
    """
    pair puede venir como 'EURUSD' o 'EUR/USD'.
    """
    pair = pair.replace(" ", "")
    if "/" in pair:
        frm, to = pair.split("/", 1)
    else:
        frm, to = pair[:3], pair[3:]
    if not hasattr(provider, "fx_daily"):
        raise NotImplementedError(f"{provider.name} no implementa fx_daily(). Elige otro provider.")
    candles: Iterable[Candle] = provider.fx_daily(frm, to, start=start, end=end)
    return to_records(candles)

def run_cli():
    p = argparse.ArgumentParser(description="Extractor financiero multi-API con salida estandarizada.")
    p.add_argument("action", choices=["fetch"], help="Acción")
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), default="alphavantage")
    p.add_argument("--datatype", choices=["prices", "fx"], default="prices", help="Tipo de datos a extraer")
    # Fuentes de símbolos
    p.add_argument("--symbol", help="Símbolo único (p.ej. AAPL o EURUSD/EUR/USD)")
    p.add_argument("--symbols", help="Lista separada por comas (p.ej. AAPL,MSFT,GOOG o EURUSD,USDJPY)")
    p.add_argument("--symbols-file", help="Fichero con un símbolo por línea")
    # Fechas y formato
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--outdir", help="Carpeta base de salida", default="data")
    args = p.parse_args()

    Provider = PROVIDERS[args.provider]
    provider = Provider()

    start = parse_date(args.start)
    end = parse_date(args.end)

    # Construir lista de series a descargar
    series: List[str] = []
    if args.symbol:
        series = [args.symbol]
    series = load_symbols(args.symbols, args.symbols_file) if (args.symbols or args.symbols_file) else series
    if not series:
        raise ValueError("Debes indicar al menos un símbolo (--symbol, --symbols o --symbols-file).")

    # Descarga en lote
    for sym in series:
        if args.datatype == "prices":
            rows = fetch_prices(provider, sym, start, end)
        else:  # fx
            rows = fetch_fx(provider, sym, start, end)

        rows = ensure_records(rows, provider=provider.name, symbol=sym, datatype=args.datatype)

        # ruta salida: data/{provider}/{datatype}/{symbol}.(csv|parquet)
        ext = "parquet" if args.format == "parquet" else "csv"
        outpath = Path(args.outdir) / provider.name / args.datatype / f"{sym}.{ext}"
        if args.format == "parquet":
            save_parquet(outpath, rows)
        else:
            save_csv(outpath, rows)

def main():
    run_cli()

if __name__ == "__main__":
    main()



