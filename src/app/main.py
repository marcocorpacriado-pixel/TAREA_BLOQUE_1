from __future__ import annotations
import os, csv, argparse
from pathlib import Path
from datetime import date
from typing import Optional
from dotenv import load_dotenv

from app.extractor.base import to_records
from app.extractor.alphavantage import AlphaVantage
from app.extractor.marketstack import MarketStack
from app.extractor.twelvedata import TwelveData
load_dotenv()
PROVIDERS = {"alphavantage": AlphaVantage,
            "marketstack" : MarketStack,
            "twelvedata" : TwelveData}

def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))

def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No hay datos para guardar.")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Guardado: {path}")

def run_cli():
    p = argparse.ArgumentParser(description="Extractor de datos financieros")
    p.add_argument("action", choices=["fetch"])
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), default="alphavantage")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--interval", choices=["daily"], default="daily")
    p.add_argument("--out", help="Ruta CSV salida")
    args = p.parse_args()

    Provider = PROVIDERS[args.provider]
    provider = Provider()

    start = parse_date(args.start)
    end = parse_date(args.end)

    candles = provider.historical_prices(symbol=args.symbol, start=start, end=end, interval=args.interval)
    records = to_records(candles)

    out = Path(args.out) if args.out else Path("data") / args.provider / f"{args.symbol}.csv"
    save_csv(out, records)

def main():
    run_cli()

if __name__ == "__main__":
    main()

