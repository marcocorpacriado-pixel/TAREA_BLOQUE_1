from __future__ import annotations
import os, time, requests
from datetime import date, datetime
from typing import Iterable
from .base import DataProvider, Candle

API_URL = "https://www.alphavantage.co/query"

class AlphaVantage(DataProvider):
    name = "alphavantage"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Falta ALPHAVANTAGE_API_KEY (usa variable de entorno).")

    # === PRECIOS (acciones/ETFs/índices vía ETF) ===
    def historical_prices(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        interval: str = "daily",
    ) -> Iterable[Candle]:
        if interval != "daily":
            raise ValueError("Alpha Vantage: solo 'daily' en este ejemplo.")

        params = {
            "function": "TIME_SERIES_DAILY",  # endpoint gratuito
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }

        time.sleep(1)  # free tier ~5 req/min
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        key = next((k for k in data if "Time Series" in k), None)
        if not key:
            if "Note" in data:         raise RuntimeError(f"Rate limit Alpha Vantage: {data['Note']}")
            if "Information" in data:  raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
            if "Error Message" in data:raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
            raise RuntimeError(f"Respuesta inesperada: {data}")

        for ds, vals in data[key].items():
            d = datetime.strptime(ds, "%Y-%m-%d").date()
            if start and d < start: continue
            if end and d > end:     continue
            yield Candle(
                date=ds,
                open=float(vals["1. open"]),
                high=float(vals["2. high"]),
                low=float(vals["3. low"]),
                close=float(vals["4. close"]),
                volume=float(vals.get("5. volume", 0.0)),
            )

    # === FX DIARIO (EURUSD, USDJPY, ...) ===
    def fx_daily(
        self,
        from_symbol: str,
        to_symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> Iterable[Candle]:
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        time.sleep(1)
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        key = next((k for k in data if "Time Series FX" in k), None)
        if not key:
            if "Note" in data:         raise RuntimeError(f"Rate limit Alpha Vantage: {data['Note']}")
            if "Information" in data:  raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
            if "Error Message" in data:raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
            raise RuntimeError(f"Respuesta inesperada (FX): {data}")

        for ds, vals in data[key].items():
            d = datetime.strptime(ds, "%Y-%m-%d").date()
            if start and d < start: continue
            if end and d > end:     continue
            yield Candle(
                date=ds,
                open=float(vals["1. open"]),
                high=float(vals["2. high"]),
                low=float(vals["3. low"]),
                close=float(vals["4. close"]),
                volume=0.0,  # FX no tiene volumen coherente en AV
            )



     

