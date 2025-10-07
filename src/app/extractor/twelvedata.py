from __future__ import annotations
import os, requests
from datetime import date
from typing import Iterable
from .base import DataProvider, Candle

API_URL = "https://api.twelvedata.com/time_series"

class TwelveData(DataProvider):
    name = "twelvedata"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise ValueError("Falta TWELVEDATA_API_KEY (usa .env o variable de entorno).")

    # PRECIOS (usa symbols de acciones/ETFs/índices-ETF)
    def historical_prices(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        interval: str = "daily",
    ) -> Iterable[Candle]:
        if interval != "daily":
            raise ValueError("Twelve Data: ejemplo limitado a 'daily'.")

        params = {
            "symbol": symbol,       # p.ej. AAPL o SPY
            "interval": "1day",
            "outputsize": 5000,
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC",
        }
        if start:
            params["start_date"] = start.isoformat()
        if end:
            params["end_date"] = end.isoformat()

        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData error: {data.get('message')}")

        values = data.get("values") or data.get("data") or []
        for row in values:
            ds = str(row["datetime"])[:10]
            yield Candle(
                date=ds,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
            )

    # FX DIARIO (símbolos 'EUR/USD', 'USD/JPY', ...)
    def fx_daily(
        self,
        from_symbol: str,
        to_symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> Iterable[Candle]:
        symbol = f"{from_symbol}/{to_symbol}"  # formato de TwelveData
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 5000,
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC",
        }
        if start:
            params["start_date"] = start.isoformat()
        if end:
            params["end_date"] = end.isoformat()

        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData error (fx): {data.get('message')}")

        values = data.get("values") or data.get("data") or []
        for row in values:
            ds = str(row["datetime"])[:10]
            yield Candle(
                date=ds,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=0.0,  # FX: volumen no consistente entre fuentes
            )
