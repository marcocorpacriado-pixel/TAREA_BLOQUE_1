from __future__ import annotations
import os, requests
from datetime import date
from typing import Iterable
from .base import DataProvider, Candle

API_URL = "https://api.marketstack.com/v1/eod"

class MarketStack(DataProvider):
    name = "marketstack"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("MARKETSTACK_API_KEY")
        if not self.api_key:
            raise ValueError("Falta MARKETSTACK_API_KEY (usa .env o variable de entorno).")

    def historical_prices(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        interval: str = "daily",
    ) -> Iterable[Candle]:
        if interval != "daily":
            raise ValueError("Marketstack: ejemplo limitado a 'daily'.")

        params = {
            "access_key": self.api_key,
            "symbols": symbol,
            "limit": 1000,
        }
        if start:
            params["date_from"] = start.isoformat()
        if end:
            params["date_to"] = end.isoformat()

        offset = 0
        while True:
            p = dict(params)
            p["offset"] = offset
            r = requests.get(API_URL, params=p, timeout=30)
            r.raise_for_status()
            payload = r.json()

            data = payload.get("data", [])
            if not data:
                break

            for row in data:
                ds = str(row["date"])[:10]  # "2024-01-02T00:00:00+0000" -> "2024-01-02"
                yield Candle(
                    date=ds,
                    open=float(row.get("open") or 0.0),
                    high=float(row.get("high") or 0.0),
                    low=float(row.get("low") or 0.0),
                    close=float(row.get("close") or 0.0),
                    volume=float(row.get("volume") or 0.0),
                )

            pag = payload.get("pagination", {})
            total = pag.get("total", 0)
            count = pag.get("count", 0)
            offset += count
            if offset >= total or count == 0:
                break
