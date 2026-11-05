from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_LIMIT = 100_000
EARLIEST_REALTIME_DATE = "1776-07-04"
LATEST_REALTIME_DATE = "9999-12-31"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FredSeriesSpec:
    """Configuration for one FRED series."""

    series_id: str
    description: str


class FredClient:
    """Small client for FRED API v1 series endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        pause: float = 0.10,
    ) -> None:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path)

        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            msg = (
                "FRED_API_KEY не найден. "
                "Добавь FRED_API_KEY=... в .env в корне проекта."
            )
            raise ValueError(msg)

        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    def _request_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        request_params["api_key"] = self.api_key
        request_params["file_type"] = "json"

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        response = self.session.get(
            url,
            params=request_params,
            timeout=self.timeout,
        )

        if not response.ok:
            body = response.text[:1500]
            msg = (
                f"FRED API returned HTTP {response.status_code}. "
                f"Endpoint: {endpoint}. Response: {body}"
            )
            raise requests.HTTPError(msg, response=response)

        return response.json()

    @staticmethod
    def _normalize_observations(
        payload: dict[str, Any],
        *,
        series_id: str,
    ) -> pd.DataFrame:
        rows = payload.get("observations", [])
        if not rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "value",
                    "realtime_start",
                    "realtime_end",
                    "series_id",
                ]
            )

        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

        for column in ["realtime_start", "realtime_end"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
            else:
                frame[column] = pd.NaT

        frame["series_id"] = series_id

        columns = [
            "date",
            "value",
            "realtime_start",
            "realtime_end",
            "series_id",
        ]

        return (
            frame[columns]
            .drop_duplicates(
                subset=["date", "realtime_start", "series_id"],
                keep="last",
            )
            .sort_values(["date", "realtime_start"])
            .reset_index(drop=True)
        )

    def get_series_info(self, series_id: str) -> dict[str, Any]:
        """Return FRED metadata for one series."""

        payload = self._request_json(
            "series",
            params={"series_id": series_id},
        )
        series = payload.get("seriess", [])
        if not series:
            msg = f"FRED metadata not found for series {series_id}."
            raise LookupError(msg)

        return series[0]

    def fetch_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
        *,
        output_type: int = 1,
        vintage_date: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        """Download observations for one FRED series."""

        params: dict[str, Any] = {
            "series_id": series_id,
            "limit": DEFAULT_LIMIT,
            "sort_order": "asc",
            "output_type": output_type,
        }

        if start is not None:
            params["observation_start"] = start
        if end is not None:
            params["observation_end"] = end

        if vintage_date is not None:
            params["realtime_start"] = vintage_date
            params["realtime_end"] = vintage_date
        elif output_type == 4:
            params["realtime_start"] = realtime_start or EARLIEST_REALTIME_DATE
            params["realtime_end"] = realtime_end or LATEST_REALTIME_DATE
        else:
            if realtime_start is not None:
                params["realtime_start"] = realtime_start
            if realtime_end is not None:
                params["realtime_end"] = realtime_end

        payload = self._request_json(
            "series/observations",
            params=params,
        )
        return self._normalize_observations(
            payload,
            series_id=series_id,
        )

    def fetch_many(
        self,
        series: dict[str, FredSeriesSpec],
        start: str | None = None,
        end: str | None = None,
        *,
        output_type: int = 1,
    ) -> pd.DataFrame:
        """Download several FRED series into one wide DataFrame."""

        frames: list[pd.DataFrame] = []

        for name, spec in series.items():
            logger.info("Downloading %s (%s)", name, spec.series_id)

            frame = self.fetch_series(
                spec.series_id,
                start=start,
                end=end,
                output_type=output_type,
            )

            if frame.empty:
                logger.warning("No data returned for %s", name)
                continue

            frames.append(frame[["date", "value"]].rename(columns={"value": name}))
            time.sleep(self.pause)

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(
                frame,
                on="date",
                how="outer",
                validate="one_to_one",
            )

        return result.sort_values("date").reset_index(drop=True)

    def fetch_many_long(
        self,
        series: dict[str, FredSeriesSpec],
        start: str | None = None,
        end: str | None = None,
        *,
        output_type: int = 1,
    ) -> pd.DataFrame:
        """Download several FRED series while preserving real-time fields."""

        frames: list[pd.DataFrame] = []

        for name, spec in series.items():
            logger.info("Downloading %s (%s)", name, spec.series_id)

            frame = self.fetch_series(
                spec.series_id,
                start=start,
                end=end,
                output_type=output_type,
            )

            if frame.empty:
                logger.warning("No data returned for %s", name)
                continue

            frame = frame.copy()
            frame["name"] = name
            frame["description"] = spec.description
            frames.append(frame)
            time.sleep(self.pause)

        if not frames:
            return pd.DataFrame()

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["name", "date", "realtime_start"])
            .reset_index(drop=True)
        )

    def fetch_vintage(
        self,
        series_id: str,
        vintage_date: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Download a series exactly as available on one historical date."""

        return self.fetch_series(
            series_id,
            start=start,
            end=end,
            vintage_date=vintage_date,
        )

    def test_connection(self) -> bool:
        """Check API key and access with a short VIX request."""

        test = self.fetch_series(
            "VIXCLS",
            start="2026-01-01",
            end="2026-01-15",
        )
        if test.empty:
            msg = "FRED connection works, but the test series is empty."
            raise RuntimeError(msg)

        return True


FRED_DAILY_CONTROLS: dict[str, FredSeriesSpec] = {
    "usd_broad": FredSeriesSpec("DTWEXBGS", "Nominal Broad U.S. Dollar Index"),
    "vix": FredSeriesSpec("VIXCLS", "CBOE Volatility Index"),
    "fed_funds": FredSeriesSpec("DFF", "Federal Funds Effective Rate"),
    "treasury_2y": FredSeriesSpec(
        "DGS2",
        "2-Year Treasury Constant Maturity Rate",
    ),
    "treasury_10y": FredSeriesSpec(
        "DGS10",
        "10-Year Treasury Constant Maturity Rate",
    ),
    "yield_curve_10y2y": FredSeriesSpec(
        "T10Y2Y",
        "10-Year Treasury Minus 2-Year Treasury",
    ),
    "real_yield_10y": FredSeriesSpec(
        "DFII10",
        "10-Year Treasury Inflation-Indexed Yield",
    ),
    "breakeven_10y": FredSeriesSpec(
        "T10YIE",
        "10-Year Breakeven Inflation Rate",
    ),
    "baa_spread": FredSeriesSpec(
        "BAA10Y",
        "Baa Corporate Bond Yield Minus 10-Year Treasury",
    ),
}

FRED_MONTHLY_CONTROLS: dict[str, FredSeriesSpec] = {
    "cpi": FredSeriesSpec(
        "CPIAUCSL",
        "Consumer Price Index for All Urban Consumers",
    ),
    "industrial_production": FredSeriesSpec(
        "INDPRO",
        "Industrial Production: Total Index",
    ),
    "unemployment": FredSeriesSpec(
        "UNRATE",
        "Unemployment Rate",
    ),
}
