from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

BASE_URL = "https://api.eia.gov/v2"

logger = logging.getLogger(__name__)

SERIES_COLUMNS = [
    "period",
    "value",
    "series",
    "series_description",
    "units",
    "route",
    "frequency",
]


@dataclass(frozen=True)
class EIASeriesSpec:
    route: str
    series: str
    frequency: str


class EIAClient:
    """Client for the EIA Open Data API v2."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        page_size: int = 5000,
        pause: float = 0.10,
    ) -> None:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path)

        self.api_key = api_key or os.getenv("EIA_API_KEY")

        if not self.api_key:
            msg = (
                "EIA_API_KEY не найден. "
                "Создай .env в корне проекта и добавь:\n"
                "EIA_API_KEY=твой_ключ"
            )
            raise ValueError(msg)

        self.timeout = timeout
        self.page_size = min(int(page_size), 5000)
        self.pause = pause

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "oil-news-graph-coursework/1.0",
                "Accept": "application/json",
            }
        )

    @staticmethod
    def _clean_route(route: str) -> str:
        route = route.strip().strip("/")
        if route.endswith("/data"):
            route = route[:-5].rstrip("/")
        return route

    def _request_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        request_params["api_key"] = self.api_key

        response = self.session.get(
            url,
            params=request_params,
            timeout=self.timeout,
        )

        if not response.ok:
            response_body = response.text[:2000]
            msg = (
                f"EIA API returned HTTP {response.status_code}\n"
                f"URL: {response.url}\n"
                f"Response: {response_body}"
            )
            raise requests.HTTPError(msg, response=response)

        payload = response.json()

        if "error" in payload:
            msg = f"EIA API error: {payload['error']}"
            raise RuntimeError(msg)

        return payload

    def get_route_metadata(self, route: str) -> dict[str, Any]:
        route = self._clean_route(route)
        url = f"{BASE_URL}/{route}/"
        return self._request_json(url)

    def get_facet_values(
        self,
        route: str,
        facet: str = "series",
        offset: int = 0,
        length: int = 5000,
    ) -> pd.DataFrame:
        route = self._clean_route(route)
        url = f"{BASE_URL}/{route}/facet/{facet}/"

        payload = self._request_json(
            url,
            params={
                "offset": offset,
                "length": min(length, 5000),
            },
        )

        response = payload.get("response", {})
        data = response.get("facets", response.get("data", []))
        return pd.DataFrame(data)

    @staticmethod
    def _add_extra_facets(
        params: dict[str, Any],
        extra_facets: dict[str, str | list[str]] | None,
    ) -> None:
        if not extra_facets:
            return

        for facet_name, facet_value in extra_facets.items():
            values = facet_value if isinstance(facet_value, list) else [facet_value]
            for index, value in enumerate(values):
                params[f"facets[{facet_name}][{index}]"] = value

    def _build_data_params(
        self,
        *,
        series: str,
        frequency: str,
        value_column: str,
        offset: int,
        start: str | None,
        end: str | None,
        extra_facets: dict[str, str | list[str]] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "frequency": frequency,
            "data[0]": value_column,
            "facets[series][]": series,
            "offset": offset,
            "length": self.page_size,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }

        if start is not None:
            params["start"] = start

        if end is not None:
            params["end"] = end

        self._add_extra_facets(params, extra_facets)
        return params

    @staticmethod
    def _empty_series_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=SERIES_COLUMNS)

    @staticmethod
    def _normalize_series_frame(
        df: pd.DataFrame,
        *,
        series: str,
        route: str,
        frequency: str,
        value_column: str,
    ) -> pd.DataFrame:
        rename_map = {
            "seriesDescription": "series_description",
            "series-description": "series_description",
            "unit": "units",
        }
        df = df.rename(
            columns={
                key: value for key, value in rename_map.items() if key in df.columns
            }
        )

        if "period" in df.columns:
            df["period"] = pd.to_datetime(df["period"], errors="coerce")

        if value_column in df.columns:
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

        if value_column != "value":
            df = df.rename(columns={value_column: "value"})

        if "series" not in df.columns:
            df["series"] = series

        if "series_description" not in df.columns:
            df["series_description"] = None

        if "units" not in df.columns:
            df["units"] = None

        df["route"] = route
        df["frequency"] = frequency

        for column in SERIES_COLUMNS:
            if column not in df.columns:
                df[column] = None

        return (
            df[SERIES_COLUMNS]
            .drop_duplicates(subset=["period", "series"])
            .sort_values("period")
            .reset_index(drop=True)
        )

    def fetch_series(
        self,
        route: str,
        series: str,
        frequency: str,
        start: str | None = None,
        end: str | None = None,
        value_column: str = "value",
        extra_facets: dict[str, str | list[str]] | None = None,
    ) -> pd.DataFrame:
        route = self._clean_route(route)
        url = f"{BASE_URL}/{route}/data/"

        offset = 0
        frames: list[pd.DataFrame] = []

        while True:
            params = self._build_data_params(
                series=series,
                frequency=frequency,
                value_column=value_column,
                offset=offset,
                start=start,
                end=end,
                extra_facets=extra_facets,
            )

            payload = self._request_json(url=url, params=params)
            response = payload.get("response", {})
            rows = response.get("data", [])

            if not rows:
                break

            page = pd.DataFrame(rows)
            frames.append(page)

            total = int(response.get("total", offset + len(page)))
            offset += len(page)

            if offset >= total:
                break

            time.sleep(self.pause)

        if not frames:
            return self._empty_series_frame()

        df = pd.concat(frames, ignore_index=True)
        return self._normalize_series_frame(
            df,
            series=series,
            route=route,
            frequency=frequency,
            value_column=value_column,
        )

    def fetch_spec(
        self,
        spec: EIASeriesSpec,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return self.fetch_series(
            route=spec.route,
            series=spec.series,
            frequency=spec.frequency,
            start=start,
            end=end,
        )

    def fetch_many(
        self,
        series: dict[str, EIASeriesSpec | dict[str, str]],
        start: str | None = None,
        end: str | None = None,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for name, raw_spec in series.items():
            if isinstance(raw_spec, EIASeriesSpec):
                spec = raw_spec
            else:
                spec = EIASeriesSpec(
                    route=raw_spec["route"],
                    series=raw_spec["series"],
                    frequency=raw_spec["frequency"],
                )

            if verbose:
                logger.info(
                    "Downloading %s: %s | %s | %s",
                    name,
                    spec.route,
                    spec.series,
                    spec.frequency,
                )

            try:
                df = self.fetch_spec(spec=spec, start=start, end=end)
            except Exception as exc:
                msg = f"Ошибка при загрузке '{name}' ({spec.series})"
                raise RuntimeError(msg) from exc

            if df.empty:
                if verbose:
                    logger.warning("No data returned for %s", name)
                continue

            tmp = df[["period", "value"]].copy()
            tmp = tmp.rename(columns={"value": name})
            frames.append(tmp)

            if verbose:
                first_date = tmp["period"].min()
                last_date = tmp["period"].max()
                first_text = first_date.date() if pd.notna(first_date) else "NA"
                last_text = last_date.date() if pd.notna(last_date) else "NA"

                logger.info(
                    "%s observations | %s -> %s",
                    f"{len(tmp):,}",
                    first_text,
                    last_text,
                )

            time.sleep(self.pause)

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(
                frame,
                on="period",
                how="outer",
                validate="one_to_one",
            )

        return result.sort_values("period").reset_index(drop=True)

    def test_connection(self) -> bool:
        test = self.fetch_series(
            route="petroleum/stoc/wstk",
            series="WCESTUS1",
            frequency="weekly",
            start="2025-01-01",
            end="2025-01-31",
        )

        if test.empty:
            msg = "Соединение с EIA установлено, но тестовый ряд пуст."
            raise RuntimeError(msg)

        return True


EIA_WEEKLY_SERIES: dict[str, dict[str, str]] = {
    "crude_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "WCESTUS1",
        "frequency": "weekly",
    },
    "cushing_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "W_EPC0_SAX_YCUOK_MBBL",
        "frequency": "weekly",
    },
    "padd1_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "WCESTP11",
        "frequency": "weekly",
    },
    "padd2_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "WCESTP21",
        "frequency": "weekly",
    },
    "padd3_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "WCESTP31",
        "frequency": "weekly",
    },
    "padd4_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "WCESTP41",
        "frequency": "weekly",
    },
    "padd5_stocks": {
        "route": "petroleum/stoc/wstk",
        "series": "WCESTP51",
        "frequency": "weekly",
    },
    "crude_production": {
        "route": "petroleum/sum/sndw",
        "series": "WCRFPUS2",
        "frequency": "weekly",
    },
}


EIA_WTI_FUTURES_SERIES: dict[str, dict[str, str]] = {
    "wti_f1": {
        "route": "petroleum/pri/fut",
        "series": "RCLC1",
        "frequency": "daily",
    },
    "wti_f2": {
        "route": "petroleum/pri/fut",
        "series": "RCLC2",
        "frequency": "daily",
    },
    "wti_f3": {
        "route": "petroleum/pri/fut",
        "series": "RCLC3",
        "frequency": "daily",
    },
    "wti_f4": {
        "route": "petroleum/pri/fut",
        "series": "RCLC4",
        "frequency": "daily",
    },
}
