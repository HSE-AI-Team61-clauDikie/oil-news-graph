from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

BARCHART_HISTORY_URL = "https://ondemand.websol.barchart.com/getHistory.json"
WTI_ROOT_SYMBOL = "CL"
WTI_NEARBYS = (1, 2, 3, 4, 5, 6, 12)


@dataclass(frozen=True)
class BarchartHistoryRequest:
    nearby: int
    start_date: date
    end_date: date
    max_records: int = 10_000


class BarchartClient:
    def __init__(self, api_key: str, timeout: int = 120) -> None:
        if not api_key.strip():
            msg = "Barchart API key is empty."
            raise ValueError(msg)

        self.api_key = api_key.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    @staticmethod
    def _format_date(value: date) -> str:
        return value.strftime("%Y%m%d")

    def fetch_wti_nearby(
        self,
        request: BarchartHistoryRequest,
    ) -> pd.DataFrame:
        if request.nearby < 1:
            msg = "nearby must be positive."
            raise ValueError(msg)

        response = self.session.get(
            BARCHART_HISTORY_URL,
            params={
                "apikey": self.api_key,
                "symbol": WTI_ROOT_SYMBOL,
                "type": "dailyNearest",
                "startDate": self._format_date(request.start_date),
                "endDate": self._format_date(request.end_date),
                "maxRecords": request.max_records,
                "order": "asc",
                "volume": "contract",
                "nearby": request.nearby,
                "backAdjust": "false",
                "daysToExpiration": 0,
                "contractRoll": "expiration",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        status = payload.get("status") or {}

        if status.get("code") != 200:
            msg = (
                "Barchart API error for "
                f"nearby={request.nearby}: "
                f"{status.get('code')} {status.get('message')}"
            )
            raise RuntimeError(msg)

        frame = pd.DataFrame(payload.get("results") or [])

        if frame.empty:
            msg = f"Barchart returned no rows for nearby={request.nearby}."
            raise ValueError(msg)

        frame = frame.rename(
            columns={
                "tradingDay": "trading_day",
                "openInterest": "open_interest",
                "sessionCode": "session_code",
            }
        )

        frame["trading_day"] = pd.to_datetime(
            frame["trading_day"],
            errors="raise",
        )

        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(
                frame["timestamp"],
                errors="coerce",
                utc=True,
            )

        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        frame["nearby"] = request.nearby
        frame["source"] = "barchart_ondemand"
        frame["series_type"] = "dailyNearest"
        frame["back_adjusted"] = False
        frame["contract_roll"] = "expiration"
        frame["days_to_expiration"] = 0

        return (
            frame.sort_values("trading_day")
            .drop_duplicates(subset=["trading_day"], keep="last")
            .reset_index(drop=True)
        )

    def fetch_wti_nearbys(
        self,
        nearbys: tuple[int, ...],
        start_date: date,
        end_date: date,
        preloaded: dict[int, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        cached = preloaded or {}
        frames: list[pd.DataFrame] = []

        for nearby in nearbys:
            if nearby in cached:
                frame = cached[nearby].copy()
            else:
                frame = self.fetch_wti_nearby(
                    BarchartHistoryRequest(
                        nearby=nearby,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def build_close_curve(
        long_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        required = {"trading_day", "nearby", "close"}
        missing = required - set(long_frame.columns)

        if missing:
            msg = f"Missing Barchart columns: {sorted(missing)}"
            raise KeyError(msg)

        if long_frame.duplicated(["trading_day", "nearby"]).any():
            msg = "Duplicate trading_day/nearby rows in Barchart history."
            raise ValueError(msg)

        curve = (
            long_frame.pivot(
                index="trading_day",
                columns="nearby",
                values="close",
            )
            .rename(columns=lambda value: f"wti_f{value}")
            .reset_index()
            .sort_values("trading_day")
            .reset_index(drop=True)
        )

        for column in (
            "wti_f1",
            "wti_f2",
            "wti_f3",
            "wti_f4",
            "wti_f5",
            "wti_f6",
            "wti_f12",
        ):
            if column not in curve.columns:
                curve[column] = pd.NA

        curve["spread_f1_f2"] = curve["wti_f1"] - curve["wti_f2"]
        curve["spread_f2_f3"] = curve["wti_f2"] - curve["wti_f3"]
        curve["spread_f1_f3"] = curve["wti_f1"] - curve["wti_f3"]
        curve["spread_f1_f6"] = curve["wti_f1"] - curve["wti_f6"]
        curve["spread_f1_f12"] = curve["wti_f1"] - curve["wti_f12"]

        for column in (
            "spread_f1_f2",
            "spread_f2_f3",
            "spread_f1_f3",
            "spread_f1_f6",
            "spread_f1_f12",
        ):
            curve[f"delta_{column}"] = curve[column].diff()

        curve["source"] = "barchart_ondemand"
        curve["price_field"] = "close"
        curve["curve_definition"] = "dailyNearest_expiration_roll_dte0"

        return curve

    @staticmethod
    def build_symbol_curve(
        long_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        required = {"trading_day", "nearby", "symbol"}
        missing = required - set(long_frame.columns)

        if missing:
            msg = f"Missing Barchart symbol columns: {sorted(missing)}"
            raise KeyError(msg)

        return (
            long_frame.pivot(
                index="trading_day",
                columns="nearby",
                values="symbol",
            )
            .rename(columns=lambda value: f"wti_f{value}_symbol")
            .reset_index()
            .sort_values("trading_day")
            .reset_index(drop=True)
        )
