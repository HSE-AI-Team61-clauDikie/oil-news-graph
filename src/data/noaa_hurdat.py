from __future__ import annotations

import logging
import re
from io import StringIO

import pandas as pd
import requests

ATLANTIC_HURDAT_URL = (
    "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"
)

logger = logging.getLogger(__name__)

TRACK_COLUMNS = [
    "date",
    "time",
    "record_identifier",
    "status",
    "latitude",
    "longitude",
    "max_wind_kt",
    "min_pressure_mb",
    "r34_ne",
    "r34_se",
    "r34_sw",
    "r34_nw",
    "r50_ne",
    "r50_se",
    "r50_sw",
    "r50_nw",
    "r64_ne",
    "r64_se",
    "r64_sw",
    "r64_nw",
]


def _parse_coordinate(value: str) -> float:
    value = value.strip()
    direction = value[-1]
    number = float(value[:-1])

    if direction in {"S", "W"}:
        number *= -1

    return number


def saffir_simpson_category(max_wind_kt: float) -> int:
    """Return Saffir-Simpson category from maximum sustained wind."""

    if max_wind_kt < 64:
        return 0
    if max_wind_kt < 83:
        return 1
    if max_wind_kt < 96:
        return 2
    if max_wind_kt < 113:
        return 3
    if max_wind_kt < 137:
        return 4
    return 5


class NoaaHurdatClient:
    """Downloader and parser for Atlantic HURDAT2 best-track data."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    def download_text(self) -> str:
        """Download the pinned Atlantic HURDAT2 release."""

        response = self.session.get(
            ATLANTIC_HURDAT_URL,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def parse(text: str) -> pd.DataFrame:
        """Parse HURDAT2 text into one row per six-hour track point."""

        rows: list[dict[str, object]] = []
        storm_id = ""
        storm_name = ""

        for raw_line in StringIO(text):
            parts = [part.strip() for part in raw_line.strip().split(",")]

            if not parts or not parts[0]:
                continue

            if re.fullmatch(r"[A-Z]{2}\d{6}", parts[0]):
                storm_id = parts[0]
                storm_name = parts[1]
                continue

            if len(parts) < len(TRACK_COLUMNS):
                continue

            record = dict(zip(TRACK_COLUMNS, parts, strict=False))
            record["storm_id"] = storm_id
            record["storm_name"] = storm_name
            rows.append(record)

        frame = pd.DataFrame(rows)

        frame["latitude"] = frame["latitude"].map(_parse_coordinate)
        frame["longitude"] = frame["longitude"].map(_parse_coordinate)

        numeric_columns = [
            "max_wind_kt",
            "min_pressure_mb",
            "r34_ne",
            "r34_se",
            "r34_sw",
            "r34_nw",
            "r50_ne",
            "r50_se",
            "r50_sw",
            "r50_nw",
            "r64_ne",
            "r64_se",
            "r64_sw",
            "r64_nw",
        ]

        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            frame[column] = frame[column].replace(-999, pd.NA)

        frame["datetime"] = pd.to_datetime(
            frame["date"].astype(str) + frame["time"].astype(str).str.zfill(4),
            format="%Y%m%d%H%M",
            utc=True,
            errors="coerce",
        )

        frame["year"] = frame["datetime"].dt.year
        frame["category"] = (
            frame["max_wind_kt"].fillna(0).astype(float).map(saffir_simpson_category)
        )

        columns = [
            "storm_id",
            "storm_name",
            "datetime",
            "year",
            "status",
            "record_identifier",
            "latitude",
            "longitude",
            "max_wind_kt",
            "min_pressure_mb",
            "category",
            "r34_ne",
            "r34_se",
            "r34_sw",
            "r34_nw",
            "r50_ne",
            "r50_se",
            "r50_sw",
            "r50_nw",
            "r64_ne",
            "r64_se",
            "r64_sw",
            "r64_nw",
        ]

        return (
            frame[columns].sort_values(["storm_id", "datetime"]).reset_index(drop=True)
        )

    def fetch(
        self,
        *,
        start_year: int = 2015,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """Download, parse, and filter Atlantic HURDAT2."""

        frame = self.parse(self.download_text())
        return frame[
            frame["year"].between(
                start_year,
                end_year,
            )
        ].reset_index(drop=True)
