from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import requests

DEFAULT_EVENT_ID = -1065094132
DEFAULT_YEARS_BACK = 12
BASE_URL = "https://www.myfxbook.com/calendar-chart-years-back.json"
SOURCE_NAME = "myfxbook"
EVENT_NAME = "EIA Crude Oil Stocks Change"

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type PointSeries = list[tuple[pd.Timestamp, float | None]]

_DATE_KEYS = {
    "date",
    "datetime",
    "releasedate",
    "releasedatetime",
    "time",
    "timestamp",
    "x",
}
_ACTUAL_KEYS = {"actual", "value", "actualvalue"}
_CONSENSUS_KEYS = {"consensus", "forecast", "expected", "estimate"}
_PREVIOUS_KEYS = {"previous", "prev", "prior"}
_EVENT_KEYS = {"event", "eventname", "name", "title", "label"}
_SERIES_ALIASES = {
    "actual_mmbbl": _ACTUAL_KEYS,
    "consensus_mmbbl": _CONSENSUS_KEYS,
    "previous_mmbbl": _PREVIOUS_KEYS,
}


@dataclass(frozen=True)
class MyfxbookConfig:
    event_id: int = DEFAULT_EVENT_ID
    years_back: int = DEFAULT_YEARS_BACK
    timeout: int = 60


class MyfxbookConsensusClient:
    """Download and normalize Myfxbook EIA crude-oil consensus history."""

    def __init__(
        self,
        config: MyfxbookConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or MyfxbookConfig()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Referer": (
                    "https://www.myfxbook.com/forex-economic-calendar/"
                    "united-states/eia-crude-oil-stocks-change"
                ),
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/156.0 Safari/537.36"
                ),
            }
        )

    @property
    def url(self) -> str:
        return BASE_URL

    @property
    def params(self) -> dict[str, int]:
        return {
            "eventId": self.config.event_id,
            "yearsBack": self.config.years_back,
        }

    def fetch_raw(self) -> JsonValue:
        response = self.session.get(
            self.url,
            params=self.params,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return cast("JsonValue", response.json())

    @staticmethod
    def save_raw(payload: JsonValue, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    def fetch(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        payload = self.fetch_raw()
        return normalize_myfxbook_payload(payload, start=start, end=end)


def _compact_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _first_value(
    row: Mapping[str, object],
    aliases: set[str],
) -> object | None:
    for key, value in row.items():
        if _compact_key(key) in aliases:
            return value
    return None


def _extract_number_and_suffix(
    text: str,
) -> tuple[float, str] | None:
    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*([kmb]?)",
        text.casefold(),
    )

    if match is None:
        return None

    return float(match.group(1)), match.group(2)


def _convert_to_millions(
    number: float,
    suffix: str,
    *,
    default_unit: str,
) -> float:
    if suffix == "k":
        return number / 1000

    if suffix == "b":
        return number * 1000

    if suffix == "m":
        return number

    if default_unit == "million":
        return number

    return number


def parse_million_barrels(
    value: object,
    *,
    default_unit: str = "million",
) -> float | None:
    """Convert Myfxbook-style values to million barrels."""
    if value is None or value is pd.NA:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    text = str(value).strip().replace(",", "")

    if not text:
        return None

    if text.casefold() in {"-", "—", "na", "n/a", "null", "none"}:
        return None

    parsed = _extract_number_and_suffix(text)

    if parsed is None:
        return None

    number, suffix = parsed

    return _convert_to_millions(
        number,
        suffix,
        default_unit=default_unit,
    )


def _parse_datetime(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        unit = "ms" if abs(numeric) > 10_000_000_000 else "s"
        return pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    text = str(value).strip()

    if not text:
        return pd.NaT

    if re.fullmatch(r"\d{10,13}", text):
        numeric = int(text)
        unit = "ms" if len(text) >= 12 else "s"
        return pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    return pd.to_datetime(
        text,
        utc=True,
        errors="coerce",
    )


def _is_row_candidate(node: Mapping[str, object]) -> bool:
    compact_keys = {_compact_key(key) for key in node}

    has_date = bool(compact_keys & _DATE_KEYS)

    value_groups = sum(
        bool(compact_keys & aliases)
        for aliases in (
            _ACTUAL_KEYS,
            _CONSENSUS_KEYS,
            _PREVIOUS_KEYS,
        )
    )

    return has_date and value_groups >= 1


def _row_like_dicts(
    payload: JsonValue,
) -> list[dict[str, JsonValue]]:
    candidates: list[dict[str, JsonValue]] = []

    def walk(node: JsonValue) -> None:
        if isinstance(node, dict):
            if _is_row_candidate(node):
                candidates.append(node.copy())

            for value in node.values():
                walk(value)

            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return candidates


def _point_series(node: object) -> PointSeries:
    if not isinstance(node, list):
        return []

    points: PointSeries = []

    for item in node:
        x_value: object | None = None
        y_value: object | None = None

        if isinstance(item, dict):
            x_value = _first_value(
                item,
                _DATE_KEYS,
            )
            y_value = _first_value(
                item,
                {
                    "y",
                    "value",
                    "actual",
                    "forecast",
                    "consensus",
                },
            )

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x_value = item[0]
            y_value = item[1]

        timestamp = _parse_datetime(x_value)

        if pd.notna(timestamp):
            points.append(
                (
                    timestamp,
                    parse_million_barrels(y_value),
                )
            )

    return points


def _recognized_series(
    node: Mapping[str, object],
) -> dict[str, PointSeries]:
    compact_to_original = {_compact_key(key): key for key in node}

    recognized: dict[str, PointSeries] = {}

    for output_name, aliases in _SERIES_ALIASES.items():
        key = next(
            (
                compact_to_original[alias]
                for alias in aliases
                if alias in compact_to_original
            ),
            None,
        )

        if key is None:
            continue

        points = _point_series(node[key])

        if points:
            recognized[output_name] = points

    return recognized


def _merge_series(
    recognized: Mapping[str, PointSeries],
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for name, points in recognized.items():
        frame = pd.DataFrame(
            points,
            columns=[
                "release_datetime",
                name,
            ],
        )

        frame = frame.drop_duplicates(
            "release_datetime",
            keep="last",
        )

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame,
                on="release_datetime",
                how="outer",
            )

    if merged is None:
        return pd.DataFrame()

    return merged


def _series_frame_from_mapping(
    node: Mapping[str, object],
) -> pd.DataFrame:
    recognized = _recognized_series(node)

    if not recognized:
        return pd.DataFrame()

    return _merge_series(recognized)


def _series_like_frames(
    payload: JsonValue,
) -> list[pd.DataFrame]:
    """Handle chart JSON where values are stored in separate point arrays."""
    frames: list[pd.DataFrame] = []

    def walk(node: JsonValue) -> None:
        if isinstance(node, dict):
            frame = _series_frame_from_mapping(node)

            if not frame.empty:
                frames.append(frame)

            for value in node.values():
                walk(value)

            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return frames


def _normalize_row_candidates(
    rows: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for row in rows:
        release_datetime = _parse_datetime(
            _first_value(
                row,
                _DATE_KEYS,
            )
        )

        if pd.isna(release_datetime):
            continue

        event = _first_value(
            row,
            _EVENT_KEYS,
        )

        records.append(
            {
                "release_datetime": release_datetime,
                "event": event or EVENT_NAME,
                "actual_mmbbl": parse_million_barrels(
                    _first_value(
                        row,
                        _ACTUAL_KEYS,
                    )
                ),
                "consensus_mmbbl": parse_million_barrels(
                    _first_value(
                        row,
                        _CONSENSUS_KEYS,
                    )
                ),
                "previous_mmbbl": parse_million_barrels(
                    _first_value(
                        row,
                        _PREVIOUS_KEYS,
                    )
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def _previous_friday(
    timestamp: pd.Timestamp,
) -> pd.Timestamp:
    date = timestamp.tz_convert("UTC").normalize().tz_localize(None)

    days_since_friday = (date.weekday() - 4) % 7

    if days_since_friday == 0:
        days_since_friday = 7

    return date - pd.Timedelta(
        days=days_since_friday,
    )


def _prepare_combined_frame(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if "event" not in combined:
        combined["event"] = EVENT_NAME

    combined["event"] = combined["event"].fillna(EVENT_NAME)

    value_columns = [
        "actual_mmbbl",
        "consensus_mmbbl",
        "previous_mmbbl",
    ]

    for column in value_columns:
        if column not in combined:
            combined[column] = pd.NA

        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    return (
        combined.sort_values("release_datetime")
        .groupby(
            "release_datetime",
            as_index=False,
        )
        .agg(
            event=("event", "last"),
            actual_mmbbl=("actual_mmbbl", "last"),
            consensus_mmbbl=("consensus_mmbbl", "last"),
            previous_mmbbl=("previous_mmbbl", "last"),
        )
    )


def _add_derived_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["release_date"] = (
        result["release_datetime"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    result["eia_week_ending"] = result["release_datetime"].map(_previous_friday)

    result["inventory_surprise_mmbbl"] = (
        result["actual_mmbbl"] - result["consensus_mmbbl"]
    )

    result["source"] = SOURCE_NAME
    result["event_id"] = DEFAULT_EVENT_ID
    result["unit"] = "million barrels"

    return result


def _filter_release_dates(
    frame: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    result = frame

    if start is not None:
        result = result[result["release_date"] >= pd.Timestamp(start)]

    if end is not None:
        result = result[result["release_date"] <= pd.Timestamp(end)]

    return result


def normalize_myfxbook_payload(
    payload: JsonValue,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Normalize plausible Myfxbook chart JSON layouts to one table."""
    row_frame = _normalize_row_candidates(_row_like_dicts(payload))

    series_frames = _series_like_frames(payload)

    frames = [
        frame
        for frame in [
            row_frame,
            *series_frames,
        ]
        if not frame.empty
    ]

    if not frames:
        msg = (
            "Не удалось распознать структуру Myfxbook JSON. "
            "Сохрани raw payload и проверь ключи верхнего уровня."
        )
        raise ValueError(msg)

    combined = _prepare_combined_frame(frames)

    combined = _add_derived_columns(combined)

    combined = _filter_release_dates(
        combined,
        start=start,
        end=end,
    )

    columns = [
        "release_datetime",
        "release_date",
        "eia_week_ending",
        "event",
        "actual_mmbbl",
        "consensus_mmbbl",
        "previous_mmbbl",
        "inventory_surprise_mmbbl",
        "source",
        "event_id",
        "unit",
    ]

    return combined[columns].sort_values("release_datetime").reset_index(drop=True)
