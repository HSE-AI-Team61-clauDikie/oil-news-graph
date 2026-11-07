from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

MODEL_START_DATE = pd.Timestamp("2015-01-01")

# Trendonify contains two obvious date-label anomalies in the chart series.
# The values fit the surrounding weekly sequence, but the dates are shown
# on a weekend instead of the corresponding week-ending Friday.
KNOWN_REFERENCE_DATE_CORRECTIONS: dict[str, str] = {
    "2018-12-30": "2018-12-28",
    "2020-02-29": "2020-02-28",
}


class _AstroChartPropsParser(HTMLParser):
    """Find the Astro island whose props contain chartData."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chart_props: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "astro-island":
            return

        attributes = dict(attrs)
        props = attributes.get("props")

        if props is None or '"chartData"' not in props:
            return

        self.chart_props = props


@dataclass(frozen=True)
class TrendonifyFetchResult:
    """Parsed Trendonify API crude history."""

    frame: pd.DataFrame
    fetched_at_utc: datetime
    raw_html_path: Path


def _unwrap_astro(value: JsonValue) -> JsonValue:
    """Recursively unwrap Astro's serialized [tag, value] representation."""

    if isinstance(value, dict):
        return {key: _unwrap_astro(item) for key, item in value.items()}

    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], int):
            return _unwrap_astro(value[1])

        return [_unwrap_astro(item) for item in value]

    return value


def _apply_known_date_corrections(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Correct documented Trendonify weekend date-label anomalies."""

    result = frame.copy()

    result["original_reference_week_end"] = result["reference_week_end"]

    correction_map = {
        pd.Timestamp(source): pd.Timestamp(target)
        for source, target in KNOWN_REFERENCE_DATE_CORRECTIONS.items()
    }

    result["reference_week_end"] = result["reference_week_end"].replace(correction_map)

    result["reference_date_adjusted"] = (
        result["reference_week_end"] != result["original_reference_week_end"]
    )

    result["reference_date_adjustment_days"] = (
        result["reference_week_end"] - result["original_reference_week_end"]
    ).dt.days

    return result


def extract_chart_data_from_html(
    html: str,
) -> pd.DataFrame:
    """Extract the full API crude Date/Value history from saved HTML."""

    parser = _AstroChartPropsParser()
    parser.feed(html)

    if parser.chart_props is None:
        msg = "Could not find an astro-island containing chartData in Trendonify HTML."
        raise ValueError(msg)

    try:
        props: JsonValue = json.loads(parser.chart_props)
    except json.JSONDecodeError as exc:
        msg = (
            "Trendonify chartData props are not valid JSON after HTML entity decoding."
        )
        raise ValueError(msg) from exc

    if not isinstance(props, dict):
        msg = "Trendonify Astro props did not decode to an object."
        raise TypeError(msg)

    if "chartData" not in props:
        msg = "Trendonify Astro props do not contain chartData."
        raise KeyError(msg)

    chart_data = _unwrap_astro(props["chartData"])

    if not isinstance(chart_data, list):
        msg = "Trendonify chartData did not decode to a list."
        raise TypeError(msg)

    rows: list[dict[str, JsonScalar]] = []

    for item in chart_data:
        if not isinstance(item, dict):
            continue

        date = item.get("Date")
        value = item.get("Value")
        country = item.get("Country")

        if not isinstance(date, str) or not isinstance(value, int | float):
            continue

        rows.append(
            {
                "reference_week_end": date,
                "api_actual_million": value,
                "country": (country if isinstance(country, str) else None),
            }
        )

    if not rows:
        msg = "No Date/Value rows found in Trendonify chartData."
        raise ValueError(msg)

    frame = pd.DataFrame(rows)

    frame["reference_week_end"] = pd.to_datetime(
        frame["reference_week_end"],
        errors="raise",
    )

    frame["api_actual_million"] = pd.to_numeric(
        frame["api_actual_million"],
        errors="raise",
    )

    frame = _apply_known_date_corrections(frame)

    frame["country"] = frame["country"].astype("string").str.lower()

    frame["source"] = "trendonify"
    frame["series"] = "api_crude_oil_stock_change"
    frame["unit"] = "million_barrels"

    frame = (
        frame.sort_values("reference_week_end")
        .drop_duplicates(
            subset=["reference_week_end"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    _validate_chart_history(frame)

    return frame


def _validate_chart_history(
    frame: pd.DataFrame,
) -> None:
    """Validate source integrity and model-period weekly cadence."""

    if frame["reference_week_end"].duplicated().any():
        msg = "Duplicate Trendonify reference weeks found after date correction."
        raise ValueError(msg)

    if frame["api_actual_million"].isna().any():
        msg = "Missing API crude values found in Trendonify history."
        raise ValueError(msg)

    weekdays = set(frame["reference_week_end"].dt.dayofweek.unique())

    if weekdays != {4}:
        unexpected = frame.loc[
            frame["reference_week_end"].dt.dayofweek.ne(4),
            [
                "original_reference_week_end",
                "reference_week_end",
                "api_actual_million",
            ],
        ]

        msg = (
            "Unexpected non-Friday reference dates remain "
            "after applying known corrections:\n"
            f"{unexpected.to_string(index=False)}"
        )
        raise ValueError(msg)

    model_frame = frame.loc[frame["reference_week_end"].ge(MODEL_START_DATE)].copy()

    gaps = model_frame["reference_week_end"].sort_values().diff().dt.days.dropna()

    bad_gaps = gaps.loc[gaps.ne(7)]

    if not bad_gaps.empty:
        msg = (
            "Trendonify has missing or irregular weeks "
            "inside the 2015+ model period. "
            f"Observed irregular gaps: "
            f"{sorted(bad_gaps.unique().tolist())}"
        )
        raise ValueError(msg)


def load_trendonify_api_crude_html(
    raw_html_path: Path,
) -> TrendonifyFetchResult:
    """Load a manually saved Trendonify HTML snapshot."""

    if not raw_html_path.exists():
        msg = f"Trendonify HTML snapshot not found: {raw_html_path}"
        raise FileNotFoundError(msg)

    html = raw_html_path.read_text(encoding="utf-8")

    fetched_at_utc = datetime.fromtimestamp(
        raw_html_path.stat().st_mtime,
        tz=UTC,
    )

    frame = extract_chart_data_from_html(html)

    frame["fetched_at_utc"] = fetched_at_utc

    return TrendonifyFetchResult(
        frame=frame,
        fetched_at_utc=fetched_at_utc,
        raw_html_path=raw_html_path,
    )


def fetch_trendonify_api_crude(
    *,
    raw_html_path: Path | None = None,
    force: bool = False,
) -> TrendonifyFetchResult:
    """Compatibility wrapper for the notebook.

    Trendonify blocks automated HTTP requests with HTTP 403,
    so this project intentionally parses a manually saved HTML
    snapshot instead of attempting to bypass the site's protection.
    """

    if force:
        msg = (
            "Automated Trendonify download is disabled. "
            "Save the page HTML manually and use force=False."
        )
        raise RuntimeError(msg)

    if raw_html_path is None:
        msg = (
            "raw_html_path is required because Trendonify "
            "is parsed from a saved HTML snapshot."
        )
        raise ValueError(msg)

    return load_trendonify_api_crude_html(raw_html_path)


def select_model_period(
    frame: pd.DataFrame,
    *,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Select the project modelling period without changing raw history."""

    start = pd.Timestamp(start_date)

    mask = frame["reference_week_end"].ge(start)

    if end_date is not None:
        end = pd.Timestamp(end_date)

        mask &= frame["reference_week_end"].le(end)

    return frame.loc[mask].copy().reset_index(drop=True)


def select_trendonify_pre_myfxbook(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Select Trendonify observations for the 2015-2016 gap."""

    return (
        frame.loc[
            frame["reference_week_end"].between(
                pd.Timestamp("2015-01-01"),
                pd.Timestamp("2016-12-31"),
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )
