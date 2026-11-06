from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

EXPECTED_NEARBYS = (1, 2, 3, 4, 5, 6, 12)

_FILENAME_PATTERN = re.compile(
    r"^cl_f(?P<nearby>1|2|3|4|5|6|12)_daily_nearby\.csv$",
    flags=re.IGNORECASE,
)
_EXPRESSION_PATTERN = re.compile(
    r"Expression:\s*\{CL\*(?P<nearby>\d+)\}",
    flags=re.IGNORECASE,
)
_DOWNLOAD_PATTERN = re.compile(
    r"Downloaded from Barchart\.com as of "
    r"(?P<date>\d{2}-\d{2}-\d{4}) "
    r"(?P<time>\d{2}:\d{2}(?:am|pm)) "
    r"(?P<timezone>[A-Z]{2,4})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedNearbyFile:
    """One manually downloaded Barchart Premier nearby series."""

    nearby: int
    expression: str
    downloaded_at: pd.Timestamp
    download_timezone: str
    source_path: Path
    frame: pd.DataFrame


def _read_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue

    msg = f"Could not decode Barchart CSV: {path.name}"
    raise UnicodeError(msg)


def _nearby_from_filename(path: Path) -> int:
    match = _FILENAME_PATTERN.fullmatch(path.name)

    if match is None:
        msg = (
            "Unexpected Barchart filename "
            f"{path.name!r}. Expected names like "
            "'cl_f1_daily_nearby.csv' and "
            "'cl_f12_daily_nearby.csv'."
        )
        raise ValueError(msg)

    return int(match.group("nearby"))


def _parse_header_expression(
    first_line: str,
) -> int:
    match = _EXPRESSION_PATTERN.search(first_line)

    if match is None:
        msg = (
            "Could not find a Barchart nearby expression "
            "such as 'Expression: {CL*1}' in the first row."
        )
        raise ValueError(msg)

    return int(match.group("nearby"))


def _parse_download_footer(
    last_line: str,
) -> tuple[pd.Timestamp, str]:
    match = _DOWNLOAD_PATTERN.search(last_line)

    if match is None:
        msg = f"Could not parse Barchart download footer. Last row: {last_line!r}"
        raise ValueError(msg)

    downloaded_at = pd.to_datetime(
        (f"{match.group('date')} {match.group('time')}"),
        format="%m-%d-%Y %I:%M%p",
        errors="raise",
    )

    return downloaded_at, match.group("timezone").upper()


def load_barchart_premier_csv(
    path: Path,
) -> ParsedNearbyFile:
    """Parse one Interactive Chart Daily Nearby CSV."""

    nearby = _nearby_from_filename(path)
    lines = _read_text_lines(path)

    if len(lines) < 4:
        msg = f"Barchart CSV is unexpectedly short: {path.name}"
        raise ValueError(msg)

    expression_nearby = _parse_header_expression(lines[0])

    if expression_nearby != nearby:
        msg = (
            f"Filename says F{nearby}, but the CSV "
            f"contains Expression {{CL*{expression_nearby}}}."
        )
        raise ValueError(msg)

    downloaded_at, download_timezone = _parse_download_footer(lines[-1])

    frame = pd.read_csv(
        path,
        skiprows=1,
    ).rename(
        columns={
            "Date Time": "trading_day",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Change": "change",
            "Volume": "chart_volume_like_main",
            "Open Interest": "open_interest",
        }
    )

    required_columns = {
        "trading_day",
        "open",
        "high",
        "low",
        "close",
    }
    missing = required_columns - set(frame.columns)

    if missing:
        msg = f"Unsupported Barchart schema in {path.name}: missing {sorted(missing)}."
        raise KeyError(msg)

    frame["trading_day"] = pd.to_datetime(
        frame["trading_day"],
        errors="coerce",
    )

    # The final human-readable download footer is not a market row.
    frame = frame.loc[frame["trading_day"].notna()].copy()

    # Barchart may already show the next Globex trading date while
    # the download footer is still stamped with the prior calendar day.
    # That next-day row is an unfinished session and must not enter
    # historical modelling.
    completed_through = downloaded_at.normalize()
    frame = frame.loc[frame["trading_day"] <= completed_through].copy()

    for column in (
        "open",
        "high",
        "low",
        "close",
        "change",
        "chart_volume_like_main",
        "open_interest",
    ):
        if column not in frame.columns:
            continue

        frame[column] = pd.to_numeric(
            frame[column].astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )

    frame["nearby"] = nearby
    frame["expression"] = f"{{CL*{nearby}}}"
    frame["source"] = "barchart_premier_interactive_chart"
    frame["raw_filename"] = path.name
    frame["downloaded_at"] = downloaded_at
    frame["download_timezone"] = download_timezone

    frame = (
        frame.sort_values("trading_day")
        .drop_duplicates(
            subset=["trading_day"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return ParsedNearbyFile(
        nearby=nearby,
        expression=f"{{CL*{nearby}}}",
        downloaded_at=downloaded_at,
        download_timezone=download_timezone,
        source_path=path,
        frame=frame,
    )


def discover_barchart_csvs(
    source_dir: Path,
) -> dict[int, Path]:
    """Find exactly one CSV for every required nearby maturity."""

    discovered: dict[int, Path] = {}

    for path in sorted(source_dir.glob("*.csv")):
        match = _FILENAME_PATTERN.fullmatch(path.name)

        if match is None:
            continue

        nearby = int(match.group("nearby"))

        if nearby in discovered:
            msg = (
                f"Multiple CSV files found for F{nearby}: "
                f"{discovered[nearby].name}, {path.name}"
            )
            raise ValueError(msg)

        discovered[nearby] = path

    missing = set(EXPECTED_NEARBYS) - set(discovered)

    if missing:
        missing_labels = ", ".join(f"F{nearby}" for nearby in sorted(missing))
        msg = f"Missing required Barchart nearby CSV files: {missing_labels}"
        raise FileNotFoundError(msg)

    return discovered


def _validate_distinct_price_series(
    frames: dict[int, pd.DataFrame],
    identical_threshold: float = 0.995,
) -> None:
    """Detect accidentally duplicated nearby downloads."""

    for left_nearby, right_nearby in combinations(
        EXPECTED_NEARBYS,
        2,
    ):
        left = frames[left_nearby][["trading_day", "close"]].rename(
            columns={"close": "left_close"}
        )
        right = frames[right_nearby][["trading_day", "close"]].rename(
            columns={"close": "right_close"}
        )

        overlap = left.merge(
            right,
            on="trading_day",
            how="inner",
        )

        if overlap.empty:
            msg = f"No date overlap between F{left_nearby} and F{right_nearby}."
            raise ValueError(msg)

        same_close_share = overlap["left_close"].eq(overlap["right_close"]).mean()

        if same_close_share >= identical_threshold:
            msg = (
                f"F{left_nearby} and F{right_nearby} "
                "are effectively identical "
                f"(same close on {same_close_share:.2%} "
                "of overlapping dates). "
                "Check Barchart download settings."
            )
            raise ValueError(msg)


def _validate_model_date_coverage(
    frames: dict[int, pd.DataFrame],
    start_date: str,
) -> None:
    """Require identical trading-date coverage over the model period."""

    start = pd.Timestamp(start_date)

    reference_dates = set(
        frames[EXPECTED_NEARBYS[0]].loc[
            lambda frame: frame["trading_day"] >= start,
            "trading_day",
        ]
    )

    for nearby in EXPECTED_NEARBYS[1:]:
        dates = set(
            frames[nearby].loc[
                lambda frame: frame["trading_day"] >= start,
                "trading_day",
            ]
        )

        if dates == reference_dates:
            continue

        missing = sorted(reference_dates - dates)
        extra = sorted(dates - reference_dates)

        msg = (
            f"Trading-date mismatch for F{nearby}. "
            f"Missing={len(missing)}, extra={len(extra)}."
        )
        raise ValueError(msg)


def load_barchart_wti_curve(
    source_dir: Path,
    start_date: str = "2015-01-01",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build long data, daily WTI curve, and source metadata."""

    files = discover_barchart_csvs(source_dir)

    parsed = {nearby: load_barchart_premier_csv(path) for nearby, path in files.items()}
    frames = {nearby: item.frame for nearby, item in parsed.items()}

    _validate_distinct_price_series(frames)
    _validate_model_date_coverage(
        frames,
        start_date=start_date,
    )

    start = pd.Timestamp(start_date)

    long_frame = pd.concat(
        [
            frames[nearby].loc[lambda frame: frame["trading_day"] >= start]
            for nearby in EXPECTED_NEARBYS
        ],
        ignore_index=True,
    )

    long_frame = long_frame.sort_values(["trading_day", "nearby"]).reset_index(
        drop=True
    )

    if long_frame.duplicated(["trading_day", "nearby"]).any():
        msg = "Duplicate trading_day/nearby rows after combining Barchart files."
        raise ValueError(msg)

    curve = (
        long_frame.pivot(
            index="trading_day",
            columns="nearby",
            values="close",
        )
        .rename(columns=lambda nearby: f"wti_f{nearby}")
        .reset_index()
        .sort_values("trading_day")
        .reset_index(drop=True)
    )

    price_columns = [f"wti_f{nearby}" for nearby in EXPECTED_NEARBYS]

    missing_curve_columns = set(price_columns) - set(curve.columns)

    if missing_curve_columns:
        msg = f"Missing curve columns after pivot: {sorted(missing_curve_columns)}"
        raise KeyError(msg)

    spread_pairs = (
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (1, 3),
        (1, 6),
        (1, 12),
    )

    for near_nearby, far_nearby in spread_pairs:
        column = f"spread_f{near_nearby}_f{far_nearby}"
        curve[column] = curve[f"wti_f{near_nearby}"] - curve[f"wti_f{far_nearby}"]
        curve[f"delta_{column}"] = curve[column].diff()

    curve["source"] = "barchart_premier_interactive_chart"
    curve["roll_method"] = "barchart_daily_nearby_liquidity_roll"
    curve["price_field"] = "close"

    metadata = pd.DataFrame(
        [
            {
                "nearby": nearby,
                "expression": item.expression,
                "raw_filename": item.source_path.name,
                "downloaded_at": item.downloaded_at,
                "download_timezone": (item.download_timezone),
                "raw_first_day": (item.frame["trading_day"].min()),
                "raw_last_day": (item.frame["trading_day"].max()),
                "raw_rows": len(item.frame),
                "model_rows": (item.frame["trading_day"].ge(start).sum()),
                "open_interest_available": (
                    item.frame["open_interest"].notna().any()
                    if "open_interest" in item.frame.columns
                    else False
                ),
            }
            for nearby, item in sorted(parsed.items())
        ]
    )

    return long_frame, curve, metadata
