"""Boundary checks for the news EDA's calendar and missing-data accounting."""

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture
def audit(monkeypatch, tmp_path) -> ModuleType:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "eda"))
    import news_eda

    return news_eda


def test_packed_windows_include_every_month(audit) -> None:
    assert audit.covered_months(Path("stories_2020-11_2021-02.parquet")) == {
        "2020-11",
        "2020-12",
        "2021-01",
        "2021-02",
    }
    assert audit.covered_months(Path("stories_2026-10.parquet")) == {"2026-10"}


def test_release_cutoff_excludes_same_day_even_with_intraday_time(audit) -> None:
    releases = pd.DatetimeIndex(["2026-01-07", "2026-01-15", "2026-01-21"])
    articles = pd.Series(
        pd.to_datetime(
            [
                "2026-01-06 23:59",
                "2026-01-07 00:01",
                "2026-01-15 00:00",
            ],
            format="mixed",
        ),
        index=[11, 15, 99],
    )
    got = audit.assign_release(articles, releases)
    expected = pd.Series(
        pd.to_datetime(
            [
                "2026-01-07",
                "2026-01-15",
                "2026-01-21",
            ]
        ),
        index=articles.index,
        dtype="datetime64[ns]",
    )
    pd.testing.assert_series_equal(got, expected)


def test_missing_dates_and_dates_after_last_release_stay_missing(audit) -> None:
    articles = pd.Series(pd.to_datetime([None, "2026-01-07", "2026-01-08"]))
    got = audit.assign_release(articles, pd.DatetimeIndex(["2026-01-07"]))
    assert got.isna().all()


def test_empty_calendar_cannot_wrap_to_future_data(audit) -> None:
    articles = pd.Series(pd.to_datetime(["2026-01-07"]))
    got = audit.assign_release(articles, pd.DatetimeIndex([]))
    assert got.isna().all()


def test_source_concentration_uses_host_shares(audit) -> None:
    got = audit.concentration(pd.Series([90, 10]))
    assert got["top1_share"] == pytest.approx(0.9)
    assert got["effective_sources"] == pytest.approx(1 / (0.9**2 + 0.1**2))
