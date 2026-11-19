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


def test_correlation_excludes_truncated_week_and_counts_identical_screens_once(
    audit,
) -> None:
    dates = pd.date_range("2026-01-07", "2026-11-04", freq="7D")
    n = len(dates)
    weekly = pd.DataFrame(
        {
            "region": "us",
            "release_date": dates,
            "window_days": 7,
            "attempted": 10,
            "attempt_fraction": 1.0,
            "assembled": range(n),
            "surprise_mmbbl": range(n),
        }
    )
    result = audit.count_correlations(weekly)
    assert result[["screen", "n"]].values.tolist() == [["both", n - 1]]
    weekly.loc[0, "attempt_fraction"] = 0.5
    result = audit.count_correlations(weekly)
    assert dict(zip(result.screen, result.n, strict=True)) == {
        "any_attempt": n - 1,
        "80pct_attempted": n - 2,
    }


def test_api_comparison_fits_only_earlier_releases_on_common_windows(audit) -> None:
    dates = pd.date_range("2022-11-02", periods=15, freq="7D")
    weekly = pd.DataFrame(
        {
            "release_date": dates,
            "window_days": 7,
            "attempted": 10,
            "candidates": 10,
            "unique_screened": [3, 7, 2] * 5,
            "physical_candidates": [1, 0, 1] * 5,
            "surprise_mmbbl": [
                2 * i + 1 if d.year == 2022 else 2 * i + 11 for i, d in enumerate(dates)
            ],
            "api_mmbbl": range(15),
            "consensus_mmbbl": 0,
        }
    )
    weekly.loc[0, "window_days"] = 8
    weekly.loc[13, "attempted"] = 5
    result = audit.volume_benchmark(weekly)
    assert result.train_weeks.tolist() == [8, 8]
    assert result.test_weeks.tolist() == [5, 5]
    assert result.mae_mmbbl.tolist() == pytest.approx([10, 10])
