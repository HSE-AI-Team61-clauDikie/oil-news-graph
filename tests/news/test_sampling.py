import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news.regions import Region
from news.sampling import collect, gold, merge, pack_days, pick_days
from news.settings import NewsSettings

CANDIDATES_PER_DAY = 3
BACKGROUND_PER_DAY = 50


def make_story(story_id: str, day: date) -> dict:
    """
    История в формате ответа `story_list`.

    Parameters:
        - story_id: id в Media Cloud;
        - day: дата публикации.

    Returns:
        Словарь с полями истории.
    """

    return {
        "id": story_id,
        "publish_date": day,
        "indexed_date": None,
        "media_name": "example.com",
        "media_url": "example.com",
        "title": f"Oil {story_id}",
        "url": f"https://example.com/{story_id}",
        "language": "en",
    }


class FakeSearchApi:
    """Каждый день - 3 кандидата и 50 историй фона."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def story_count_over_time(self, query: str, start: date, end: date, **_) -> list:
        """
        Объём запроса по дням.

        Returns:
            Точки `date`, `count`.
        """

        self.calls.append("count")
        per_day = (
            CANDIDATES_PER_DAY
            if "article_title" in query
            else CANDIDATES_PER_DAY + BACKGROUND_PER_DAY
        )
        days = (end - start).days + 1
        return [
            {"date": start + timedelta(days=offset), "count": per_day}
            for offset in range(days)
        ]

    def story_list(self, query: str, *_, **kwargs) -> tuple:
        """
        Истории за дни из фильтра `publication_date`.

        Returns:
            Не больше страницы историй и пустой токен.
        """

        self.calls.append("list")
        background = "NOT" in query
        per_day = BACKGROUND_PER_DAY if background else CANDIDATES_PER_DAY
        kind = "bg" if background else "cand"
        days = re.findall(r"\[(\d{4}-\d{2}-\d{2}) TO", query)
        stories = [
            make_story(f"{kind}-{day}-{i}", date.fromisoformat(day))
            for day in days
            for i in range(per_day)
        ]
        return stories[: kwargs["page_size"]], None


@pytest.fixture
def settings(tmp_path: Path) -> NewsSettings:
    """Выборка 2024-2025: 2 кандидата в год, 1 история фона в год с 2 дней."""

    return NewsSettings(
        _env_file=None,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        sample_dir=tmp_path / "sample",
        sample_candidates=2,
        sample_background=1,
        sample_background_days=2,
    )


@pytest.fixture
def region() -> Region:
    """Регион США."""

    return Region(
        name="us",
        countries=["US"],
        collection_ids=[1],
        languages=["en"],
        terms=["oil"],
    )


def test_pick_days_until_enough_volume() -> None:
    volumes = pd.Series(
        [0, 5, 0, 10],
        index=pd.date_range("2025-01-01", periods=4),
    )
    rng = np.random.default_rng(0)

    assert len(pick_days(volumes, 12, rng, max_days=10)) == 2
    assert len(pick_days(volumes, 3, rng, max_days=10)) == 1
    assert len(pick_days(volumes, 100, rng, max_days=1)) == 1
    assert len(pick_days(volumes, 3, rng, max_days=10, min_days=2)) == 2
    assert pick_days(volumes * 0, 3, rng, max_days=10) == []


def test_pick_days_prefers_bigger_days() -> None:
    volumes = pd.Series([1, 9], index=pd.date_range("2025-01-01", periods=2))
    rng = np.random.default_rng(0)

    first = [pick_days(volumes, 1, rng, max_days=1)[0] for _ in range(2000)]

    share = np.mean([day == volumes.index[1] for day in first])
    assert share == pytest.approx(0.9, abs=0.03)


def test_pack_days_fits_page() -> None:
    days = list(pd.date_range("2025-01-01", periods=5))
    volumes = pd.Series([400, 700, 300, 1500, 100], index=days)

    groups = pack_days(days, volumes, capacity=1000, max_days=10)

    assert groups == [[days[0]], [days[1], days[2]], [days[3]], [days[4]]]
    assert pack_days(days, volumes * 0, capacity=1000, max_days=2) == [
        days[:2],
        days[2:4],
        days[4:],
    ]


def test_collect_samples_per_year_and_skips_done(
    settings: NewsSettings,
    region: Region,
) -> None:
    api = FakeSearchApi()

    collect(api, [region], settings)
    first_run = len(api.calls)
    collect(api, [region], settings)

    assert len(api.calls) == first_run
    part = pd.read_parquet(settings.sample_dir / "parts" / "us.parquet")
    candidates = part[part["stratum"] == "candidate"]
    background = part[part["stratum"] == "background"]
    assert candidates.groupby("year").size().to_dict() == {2024: 2, 2025: 2}
    assert len(background) == 2
    # Кандидатов 3 в день: 366 и 365 дней на 2 взятых
    assert sorted(candidates["weight"].unique()) == [365 * 3 / 2, 366 * 3 / 2]
    assert background["weight"].iloc[0] == 731 * BACKGROUND_PER_DAY / 2


def test_merge_drops_duplicates(settings: NewsSettings, region: Region) -> None:
    collect(FakeSearchApi(), [region], settings)
    part = settings.sample_dir / "parts" / "us.parquet"
    frame = pd.read_parquet(part)
    pd.concat([frame, frame]).to_parquet(settings.sample_dir / "parts" / "dup.parquet")

    sample = merge(settings)

    assert len(sample) == len(frame)
    assert (settings.sample_dir / "stories.parquet").exists()


def test_merge_needs_sample(settings: NewsSettings) -> None:
    with pytest.raises(FileNotFoundError):
        merge(settings)


def test_gold_takes_equal_share_per_language() -> None:
    sample = pd.DataFrame(
        {
            "story_id": [str(i) for i in range(30)],
            "language": ["en"] * 20 + ["ar"] * 8 + [None] * 2,
            "region": "us",
            "year": 2025,
        },
    )

    picked = gold(sample, size=15)

    assert picked["language"].fillna("unknown").value_counts().to_dict() == {
        "en": 5,
        "ar": 5,
        "unknown": 2,
    }
    assert picked["story_id"].is_unique
