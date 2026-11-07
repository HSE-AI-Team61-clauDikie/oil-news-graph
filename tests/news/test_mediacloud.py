from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from news.mediacloud import (
    download,
    downloaded_months,
    iter_stories,
    merge,
    month_windows,
    normalize_url,
    pack_months,
)
from news.settings import NewsSettings


def make_story(story_id: str, url: str, day: date = date(2026, 10, 5)) -> dict:
    """
    История в формате ответа `story_list`.

    Parameters:
        - story_id: id в Media Cloud;
        - url: адрес статьи;
        - day: дата публикации.

    Returns:
        Словарь с полями истории.
    """

    return {
        "id": story_id,
        "publish_date": day,
        "indexed_date": None,
        "media_name": "example.ru",
        "media_url": "example.ru",
        "title": f"Нефть {story_id}",
        "url": url,
        "language": "ru",
    }


class FakeSearchApi:
    """Отдаёт заранее заданные страницы и запоминает токены запросов."""

    def __init__(
        self,
        pages: list[list[dict]],
        fail_on_page: int | None = None,
    ) -> None:
        self.pages = pages
        self.fail_on_page = fail_on_page
        self.tokens: list[str | None] = []

    def story_list(self, *_, **kwargs) -> tuple:
        """
        Страница по номеру из токена.

        Returns:
            Истории страницы и токен следующей или None.
        """

        token: str | None = kwargs["pagination_token"]
        self.tokens.append(token)
        number = int(token or 0)
        if number == self.fail_on_page:
            msg = "обрыв посреди месяца"
            raise RuntimeError(msg)
        next_token = str(number + 1) if number + 1 < len(self.pages) else None
        return self.pages[number], next_token

    def story_count_over_time(self, _query: str, start: date, end: date, **_) -> list:
        """
        По одной истории на каждое первое число месяца периода.

        Returns:
            Точки `date`, `count`.
        """

        self.tokens.append("count")
        return [{"date": first, "count": 1} for first, _ in month_windows(start, end)]


@pytest.fixture
def settings(tmp_path: Path) -> NewsSettings:
    return NewsSettings(
        _env_file=None,
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 31),
        index_dir=tmp_path / "index",
        stories_path=tmp_path / "stories.parquet",
    )


def test_iter_stories_follows_pagination(settings: NewsSettings) -> None:
    api = FakeSearchApi(
        [
            [make_story("1", "https://a.ru/1")],
            [make_story("2", "https://a.ru/2")],
            [make_story("3", "https://a.ru/3")],
        ],
    )

    stories = list(
        iter_stories(api, settings, settings.start_date, settings.end_date),
    )

    assert [story["id"] for story in stories] == ["1", "2", "3"]
    assert api.tokens == [None, "1", "2"]


@pytest.mark.parametrize(
    ("start", "end", "count"),
    [
        (date(2015, 2, 19), date(2026, 10, 31), 141),
        (date(2024, 1, 15), date(2024, 1, 20), 1),
        (date(2023, 12, 31), date(2024, 3, 1), 4),
    ],
)
def test_month_windows_cover_period_without_gaps(
    start: date,
    end: date,
    count: int,
) -> None:
    windows = month_windows(start, end)

    assert len(windows) == count
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (_, previous_end), (next_start, _) in pairwise(windows):
        assert next_start - previous_end == timedelta(days=1)


def test_download_skips_finished_month(settings: NewsSettings) -> None:
    settings.index_dir.mkdir(parents=True)
    done = settings.index_dir / "stories_2026-10.parquet"
    pd.DataFrame({"story_id": ["old"]}).to_parquet(done)
    api = FakeSearchApi([[make_story("1", "https://a.ru/1")]])

    download(api, settings)

    assert api.tokens == []
    assert pd.read_parquet(done)["story_id"].tolist() == ["old"]


def test_download_packs_small_months_and_skips_covered(
    settings: NewsSettings,
) -> None:
    settings = settings.model_copy(
        update={"start_date": date(2026, 7, 1), "end_date": date(2026, 10, 31)},
    )
    settings.index_dir.mkdir(parents=True)
    pd.DataFrame({"story_id": ["old"]}).to_parquet(
        settings.index_dir / "stories_2026-08.parquet",
    )
    api = FakeSearchApi([[make_story("1", "https://a.ru/1")]])

    download(api, settings)

    # Июль отдельно: август скачан и разрывает окно; сентябрь и октябрь - вместе
    names = sorted(path.name for path in settings.index_dir.iterdir())
    assert names == [
        "stories_2026-07.parquet",
        "stories_2026-08.parquet",
        "stories_2026-09_2026-10.parquet",
    ]
    assert downloaded_months(settings.index_dir) == {
        "2026-07",
        "2026-08",
        "2026-09",
        "2026-10",
    }


def test_pack_months_splits_full_pages() -> None:
    months = month_windows(date(2026, 1, 1), date(2026, 4, 30))
    sizes = {"2026-01": 400, "2026-02": 500, "2026-03": 2000, "2026-04": 10}

    packed = pack_months(months, sizes, page_size=1000)

    assert packed == [
        (date(2026, 1, 1), date(2026, 2, 28)),
        (date(2026, 3, 1), date(2026, 3, 31)),
        (date(2026, 4, 1), date(2026, 4, 30)),
    ]


def test_download_leaves_no_file_when_month_breaks(settings: NewsSettings) -> None:
    api = FakeSearchApi(
        [[make_story("1", "https://a.ru/1")], [make_story("2", "https://a.ru/2")]],
        fail_on_page=1,
    )

    with pytest.raises(RuntimeError):
        download(api, settings)

    assert list(settings.index_dir.iterdir()) == []


def test_merge_drops_duplicate_ids_and_urls(settings: NewsSettings) -> None:
    api = FakeSearchApi(
        [
            [
                make_story("1", "https://www.a.ru/news/1/?utm_source=tg"),
                make_story("1", "https://www.a.ru/news/1/?utm_source=tg"),
                make_story("2", "http://a.ru/news/1"),
                make_story("3", "https://a.ru/news?id=2"),
                make_story("4", "https://a.ru/news?id=3"),
            ],
        ],
    )
    download(api, settings)

    stories = merge(settings)

    assert stories["story_id"].tolist() == ["1", "3", "4"]
    assert pd.read_parquet(settings.stories_path).shape[0] == 3


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://www.Lenta.ru/news/1/", "lenta.ru/news/1"),
        ("http://lenta.ru/news/1?utm_medium=rss&b=2&a=1", "lenta.ru/news/1?a=1&b=2"),
        ("https://kp.ru/online/news/3355209/", "kp.ru/online/news/3355209"),
    ],
)
def test_normalize_url(url: str, key: str) -> None:
    assert normalize_url(url) == key
