from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from news.fulltext import ArticleFetcher, assemble, read_articles, run
from news.settings import NewsSettings

TEXT = (
    "Цены на нефть марки Brent выросли после сообщений о сокращении добычи "
    "странами ОПЕК+. Аналитики ожидают снижения коммерческих запасов в США. "
) * 6


def article_html(text: str = TEXT, charset: str = "utf-8") -> bytes:
    """
    Страница новости с заголовком, меню и телом статьи.

    Parameters:
        - text: текст статьи;
        - charset: кодировка страницы.

    Returns:
        HTML в указанной кодировке.
    """

    paragraphs = "".join(f"<p>{part}.</p>" for part in text.split(". ") if part)
    return (
        f'<html><head><meta charset="{charset}"><title>Нефть дорожает</title>'
        "</head><body><nav><a href='/'>Главная</a><a href='/news'>Новости</a></nav>"
        f"<article><h1>Нефть дорожает</h1>{paragraphs}</article>"
        "<footer>Все права защищены</footer></body></html>"
    ).encode(charset)


@pytest.fixture
def settings(tmp_path: Path) -> NewsSettings:
    return NewsSettings(
        _env_file=None,
        wayback_interval=0,
        stories_path=tmp_path / "stories.parquet",
        fulltext_dir=tmp_path / "fulltext",
        dataset_path=tmp_path / "dataset.parquet",
    )


def make_fetcher(
    settings: NewsSettings,
    handler: Callable[[httpx.Request], httpx.Response],
) -> ArticleFetcher:
    """
    Загрузчик поверх подменённого транспорта.

    Parameters:
        - settings: настройки загрузки;
        - handler: ответы вместо сети.

    Returns:
        Загрузчик статей.
    """

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ArticleFetcher(client, settings)


def wayback(snapshot: bytes | None) -> Callable[[httpx.Request], httpx.Response]:
    """
    Сайт отдаёт 404, Wayback - снимок или пустой CDX.

    Parameters:
        - snapshot: HTML снимка; None - снимка нет.

    Returns:
        Обработчик запросов для MockTransport.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "web.archive.org":
            return httpx.Response(404)
        if request.url.path == "/cdx/search/cdx":
            assert request.url.params["closest"] == "20190115"
            if snapshot is None:
                return httpx.Response(200, json=[])
            return httpx.Response(
                200,
                json=[["urlkey", "timestamp"], ["ru,kp)/1", "20190115110624"]],
            )
        assert request.url.path == "/web/20190115110624id_/https://kp.ru/1"
        return httpx.Response(200, content=snapshot)

    return handler


async def test_live_page_gives_text(settings: NewsSettings) -> None:
    fetcher = make_fetcher(
        settings,
        lambda _: httpx.Response(200, content=article_html()),
    )

    article = await fetcher.fetch_article("1", "https://lenta.ru/1", None)

    assert article.text_source == "live"
    assert "ОПЕК+" in article.text
    assert "Главная" not in article.text
    assert article.extracted_title == "Нефть дорожает"


async def test_dead_page_falls_back_to_wayback(settings: NewsSettings) -> None:
    fetcher = make_fetcher(settings, wayback(article_html()))

    article = await fetcher.fetch_article("1", "https://kp.ru/1", date(2019, 1, 15))

    assert article.text_source == "wayback"
    assert article.wayback_timestamp == "20190115110624"
    assert "Brent" in article.text


async def test_no_text_anywhere_keeps_both_errors(settings: NewsSettings) -> None:
    fetcher = make_fetcher(settings, wayback(None))

    article = await fetcher.fetch_article("1", "https://kp.ru/1", date(2019, 1, 15))

    assert article.text_source == "none"
    assert article.text is None
    assert article.error == "HTTP 404; wayback: нет снимка"


async def test_short_text_is_not_an_article(settings: NewsSettings) -> None:
    fetcher = make_fetcher(
        settings,
        lambda _: httpx.Response(200, content=article_html("Нефть. Подробнее")),
    )

    article = await fetcher.fetch_live("1", "https://lenta.ru/1")

    assert article.text_source == "none"
    assert article.error.startswith("короткий текст")


async def test_cp1251_page_is_decoded(settings: NewsSettings) -> None:
    fetcher = make_fetcher(
        settings,
        lambda _: httpx.Response(
            200,
            content=article_html(charset="windows-1251"),
            headers={"Content-Type": "text/html"},
        ),
    )

    article = await fetcher.fetch_live("1", "https://old.ru/1")

    assert article.text_source == "live"
    assert "коммерческих запасов" in article.text


async def test_run_skips_fetched_and_retries_failed(
    settings: NewsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["ok", "failed", "new"],
            "url": ["https://a.ru/ok", "https://a.ru/failed", "https://a.ru/new"],
            "publish_date": pd.to_datetime(["2026-10-01", "2026-10-02", None]),
        },
    ).to_parquet(settings.stories_path)
    settings.fulltext_dir.mkdir()
    pd.DataFrame(
        {
            "story_id": ["ok", "failed"],
            "text_source": ["live", "none"],
            "fetched_at": pd.to_datetime(["2026-11-01", "2026-11-01"], utc=True),
        },
    ).to_parquet(settings.fulltext_dir / "part-00000.parquet")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, content=article_html())

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=transport, **kwargs),
    )

    await run(settings)
    assert requested == ["/new"]

    await run(settings, retry_failed=True)
    assert requested == ["/new", "/failed"]

    articles = read_articles(settings).set_index("story_id")["text_source"]
    assert articles.to_dict() == {"ok": "live", "new": "live", "failed": "live"}


def test_assemble_marks_reprints(settings: NewsSettings) -> None:
    pd.DataFrame(
        {
            "story_id": ["tass", "region", "other", "empty"],
            "url": ["https://tass.ru/1", "https://r.ru/1", "https://o.ru/1", "x"],
            "publish_date": pd.to_datetime(
                ["2026-10-01", "2026-10-02", "2026-10-02", "2026-10-02"],
            ),
            "title": ["Нефть"] * 4,
        },
    ).to_parquet(settings.stories_path)
    settings.fulltext_dir.mkdir()
    pd.DataFrame(
        {
            "story_id": ["region", "tass", "other", "empty"],
            "url": ["https://r.ru/1", "https://tass.ru/1", "https://o.ru/1", "x"],
            "text": [TEXT.upper(), TEXT, "Другой текст про НПЗ", None],
            "text_source": ["live", "wayback", "live", "none"],
            "fetched_at": pd.to_datetime(["2026-11-01"] * 4, utc=True),
        },
    ).to_parquet(settings.fulltext_dir / "part-00000.parquet")

    dataset = assemble(settings).set_index("story_id")

    assert sorted(dataset.index) == ["other", "region", "tass"]
    assert dataset.loc["region", "duplicate_of"] == "tass"
    assert pd.isna(dataset.loc["tass", "duplicate_of"])
    assert pd.isna(dataset.loc["other", "duplicate_of"])
    assert dataset.loc["tass", "url"] == "https://tass.ru/1"
