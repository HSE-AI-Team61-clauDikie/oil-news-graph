from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from news.fulltext import (
    ArticleFetcher,
    junk_mask,
    needs_wayback,
    read_articles,
    run,
)
from news.settings import NewsSettings

TEXT = (
    "Цены на нефть марки Brent выросли после сообщений о сокращении добычи "
    "странами ОПЕК+. Аналитики ожидают снижения коммерческих запасов в США. "
) * 6

Handler = Callable[[httpx.Request], httpx.Response]


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


def make_fetcher(settings: NewsSettings, handler: Handler) -> ArticleFetcher:
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


def patch_client(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    """
    Подменить сеть для `run`, который сам создаёт клиент.

    Parameters:
        - monkeypatch: фикстура pytest;
        - handler: ответы вместо сети.
    """

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=transport, **kwargs),
    )


def wayback(snapshot: bytes | None) -> Handler:
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


def articles_frame(rows: list[dict]) -> pd.DataFrame:
    """
    Таблица статей в формате `read_articles` из коротких описаний.

    Parameters:
        - rows: поля статей; недостающие заполняются по умолчанию.

    Returns:
        Таблица статей.
    """

    defaults = {
        "final_url": None,
        "text": None,
        "text_source": "none",
        "error": None,
        "fetched_at": pd.Timestamp("2026-11-01", tz="UTC"),
    }
    return pd.DataFrame([defaults | row for row in rows])


async def test_live_page_gives_text(settings: NewsSettings) -> None:
    fetcher = make_fetcher(
        settings,
        lambda _: httpx.Response(200, content=article_html()),
    )

    article = await fetcher.fetch_live("1", "https://lenta.ru/1")

    assert article.text_source == "live"
    assert "ОПЕК+" in article.text
    assert "Главная" not in article.text
    assert article.extracted_title == "Нефть дорожает"


async def test_archived_page_comes_from_wayback(settings: NewsSettings) -> None:
    fetcher = make_fetcher(settings, wayback(article_html()))

    article = await fetcher.fetch_archived(
        "1",
        "https://kp.ru/1",
        date(2019, 1, 15),
        "HTTP 404",
    )

    assert article.text_source == "wayback"
    assert article.wayback_timestamp == "20190115110624"
    assert "Brent" in article.text


async def test_no_snapshot_keeps_both_errors(settings: NewsSettings) -> None:
    fetcher = make_fetcher(settings, wayback(None))

    article = await fetcher.fetch_archived(
        "1",
        "https://kp.ru/1",
        date(2019, 1, 15),
        "HTTP 404",
    )

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


async def test_run_goes_to_wayback_after_live(
    settings: NewsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["dead"],
            "url": ["https://kp.ru/1"],
            "publish_date": pd.to_datetime(["2019-01-15"]),
        },
    ).to_parquet(settings.stories_path)
    patch_client(monkeypatch, wayback(article_html()))

    await run(settings)

    article = read_articles(settings).iloc[0]
    assert article["text_source"] == "wayback"
    assert len(list(settings.fulltext_dir.glob("part-*.parquet"))) == 2


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
    articles_frame(
        [
            {"story_id": "ok", "url": "https://a.ru/ok", "text_source": "live"},
            {
                "story_id": "failed",
                "url": "https://a.ru/failed",
                "error": "HTTP 404; wayback: нет снимка",
            },
        ],
    ).to_parquet(settings.fulltext_dir / "part-00000.parquet")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, content=article_html())

    patch_client(monkeypatch, handler)

    await run(settings)
    assert requested == ["/new"]

    await run(settings, retry_failed=True)
    assert requested == ["/new", "/failed"]

    articles = read_articles(settings).set_index("story_id")["text_source"]
    assert articles.to_dict() == {"ok": "live", "new": "live", "failed": "live"}


async def test_run_newest_first_goes_from_fresh_to_old(
    settings: NewsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["2015", "2020", "2026"],
            "url": ["https://a.ru/2015", "https://a.ru/2020", "https://a.ru/2026"],
            "publish_date": pd.to_datetime(["2015-03-01", "2020-03-01", "2026-03-01"]),
        },
    ).to_parquet(settings.stories_path)
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        # Свой текст на каждую страницу: три одинаковых с сайта - уже шаблон
        page = article_html(f"{TEXT}Год {request.url.path}")
        return httpx.Response(200, content=page)

    patch_client(monkeypatch, handler)

    await run(settings.model_copy(update={"batch_size": 1}), newest_first=True)

    assert requested == ["/2026", "/2020", "/2015"]


def test_junk_mask_finds_templates_and_homepage_redirects() -> None:
    footer = "© 1992-2026 АО ИА «Башинформ». Все права защищены."
    articles = articles_frame(
        [
            *(
                {
                    "story_id": f"footer-{number}",
                    "url": f"https://www.bashinform.ru/news/{number}/",
                    "text": footer,
                    "text_source": "live",
                }
                for number in range(3)
            ),
            {
                "story_id": "reprint-1",
                "url": "https://interfax.ru/russia/1",
                "text": TEXT,
                "text_source": "live",
            },
            {
                "story_id": "reprint-2",
                "url": "https://interfax.ru/business/1",
                "text": TEXT,
                "text_source": "live",
            },
            {
                "story_id": "homepage",
                "url": "http://www.vesti.ru/doc.html?id=1",
                "final_url": "https://www.vesti.ru/",
                "text": "Главные новости дня",
                "text_source": "live",
            },
            {"story_id": "empty", "url": "https://a.ru/1", "error": "HTTP 404"},
        ],
    )

    junk = junk_mask(articles, repeats=3)

    assert articles.loc[junk, "story_id"].tolist() == [
        "footer-0",
        "footer-1",
        "footer-2",
        "homepage",
    ]


def test_needs_wayback_skips_tried_and_marks_junk(settings: NewsSettings) -> None:
    articles = articles_frame(
        [
            {"story_id": "dead", "url": "https://a.ru/1", "error": "HTTP 404"},
            {
                "story_id": "tried",
                "url": "https://a.ru/2",
                "error": "HTTP 404; wayback: нет снимка",
            },
            {
                "story_id": "homepage",
                "url": "http://www.vesti.ru/doc.html?id=1",
                "final_url": "https://www.vesti.ru/",
                "text": "Главные новости дня",
                "text_source": "live",
            },
            {
                "story_id": "ok",
                "url": "https://a.ru/3",
                "text": TEXT,
                "text_source": "live",
            },
        ],
    )

    todo = needs_wayback(articles, settings).set_index("story_id")["error"]

    assert todo.to_dict() == {"dead": "HTTP 404", "homepage": "не статья"}
