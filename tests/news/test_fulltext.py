import asyncio
import ssl
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from news.fulltext import (
    Article,
    ArticleFetcher,
    fetch_in_parts,
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


async def test_tls_failure_is_an_error_not_a_crash(settings: NewsSettings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        msg = "decryption failed or bad record mac"
        raise ssl.SSLError(msg)

    article = await make_fetcher(settings, handler).fetch_live("1", "https://x.ru/1")

    assert article.text_source == "none"
    assert article.error.startswith("SSLError")


async def test_relative_url_is_skipped(settings: NewsSettings) -> None:
    fetcher = make_fetcher(settings, lambda _: httpx.Response(500))

    article = await fetcher.fetch_live("1", "/mersin-haberleri/1")

    assert article.error == "адрес без схемы"


@pytest.mark.parametrize(
    "url",
    ["https://example.org/news\x01", "https://[broken/", "https://example.org:bad/"],
)
async def test_malformed_url_does_not_abort_neighbouring_articles(
    settings: NewsSettings,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["first", "bad", "last"],
            "url": ["https://example.org/first", url, "https://example.org/last"],
            "publish_date": pd.to_datetime(["2017-03-28"] * 3),
        },
    ).to_parquet(settings.stories_path)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(404)

    patch_client(monkeypatch, handler)
    settings = settings.model_copy(update={"use_wayback": False})
    await run(settings)
    articles = read_articles(settings).set_index("story_id")
    assert set(articles.index) == {"first", "bad", "last"}
    assert articles.loc["bad", "text_source"] == "none"
    assert articles.loc["bad", "error"]
    assert articles.loc[["first", "last"], "http_status"].eq(404).all()
    assert len(requests) == 2

    await run(settings)
    assert len(requests) == 2


@pytest.mark.parametrize("retry_failed", [False, True])
async def test_resume_reads_only_attempt_metadata(
    settings: NewsSettings,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retry_failed: bool,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["existing"],
            "url": ["https://example.org/existing"],
            "publish_date": pd.to_datetime(["2017-03-28"]),
        },
    ).to_parquet(settings.stories_path)
    settings.fulltext_dir.mkdir()
    # Последняя попытка не обязана находиться в файле с большим номером.
    for number, source, day in [(0, "none", "2026-11-16"), (1, "live", "2026-11-15")]:
        pd.DataFrame(
            [asdict(Article("existing", "https://example.org/existing"))],
        ).assign(text_source=source, fetched_at=pd.Timestamp(day, tz="UTC")).to_parquet(
            settings.fulltext_dir / f"part-{number:05d}.parquet",
        )
    original = pd.read_parquet
    projections = []

    def read(path: Path, **kwargs) -> pd.DataFrame:
        if Path(path).parent == settings.fulltext_dir:
            columns = kwargs.get("columns")
            assert columns is not None
            assert "text" not in columns
            projections.append(set(columns))
        return original(path, **kwargs)

    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(404)

    patch_client(monkeypatch, handler)
    monkeypatch.setattr(pd, "read_parquet", read)
    await run(
        settings.model_copy(update={"use_wayback": False}),
        retry_failed=retry_failed,
    )
    expected = (
        {"story_id", "text_source", "fetched_at"} if retry_failed else {"story_id"}
    )
    assert projections == [expected, expected]
    assert len(requested) == int(retry_failed)


async def test_slow_request_does_not_block_later_parts(settings: NewsSettings) -> None:
    settings = settings.model_copy(update={"concurrency": 2})
    release = asyncio.Event()
    later_started = asyncio.Event()
    active = 0
    peak = 0
    todo = pd.DataFrame({"story_id": ["slow", "a", "b", "c", "d"]})

    async def fetch(row) -> Article:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if row.story_id == "slow":
                await release.wait()
            elif row.story_id == "d":
                later_started.set()
            return Article(row.story_id, "https://example.org/", error="HTTP 404")
        finally:
            active -= 1

    task = asyncio.create_task(fetch_in_parts(settings, todo, fetch, 2))
    try:
        await asyncio.wait_for(later_started.wait(), timeout=1)
        assert set(read_articles(settings).story_id) == {"a", "b"}
        assert peak <= 2
    finally:
        release.set()
        await task
    assert set(read_articles(settings).story_id) == set(todo.story_id)
    assert len(list(settings.fulltext_dir.glob("part-*.parquet"))) == 3


async def test_cancel_saves_completed_results_and_cancels_pending(
    settings: NewsSettings,
) -> None:
    settings = settings.model_copy(update={"concurrency": 2})
    started = asyncio.Event()
    blocked = asyncio.Event()
    todo = pd.DataFrame({"story_id": ["slow", "a", "b", "later"]})
    cancelled = []

    async def fetch(row) -> Article:
        if row.story_id in {"slow", "later"}:
            if row.story_id == "later":
                started.set()
            try:
                await blocked.wait()
            except asyncio.CancelledError:
                cancelled.append(row.story_id)
                raise
        return Article(row.story_id, "https://example.org/", error="HTTP 404")

    task = asyncio.create_task(fetch_in_parts(settings, todo, fetch, 5))
    await asyncio.wait_for(started.wait(), timeout=1)
    # Дать обработчику принять уже завершённые результаты перед отменой.
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(read_articles(settings).story_id) == {"a", "b"}
    assert set(cancelled) == {"slow", "later"}


async def test_continuous_fetch_preserves_global_and_host_limits(
    settings: NewsSettings,
) -> None:
    settings = settings.model_copy(update={"concurrency": 3, "per_host_concurrency": 2})
    active = 0
    hosts: dict[str, int] = {}
    peak = 0
    host_peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, host_peak
        host = request.url.host
        active += 1
        hosts[host] = hosts.get(host, 0) + 1
        peak = max(peak, active)
        host_peak = max(host_peak, hosts[host])
        await asyncio.sleep(0.001)
        hosts[host] -= 1
        active -= 1
        return httpx.Response(404)

    todo = pd.DataFrame(
        {
            "story_id": [str(i) for i in range(12)],
            "url": [f"https://host{i % 2}.example.org/{i}" for i in range(12)],
        },
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = ArticleFetcher(client, settings)
        await fetch_in_parts(
            settings,
            todo,
            lambda row: fetcher.fetch_live(row.story_id, row.url),
            6,
        )
    assert peak == settings.concurrency
    assert host_peak <= settings.per_host_concurrency
    assert len(read_articles(settings)) == len(todo)


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


async def test_run_without_wayback_stops_after_live(
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
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(404)

    patch_client(monkeypatch, handler)

    await run(settings.model_copy(update={"use_wayback": False}))

    assert hosts == ["kp.ru"]
    assert read_articles(settings).iloc[0]["error"] == "HTTP 404"


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


async def test_run_limit_resumes_with_next_unfetched_articles(
    settings: NewsSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["old", "middle", "new"],
            "url": [f"https://a.ru/{key}" for key in ["old", "middle", "new"]],
            "publish_date": pd.to_datetime(["2020-01-01", "2023-01-01", "2026-01-01"]),
        },
    ).to_parquet(settings.stories_path)
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, content=article_html())

    patch_client(monkeypatch, handler)
    settings = settings.model_copy(update={"use_wayback": False})
    await run(settings, newest_first=True, max_articles=2)
    assert requested == ["/new", "/middle"]
    assert len(read_articles(settings)) == 2

    await run(settings, newest_first=True, max_articles=2)
    assert requested == ["/new", "/middle", "/old"]
    assert len(read_articles(settings)) == 3


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
