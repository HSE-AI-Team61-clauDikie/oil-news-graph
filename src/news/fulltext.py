import asyncio
import hashlib
import logging
import re
import ssl
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
import pandas as pd
import trafilatura
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from news.settings import NewsSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Article:
    """Результат загрузки одной статьи; `text_source` - live, wayback или none."""

    story_id: str
    url: str
    final_url: str | None = None
    text: str | None = None
    text_source: str = "none"
    wayback_timestamp: str | None = None
    http_status: int | None = None
    error: str | None = None
    extracted_title: str | None = None
    extracted_date: str | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ArticleFetcher:
    """
    Загрузка статей с сайтов изданий и из Wayback Machine.

    Лимиты общие на весь прогон, поэтому экземпляр один на прогон.
    """

    def __init__(self, client: httpx.AsyncClient, settings: NewsSettings) -> None:
        self.client = client
        self.settings = settings
        self.limit = asyncio.Semaphore(settings.concurrency)
        self.host_limits: defaultdict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(settings.per_host_concurrency),
        )
        self.wayback_lock = asyncio.Lock()
        self.wayback_ready_at = 0.0

    async def fetch_live(self, story_id: str, url: str) -> Article:
        """
        Загрузить статью с сайта издания.

        Parameters:
            - story_id: id истории Media Cloud;
            - url: адрес статьи.

        Returns:
            Статья с `text_source="live"` или с ошибкой.
        """

        if not url.startswith(("http://", "https://")):
            return Article(story_id, url, error="адрес без схемы")
        try:
            # Сначала слот сайта, потом общий: задача, ждущая занятый сайт,
            # не должна держать общий слот и тормозить остальные сайты
            async with self.host_limits[urlsplit(url).netloc], self.limit:
                response = await self.client.get(url)
        # Обрыв TLS на чтении тела httpx не оборачивает в свою ошибку
        except (httpx.HTTPError, ssl.SSLError) as error:
            return Article(story_id, url, error=f"{type(error).__name__}: {error}")
        if response.status_code != httpx.codes.OK:
            return Article(
                story_id,
                url,
                final_url=str(response.url),
                http_status=response.status_code,
                error=f"HTTP {response.status_code}",
            )
        return await self.to_article(Article(story_id, url), response, "live")

    async def fetch_wayback(
        self,
        story_id: str,
        url: str,
        publish_date: date | None,
    ) -> Article:
        """
        Найти ближайший к дате публикации снимок в Wayback Machine.

        Parameters:
            - story_id: id истории Media Cloud;
            - url: адрес статьи;
            - publish_date: дата публикации или None.

        Returns:
            Статья с `text_source="wayback"` или с ошибкой.
        """

        params: dict[str, str | int] = {
            "url": url,
            "output": "json",
            "filter": "statuscode:200",
            "limit": 1,
        }
        if publish_date is not None:
            params |= {"closest": f"{publish_date:%Y%m%d}", "sort": "closest"}
        try:
            index = await self.wayback_get(
                "https://web.archive.org/cdx/search/cdx",
                params,
            )
            rows: list[list[str]] = index.json() if index.content else []
            if len(rows) < 2:
                return Article(story_id, url, error="нет снимка")
            # Первая строка CDX - заголовок, вторая колонка - timestamp снимка
            timestamp = rows[1][1]
            # id_ отдаёт исходный HTML без панели и переписанных ссылок Wayback
            snapshot = await self.wayback_get(
                f"https://web.archive.org/web/{timestamp}id_/{url}",
            )
        except (httpx.HTTPError, ValueError) as error:
            return Article(story_id, url, error=f"{type(error).__name__}: {error}")
        article = Article(story_id, url, wayback_timestamp=timestamp)
        return await self.to_article(article, snapshot, "wayback")

    async def fetch_archived(
        self,
        story_id: str,
        url: str,
        publish_date: date | None,
        live_error: str,
    ) -> Article:
        """
        Повторить статью, не давшую текста на сайте, через Wayback.

        Parameters:
            - story_id: id истории Media Cloud;
            - url: адрес статьи;
            - publish_date: дата публикации или None;
            - live_error: почему не подошла живая страница.

        Returns:
            Статья из Wayback или запись с обеими причинами.
        """

        archived = await self.fetch_wayback(story_id, url, publish_date)
        if archived.text_source == "wayback":
            return archived
        return Article(story_id, url, error=f"{live_error}; wayback: {archived.error}")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=15, max=300),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def wayback_get(
        self,
        url: str,
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        """
        GET к archive.org не чаще раза в `wayback_interval` секунд.

        Parameters:
            - url: адрес CDX или снимка;
            - params: параметры запроса.

        Returns:
            Ответ archive.org.

        Raises:
            httpx.HTTPStatusError: 429 и 5xx после всех повторов.
        """

        loop = asyncio.get_running_loop()
        async with self.wayback_lock:
            await asyncio.sleep(max(0.0, self.wayback_ready_at - loop.time()))
            self.wayback_ready_at = loop.time() + self.settings.wayback_interval
        response = await self.client.get(url, params=params)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            # Пауза для всей очереди: иначе следующие запросы тоже получат 429
            self.wayback_ready_at = loop.time() + 60
        if (
            response.status_code == httpx.codes.TOO_MANY_REQUESTS
            or response.is_server_error
        ):
            response.raise_for_status()
        return response

    async def to_article(
        self,
        article: Article,
        response: httpx.Response,
        source: str,
    ) -> Article:
        """
        Извлечь текст из ответа и проверить, что это статья.

        Parameters:
            - article: заготовка с id и адресом;
            - response: ответ сайта или Wayback;
            - source: live или wayback.

        Returns:
            Статья с текстом; короче `min_text_chars` - с ошибкой.
        """

        article = replace(
            article,
            final_url=str(response.url),
            http_status=response.status_code,
        )
        if response.status_code != httpx.codes.OK:
            return replace(article, error=f"HTTP {response.status_code}")
        # Байты, а не response.text: кодировку старых сайтов в cp1251
        # trafilatura определяет по meta надёжнее, чем httpx по заголовкам
        document = await asyncio.to_thread(
            trafilatura.bare_extraction,
            response.content,
            url=article.url,
            with_metadata=True,
            include_comments=False,
        )
        if document is None or not document.text:
            return replace(article, error="текст не извлечён")
        article = replace(
            article,
            text=document.text,
            extracted_title=document.title,
            extracted_date=document.date,
        )
        if len(document.text) < self.settings.min_text_chars:
            return replace(article, error=f"короткий текст: {len(document.text)}")
        return replace(article, text_source=source)


def read_articles(settings: NewsSettings) -> pd.DataFrame:
    """
    Прочитать все part-файлы, по каждой истории - последнюю попытку.

    Parameters:
        - settings: каталог part-файлов.

    Returns:
        Таблица статей; пустая, если ещё ничего не скачано.
    """

    paths = sorted(settings.fulltext_dir.glob("part-*.parquet"))
    if not paths:
        return pd.DataFrame(columns=[column.name for column in fields(Article)])
    articles = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return (
        articles.sort_values("fetched_at", kind="stable")
        .drop_duplicates("story_id", keep="last")
        .reset_index(drop=True)
    )


def junk_mask(articles: pd.DataFrame, repeats: int) -> pd.Series:
    """
    Отметить тексты, которые не являются статьёй.

    Это редирект на главную сайта или один и тот же текст на `repeats` и больше
    страницах одного сайта: подвал, лента анонсов, заглушка вместо статьи.

    Parameters:
        - articles: таблица из `read_articles`;
        - repeats: сколько одинаковых текстов с сайта считать шаблоном.

    Returns:
        Булева маска по строкам `articles`.
    """

    with_text = articles[articles["text"].notna() & (articles["text_source"] != "none")]
    hosts = with_text["url"].map(
        lambda url: urlsplit(url).netloc.lower().removeprefix("www."),
    )
    repeated = with_text.groupby([hosts, with_text["text"].map(text_fingerprint)])[
        "story_id"
    ].transform("size")
    to_homepage = [
        urlsplit(final).path in {"", "/"} and urlsplit(url).path not in {"", "/"}
        for url, final in zip(
            with_text["url"],
            with_text["final_url"].fillna("/x"),
            strict=True,
        )
    ]
    junk = (repeated >= repeats) | pd.Series(to_homepage, index=with_text.index)
    # astype: на пустой таблице маска получается object, и ~ ломает её
    return junk.reindex(articles.index, fill_value=False).astype(bool)


def needs_wayback(articles: pd.DataFrame, settings: NewsSettings) -> pd.DataFrame:
    """
    Статьи, которые стоит поискать в Wayback.

    Это статьи без текста и с текстом-шаблоном; уже искавшиеся в Wayback
    не повторяются.

    Parameters:
        - articles: таблица из `read_articles`;
        - settings: порог шаблона.

    Returns:
        Строки `articles`; у шаблонов в `error` - «не статья».
    """

    junk = junk_mask(articles, settings.junk_repeats)
    tried = (articles["text_source"] == "wayback") | articles["error"].fillna(
        "",
    ).str.contains("wayback:", regex=False)
    todo = articles[~tried & ((articles["text_source"] == "none") | junk)].copy()
    todo["error"] = todo["error"].where(todo["text_source"] == "none", "не статья")
    return todo


async def fetch_in_parts(
    settings: NewsSettings,
    todo: pd.DataFrame,
    fetch: Callable[[Any], Awaitable[Article]],
    part_size: int,
) -> None:
    """
    Обработать строки пачками и сохранить каждую пачку в свой part-файл.

    Parameters:
        - settings: каталог part-файлов;
        - todo: строки к загрузке;
        - fetch: загрузка одной строки из `todo.itertuples()`;
        - part_size: строк в пачке.
    """

    settings.fulltext_dir.mkdir(parents=True, exist_ok=True)
    part = len(list(settings.fulltext_dir.glob("part-*.parquet")))
    for start in range(0, len(todo), part_size):
        batch = todo.iloc[start : start + part_size]
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(fetch(row)) for row in batch.itertuples()]
        articles = pd.DataFrame([asdict(task.result()) for task in tasks])
        path = settings.fulltext_dir / f"part-{part:05d}.parquet"
        tmp = path.with_suffix(".tmp")
        articles.to_parquet(tmp, index=False)
        tmp.replace(path)
        part += 1
        sources = articles["text_source"].value_counts().to_dict()
        left = len(todo) - start - len(batch)
        logger.info(f"{path.name}: {sources}, осталось {left}")


async def run(
    settings: NewsSettings,
    *,
    retry_failed: bool = False,
    newest_first: bool = False,
    max_articles: int | None = None,
) -> None:
    """
    Скачать тексты для индекса `stories_path` в part-файлы.

    Сначала живые страницы всех статей, потом, если `use_wayback`, Wayback для
    тех, что не дали текста: медленный archive.org не задерживает живые. Скачанное
    раньше пропускается; с `retry_failed` статьи без текста идут заново.

    Parameters:
        - settings: пути, лимиты и таймауты;
        - retry_failed: повторить статьи без текста;
        - newest_first: идти от свежих статей к старым.
        - max_articles: не больше стольких новых живых страниц за проход.
    """

    stories = pd.read_parquet(
        settings.stories_path,
        columns=["story_id", "url", "publish_date"],
    ).sort_values("publish_date", ascending=not newest_first, kind="stable")
    fetched = read_articles(settings)
    if retry_failed:
        fetched = fetched[fetched["text_source"] != "none"]
    live_todo = stories[~stories["story_id"].isin(fetched["story_id"])]
    if max_articles is not None:
        live_todo = live_todo.head(max_articles)

    async with httpx.AsyncClient(
        headers={
            "User-Agent": settings.user_agent,
            "Accept-Language": settings.accept_language,
        },
        timeout=settings.request_timeout,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=settings.concurrency),
    ) as client:
        fetcher = ArticleFetcher(client, settings)

        logger.info(f"Живые страницы: {len(live_todo)} из {len(stories)} историй")
        await fetch_in_parts(
            settings,
            live_todo,
            lambda row: fetcher.fetch_live(row.story_id, row.url),
            settings.batch_size,
        )
        if not settings.use_wayback:
            return

        # Порядок задаёт stories: merge сохраняет порядок левой таблицы
        wayback_todo = stories[["story_id", "publish_date"]].merge(
            needs_wayback(read_articles(settings), settings),
            on="story_id",
        )
        wayback_todo["publish_day"] = [
            day.date() if pd.notna(day) else None
            for day in wayback_todo["publish_date"]
        ]
        logger.info(f"Wayback: {len(wayback_todo)} историй")
        await fetch_in_parts(
            settings,
            wayback_todo,
            lambda row: fetcher.fetch_archived(
                row.story_id,
                row.url,
                row.publish_day,
                row.error,
            ),
            settings.wayback_batch_size,
        )


def text_fingerprint(text: str) -> str:
    """
    Хэш текста без учёта регистра, пунктуации и пробелов.

    Parameters:
        - text: текст статьи.

    Returns:
        sha1 нормализованного текста.
    """

    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(normalized.encode(), usedforsecurity=False).hexdigest()
