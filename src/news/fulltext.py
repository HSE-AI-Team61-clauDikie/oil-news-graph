import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, date, datetime
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
    Загрузка статей: живая страница, при неудаче - снимок Wayback Machine.

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

    async def fetch_article(
        self,
        story_id: str,
        url: str,
        publish_date: date | None,
    ) -> Article:
        """
        Загрузить статью и извлечь текст.

        Parameters:
            - story_id: id истории Media Cloud;
            - url: адрес статьи;
            - publish_date: дата публикации, по ней ищется ближайший снимок.

        Returns:
            Статья с текстом или с причиной, почему текста нет.
        """

        async with self.limit:
            live = await self.fetch_live(story_id, url)
            if live.text_source == "live":
                return live
            archived = await self.fetch_wayback(story_id, url, publish_date)
        if archived.text_source == "wayback":
            return archived
        return replace(live, error=f"{live.error}; wayback: {archived.error}")

    async def fetch_live(self, story_id: str, url: str) -> Article:
        """
        Загрузить статью с сайта издания.

        Parameters:
            - story_id: id истории Media Cloud;
            - url: адрес статьи.

        Returns:
            Статья с `text_source="live"` или с ошибкой.
        """

        try:
            async with self.host_limits[urlsplit(url).netloc]:
                response = await self.client.get(url)
        except httpx.HTTPError as error:
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


async def run(settings: NewsSettings, *, retry_failed: bool = False) -> None:
    """
    Скачать тексты для индекса `stories_path` батчами в part-файлы.

    Истории, которые уже есть в part-файлах, пропускаются; с `retry_failed`
    повторяются те из них, у которых текста нет.

    Parameters:
        - settings: пути, лимиты и таймауты;
        - retry_failed: повторить истории без текста.
    """

    stories = pd.read_parquet(
        settings.stories_path,
        columns=["story_id", "url", "publish_date"],
    )
    fetched = read_articles(settings)
    if retry_failed:
        fetched = fetched[fetched["text_source"] != "none"]
    todo = stories[~stories["story_id"].isin(fetched["story_id"])]
    logger.info(f"К загрузке {len(todo)} из {len(stories)} историй")

    settings.fulltext_dir.mkdir(parents=True, exist_ok=True)
    part = len(list(settings.fulltext_dir.glob("part-*.parquet")))
    async with httpx.AsyncClient(
        headers={
            "User-Agent": settings.user_agent,
            "Accept-Language": "ru,en;q=0.8",
        },
        timeout=settings.request_timeout,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=settings.concurrency),
    ) as client:
        fetcher = ArticleFetcher(client, settings)
        for start in range(0, len(todo), settings.batch_size):
            batch = todo.iloc[start : start + settings.batch_size]
            tasks: list[asyncio.Task[Article]] = []
            async with asyncio.TaskGroup() as group:
                for row in batch.itertuples():
                    day = (
                        row.publish_date.date() if pd.notna(row.publish_date) else None
                    )
                    article = fetcher.fetch_article(row.story_id, row.url, day)
                    tasks.append(group.create_task(article))
            articles = pd.DataFrame([asdict(task.result()) for task in tasks])
            path = settings.fulltext_dir / f"part-{part:05d}.parquet"
            tmp = path.with_suffix(".tmp")
            articles.to_parquet(tmp, index=False)
            tmp.replace(path)
            part += 1
            sources = articles["text_source"].value_counts().to_dict()
            logger.info(f"{path.name}: {sources}")


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


def assemble(settings: NewsSettings) -> pd.DataFrame:
    """
    Собрать датасет: индекс Media Cloud и тексты статей.

    Перепечатки одного текста остаются в датасете, у всех, кроме самой ранней,
    заполнено `duplicate_of`.

    Parameters:
        - settings: пути индекса, part-файлов и результата.

    Returns:
        Статьи с текстом.
    """

    stories = pd.read_parquet(settings.stories_path)
    articles = read_articles(settings)
    articles = articles[articles["text_source"] != "none"].drop(columns="url")
    dataset = (
        stories.merge(articles, on="story_id", how="inner", validate="one_to_one")
        .sort_values(["publish_date", "story_id"], kind="stable")
        .reset_index(drop=True)
    )
    dataset["text_hash"] = dataset["text"].map(text_fingerprint)
    first = dataset.groupby("text_hash")["story_id"].transform("first")
    dataset["duplicate_of"] = first.where(first != dataset["story_id"])
    settings.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(settings.dataset_path, index=False)
    logger.info(
        f"{settings.dataset_path}: {len(dataset)} статей, "
        f"перепечаток {dataset['duplicate_of'].notna().sum()}",
    )
    return dataset
