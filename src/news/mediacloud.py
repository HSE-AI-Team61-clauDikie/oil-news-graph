import logging
import math
from collections.abc import Iterator
from datetime import date, timedelta
from urllib.parse import urlsplit

import pandas as pd
import requests
from mediacloud.api import DirectoryApi, SearchApi
from mediacloud.error import APIResponseError
from mediacloud.types import Story
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from news.settings import NewsSettings

logger = logging.getLogger(__name__)


def is_transient(error: BaseException) -> bool:
    """
    Ошибка, после которой запрос к Media Cloud стоит повторить.

    Пустой ответ старые серверы Media Cloud отдают вместо 429.
    """

    if isinstance(error, APIResponseError):
        status: int = error.response.status_code
        return status == 429 or status >= 500 or error.data.get("status") == "empty"
    return isinstance(error, requests.ConnectionError | requests.Timeout)


retrying = retry(
    retry=retry_if_exception(is_transient),
    wait=wait_exponential(multiplier=2, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """
    Разбить период на календарные месяцы.

    Parameters:
        - start: первый день периода;
        - end: последний день периода, включительно.

    Returns:
        Пары (первый, последний день) без пересечений и дыр.
    """

    windows: list[tuple[date, date]] = []
    current = start
    while current <= end:
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        windows.append((current, min(next_month - timedelta(days=1), end)))
        current = next_month
    return windows


def iter_stories(
    api: SearchApi,
    settings: NewsSettings,
    start: date,
    end: date,
) -> Iterator[Story]:
    """
    Выдать все истории запроса за период, страница за страницей.

    Parameters:
        - api: клиент поиска Media Cloud;
        - settings: запрос, коллекции и размер страницы;
        - start: первый день;
        - end: последний день, включительно.

    Returns:
        Истории в порядке выдачи Media Cloud.
    """

    story_list = retrying(api.story_list)
    token: str | None = None
    while True:
        stories, token = story_list(
            settings.query,
            start,
            end,
            collection_ids=settings.collection_ids,
            pagination_token=token,
            page_size=settings.page_size,
        )
        yield from stories
        if token is None:
            return


def to_frame(stories: list[Story]) -> pd.DataFrame:
    """
    Привести истории Media Cloud к таблице индекса.

    Parameters:
        - stories: истории из `story_list`.

    Returns:
        Таблица с колонкой `story_id` вместо `id`.
    """

    frame = pd.DataFrame(
        stories,
        columns=[
            "id",
            "publish_date",
            "indexed_date",
            "media_name",
            "media_url",
            "title",
            "url",
            "language",
        ],
    )
    frame["publish_date"] = pd.to_datetime(frame["publish_date"])
    return frame.rename(columns={"id": "story_id"})


def count(api: SearchApi, settings: NewsSettings) -> pd.DataFrame:
    """
    Оценить объём выдачи по месяцам одним запросом.

    Parameters:
        - api: клиент поиска Media Cloud;
        - settings: запрос, коллекции, период и размер страницы.

    Returns:
        Число историй и запросов на их выкачку по месяцам.
    """

    points = retrying(api.story_count_over_time)(
        settings.query,
        settings.start_date,
        settings.end_date,
        collection_ids=settings.collection_ids,
    )
    daily = pd.DataFrame(points, columns=["date", "count"])
    monthly = (
        daily.groupby(pd.to_datetime(daily["date"]).dt.strftime("%Y-%m"))["count"]
        .sum()
        .rename("stories")
        .rename_axis("month")
        .reset_index()
    )
    monthly["requests"] = [
        max(1, math.ceil(stories / settings.page_size))
        for stories in monthly["stories"]
    ]
    return monthly


def download(api: SearchApi, settings: NewsSettings) -> None:
    """
    Выкачать индекс историй помесячно в `index_dir`.

    Готовые месяцы пропускаются, поэтому после исчерпания квоты прогон
    продолжается с первого недокачанного месяца.

    Parameters:
        - api: клиент поиска Media Cloud;
        - settings: запрос, период и каталог индекса.
    """

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    for start, end in month_windows(settings.start_date, settings.end_date):
        path = settings.index_dir / f"stories_{start:%Y-%m}.parquet"
        if path.exists():
            logger.info(f"{path.name} уже скачан")
            continue
        frame = to_frame(list(iter_stories(api, settings, start, end)))
        # Через временный файл: прерванный месяц не должен выглядеть готовым
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        logger.info(f"{path.name}: {len(frame)} историй")


def normalize_url(url: str) -> str:
    """
    Ключ URL для поиска дублей.

    Схема, `www.`, хвостовой `/` и метки рекламных кампаний отбрасываются;
    остальные параметры остаются: на части сайтов в них id статьи.

    Parameters:
        - url: адрес статьи.

    Returns:
        Нормализованный адрес без схемы.
    """

    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    params = sorted(
        param
        for param in parts.query.split("&")
        if param and not param.startswith(("utm_", "fbclid=", "gclid=", "yclid="))
    )
    query = f"?{'&'.join(params)}" if params else ""
    return f"{host}{parts.path.rstrip('/')}{query}"


def merge(settings: NewsSettings) -> pd.DataFrame:
    """
    Склеить месячные файлы индекса и убрать дубли.

    Parameters:
        - settings: каталог индекса и путь результата.

    Returns:
        Индекс без повторов по `story_id` и по нормализованному URL.

    Raises:
        FileNotFoundError: индекс ещё не скачан.
    """

    paths = sorted(settings.index_dir.glob("stories_*.parquet"))
    if not paths:
        msg = f"В {settings.index_dir} нет файлов индекса, сначала download"
        raise FileNotFoundError(msg)

    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["url_key"] = frame["url"].map(normalize_url)
    stories = (
        frame.sort_values("publish_date", kind="stable")
        .drop_duplicates("story_id")
        .drop_duplicates("url_key")
        .drop(columns="url_key")
        .reset_index(drop=True)
    )
    settings.stories_path.parent.mkdir(parents=True, exist_ok=True)
    stories.to_parquet(settings.stories_path, index=False)
    logger.info(f"{settings.stories_path}: {len(stories)} из {len(frame)} историй")
    return stories


def find_collections(api: DirectoryApi, name: str) -> pd.DataFrame:
    """
    Найти коллекции онлайн-новостей по названию.

    Parameters:
        - api: клиент каталога Media Cloud;
        - name: часть названия, например `Russia`.

    Returns:
        Коллекции в том виде, в каком их отдаёт каталог.
    """

    page = api.collection_list(
        platform=DirectoryApi.PLATFORM_ONLINE_NEWS,
        name=name,
        limit=100,
    )
    return pd.DataFrame(page["results"])
