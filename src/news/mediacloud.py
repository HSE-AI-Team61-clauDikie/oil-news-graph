import logging
import math
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
from mediacloud.api import DirectoryApi, SearchApi
from mediacloud.error import APIResponseError
from mediacloud.types import Story
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from news.settings import NewsSettings
from news.storage import write_atomic

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
    # Пауза перед повтором длится до двух минут: без строки в логе её не
    # отличить от зависания
    before_sleep=before_sleep_log(logger, logging.WARNING),
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
    page = 0
    while True:
        stories, token = story_list(
            settings.query,
            start,
            end,
            collection_ids=settings.collection_ids,
            pagination_token=token,
            page_size=settings.page_size,
        )
        page += 1
        if token is not None:
            logger.info(f"{start:%Y-%m-%d} - {end:%Y-%m-%d}: страница {page}")
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


def downloaded_months(index_dir: Path) -> set[str]:
    """
    Месяцы, уже скачанные в `index_dir`.

    Parameters:
        - index_dir: каталог файлов `stories_YYYY-MM[_YYYY-MM].parquet`.

    Returns:
        Месяцы `YYYY-MM`, покрытые файлами индекса.
    """

    months: set[str] = set()
    for path in index_dir.glob("stories_*.parquet"):
        bounds = path.stem.removeprefix("stories_").split("_")
        first = date.fromisoformat(f"{bounds[0]}-01")
        last = date.fromisoformat(f"{bounds[-1]}-01")
        months |= {f"{start:%Y-%m}" for start, _ in month_windows(first, last)}
    return months


def pack_months(
    windows: list[tuple[date, date]],
    sizes: dict[str, int],
    page_size: int,
) -> list[tuple[date, date]]:
    """
    Склеить соседние месяцы в окна, пока окно умещается в одну страницу.

    Media Cloud считает квоту запросами, а не историями: месяц с десятком
    историй стоит столько же, сколько месяц с тысячей.

    Parameters:
        - windows: месяцы по возрастанию;
        - sizes: число историй по месяцам `YYYY-MM`;
        - page_size: историй на странице.

    Returns:
        Окна (первый, последний день); месяцы не из `windows` в них не попадают.
    """

    packed: list[tuple[date, date]] = []
    filled = 0
    for start, end in windows:
        size = sizes.get(f"{start:%Y-%m}", 0)
        adjacent = packed and start - packed[-1][1] == timedelta(days=1)
        if adjacent and filled + size <= page_size:
            packed[-1] = (packed[-1][0], end)
            filled += size
        else:
            packed.append((start, end))
            filled = size
    return packed


def download(api: SearchApi, settings: NewsSettings) -> None:
    """
    Выкачать индекс историй в `index_dir` окнами из одного или нескольких месяцев.

    Скачанные месяцы пропускаются, поэтому после исчерпания квоты прогон
    продолжается с первого недокачанного. Соседние малые месяцы идут одним
    окном (`pack_months`): объём по месяцам - ещё один запрос.

    Parameters:
        - api: клиент поиска Media Cloud;
        - settings: запрос, период и каталог индекса.
    """

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    done = downloaded_months(settings.index_dir)
    todo = [
        window
        for window in month_windows(settings.start_date, settings.end_date)
        if f"{window[0]:%Y-%m}" not in done
    ]
    logger.info(f"Скачано месяцев: {len(done)}, осталось {len(todo)}")
    if not todo:
        return
    monthly = count(api, settings)
    sizes = dict(zip(monthly["month"], monthly["stories"], strict=True))
    windows = pack_months(todo, sizes, settings.page_size)
    for number, (start, end) in enumerate(windows, start=1):
        name = (
            f"{start:%Y-%m}"
            if start.month == end.month and start.year == end.year
            else f"{start:%Y-%m}_{end:%Y-%m}"
        )
        path = settings.index_dir / f"stories_{name}.parquet"
        frame = to_frame(list(iter_stories(api, settings, start, end)))
        # Через временный файл: прерванное окно не должно выглядеть готовым
        write_atomic(frame, path)
        logger.info(f"{path.name}: {len(frame)} историй, окно {number}/{len(windows)}")


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
    write_atomic(stories, settings.stories_path)
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
