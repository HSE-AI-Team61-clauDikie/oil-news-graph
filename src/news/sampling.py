import logging
import zlib
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from mediacloud.api import SearchApi

from news.mediacloud import normalize_url, retrying, to_frame
from news.regions import Region
from news.settings import NewsSettings
from news.storage import write_atomic

logger = logging.getLogger(__name__)


def sample_settings(base: NewsSettings) -> NewsSettings:
    """
    Настройки `fulltext` для выборки на разметку.

    Parameters:
        - base: общие настройки.

    Returns:
        Копия `base`: индекс - выборка, тексты - в свой каталог, без Wayback.
    """

    return base.model_copy(
        update={
            "stories_path": base.sample_dir / "stories.parquet",
            "fulltext_dir": Path("data/raw/news/sample/fulltext"),
            "use_wayback": False,
            "accept_language": "*",
        },
    )


def daily_count(
    api: SearchApi,
    query: str,
    region: Region,
    settings: NewsSettings,
) -> pd.Series:
    """
    Число историй запроса по дням.

    Parameters:
        - api: клиент поиска Media Cloud;
        - query: Lucene-запрос;
        - region: коллекции;
        - settings: период.

    Returns:
        Число историй, индекс - дата.
    """

    points = retrying(api.story_count_over_time)(
        query,
        settings.start_date,
        settings.end_date,
        collection_ids=region.collection_ids,
    )
    daily = pd.DataFrame(points, columns=["date", "count"])
    return daily.set_index(pd.to_datetime(daily["date"]))["count"]


def stratum_counts(
    api: SearchApi,
    region: Region,
    settings: NewsSettings,
) -> pd.DataFrame:
    """
    Объём страт региона по дням: кандидаты и фон.

    Parameters:
        - api: клиент поиска Media Cloud;
        - region: запросы и коллекции;
        - settings: период.

    Returns:
        Таблица `region`, `date`, `candidates`, `background`.
    """

    candidates = daily_count(api, region.recall_query(), region, settings)
    total = daily_count(api, region.language_filter(), region, settings)
    counts = pd.DataFrame({"candidates": candidates, "total": total})
    counts = counts.fillna(0).astype(int).rename_axis("date").reset_index()
    counts["background"] = (counts["total"] - counts["candidates"]).clip(lower=0)
    counts.insert(0, "region", region.name)
    return counts[["region", "date", "candidates", "background"]]


def pick_days(
    volumes: pd.Series,
    need: int,
    rng: np.random.Generator,
    max_days: int,
    min_days: int = 1,
) -> list[pd.Timestamp]:
    """
    Случайные дни с вероятностью по объёму, пока в них не наберётся `need` историй.

    Отбор без возвращения по ключам Эфраимидиса-Спиракиса: день с вдвое большим
    объёмом попадает вдвое чаще, поэтому история любого дня выбирается
    примерно с равной вероятностью.

    Parameters:
        - volumes: объём по дням, индекс - дата;
        - need: сколько историй набрать;
        - rng: генератор случайных чисел;
        - max_days: не больше стольких дней;
        - min_days: не меньше стольких дней, если они есть.

    Returns:
        Даты по возрастанию; пусто, если историй нет.
    """

    volumes = volumes[volumes > 0]
    if volumes.empty:
        return []
    keys = rng.random(len(volumes)) ** (1 / volumes.to_numpy())
    ordered = volumes.iloc[np.argsort(-keys)]
    enough = int(np.searchsorted(ordered.cumsum().to_numpy(), need)) + 1
    return sorted(ordered.index[: min(max(enough, min_days), max_days)])


def pack_days(
    days: list[pd.Timestamp],
    volumes: pd.Series,
    capacity: int,
    max_days: int,
) -> list[list[pd.Timestamp]]:
    """
    Разложить дни по запросам: не больше `capacity` историй и `max_days` дней.

    День больше `capacity` идёт отдельным запросом.

    Parameters:
        - days: выбранные дни;
        - volumes: объём по дням;
        - capacity: историй на запрос;
        - max_days: дней на запрос.

    Returns:
        Группы дней, по запросу на группу.
    """

    groups: list[list[pd.Timestamp]] = []
    filled = 0
    for day in days:
        full = groups and (
            filled + volumes[day] > capacity or len(groups[-1]) >= max_days
        )
        if not groups or full:
            groups.append([])
            filled = 0
        groups[-1].append(day)
        filled += volumes[day]
    return groups


def days_per_query(query: str, settings: NewsSettings) -> int:
    """
    Сколько дней фильтра `publication_date` уместится в запрос.

    Media Cloud принимает запрос в GET: длиннее `sample_query_bytes` после
    URL-кодирования он отвечает 400.

    Parameters:
        - query: запрос страты без фильтра дней;
        - settings: предел длины.

    Returns:
        Не меньше одного дня.
    """

    per_day = len(quote(" OR [2016-01-01 TO 2016-01-01]"))
    room = settings.sample_query_bytes - len(quote(query))
    return max(1, room // per_day)


def fetch_days(
    api: SearchApi,
    region: Region,
    query: str,
    days: list[pd.Timestamp],
    page_size: int,
) -> pd.DataFrame:
    """
    Первая страница историй запроса за выбранные дни.

    `story_list` отдаёт историю в порядке даты, без ранжирования по
    совпадениям: `story_sample` для этого не годится, он детерминированно
    отдаёт самые релевантные.

    Parameters:
        - api: клиент поиска Media Cloud;
        - region: коллекции;
        - query: запрос страты;
        - days: дни группы;
        - page_size: историй на странице.

    Returns:
        Истории индекса.
    """

    ranges = " OR ".join(f"[{day:%Y-%m-%d} TO {day:%Y-%m-%d}]" for day in days)
    stories, _ = retrying(api.story_list)(
        f"{query} AND publication_date:({ranges})",
        days[0].date(),
        days[-1].date(),
        collection_ids=region.collection_ids,
        page_size=page_size,
    )
    return to_frame(stories)


def sample_candidates(
    api: SearchApi,
    region: Region,
    counts: pd.DataFrame,
    settings: NewsSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Кандидаты recall-запроса: до `sample_candidates` на год из
    `sample_min_days` дней и больше.

    Parameters:
        - api: клиент поиска Media Cloud;
        - region: запросы и коллекции;
        - counts: объёмы страт по дням;
        - settings: размеры выборки;
        - rng: генератор случайных чисел.

    Returns:
        Истории с колонками `year`, `weight`.
    """

    volumes = counts.set_index("date")["candidates"]
    days = [
        day
        for _, year_volumes in volumes.groupby(volumes.index.year)
        for day in pick_days(
            year_volumes,
            2 * settings.sample_candidates,
            rng,
            settings.sample_max_days,
            settings.sample_min_days,
        )
    ]
    # Запрос сверх страницы теряет только хвост последнего дня, зато дней
    # в году больше, а запросов не больше одного-двух на год
    capacity = settings.page_size * 3 // 2
    pool = pd.concat(
        [to_frame([])]
        + [
            fetch_days(api, region, region.recall_query(), group, settings.page_size)
            for group in pack_days(
                days,
                volumes,
                capacity,
                days_per_query(region.recall_query(), settings),
            )
        ],
        ignore_index=True,
    ).drop_duplicates("story_id")
    pool["year"] = pool["publish_date"].dt.year
    years: list[pd.DataFrame] = [pool.head(0).assign(weight=0.0)]
    for year, stories in pool.groupby("year"):
        taken = stories.sample(
            n=min(settings.sample_candidates, len(stories)),
            random_state=rng.integers(2**31),
        )
        size = volumes[volumes.index.year == year].sum()
        years.append(taken.assign(weight=size / len(taken)))
    return pd.concat(years, ignore_index=True)


def sample_background(
    api: SearchApi,
    region: Region,
    counts: pd.DataFrame,
    settings: NewsSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Фон вне recall-запроса: `sample_background` историй на год региона.

    Дни фона почти везде больше страницы `story_list`, поэтому страта фона -
    весь регион: `sample_background_days` случайных дней, поровну историй
    с каждого.

    Parameters:
        - api: клиент поиска Media Cloud;
        - region: запросы и коллекции;
        - counts: объёмы страт по дням;
        - settings: размеры выборки;
        - rng: генератор случайных чисел.

    Returns:
        Истории с колонками `year`, `weight`.
    """

    volumes = counts.set_index("date")["background"]
    total = settings.sample_background * volumes.index.year.nunique()
    days = pick_days(
        volumes,
        total,
        rng,
        max_days=settings.sample_background_days,
        min_days=settings.sample_background_days,
    )
    per_day = -(-total // max(1, len(days)))
    parts: list[pd.DataFrame] = [to_frame([])]
    for group in pack_days(
        days,
        volumes,
        settings.page_size,
        days_per_query(region.background_query(), settings),
    ):
        stories = fetch_days(
            api,
            region,
            region.background_query(),
            group,
            settings.page_size,
        )
        for _, day_stories in stories.groupby(stories["publish_date"].dt.date):
            parts.append(
                day_stories.sample(
                    n=min(per_day, len(day_stories)),
                    random_state=rng.integers(2**31),
                ),
            )
    background = pd.concat(parts, ignore_index=True).drop_duplicates("story_id")
    background["year"] = background["publish_date"].dt.year
    background["weight"] = volumes.sum() / len(background)
    return background


def sample_region(
    api: SearchApi,
    region: Region,
    counts: pd.DataFrame,
    settings: NewsSettings,
) -> pd.DataFrame:
    """
    Выборка региона: кандидаты по годам и фон.

    Parameters:
        - api: клиент поиска Media Cloud;
        - region: запросы и коллекции;
        - counts: объёмы страт по дням из `stratum_counts`;
        - settings: период и размеры выборки.

    Returns:
        Истории индекса с колонками `region`, `year`, `stratum`, `weight`.
    """

    rng = np.random.default_rng(zlib.crc32(region.name.encode()))
    frame = pd.concat(
        [
            sample_candidates(api, region, counts, settings, rng).assign(
                stratum="candidate",
            ),
            sample_background(api, region, counts, settings, rng).assign(
                stratum="background",
            ),
        ],
        ignore_index=True,
    )
    frame.insert(0, "region", region.name)
    return frame


def collect(
    api: SearchApi,
    regions: list[Region],
    settings: NewsSettings,
) -> None:
    """
    Собрать объёмы страт и выборку по регионам.

    Готовые регионы пропускаются: после исчерпания квоты прогон
    продолжается с места остановки.

    Parameters:
        - api: клиент поиска Media Cloud;
        - regions: регионы выборки;
        - settings: период, размеры выборки и каталог.
    """

    for region in regions:
        path = settings.sample_dir / "parts" / f"{region.name}.parquet"
        if path.exists():
            continue
        counts_path = settings.sample_dir / "counts" / f"{region.name}.parquet"
        if not counts_path.exists():
            write_atomic(stratum_counts(api, region, settings), counts_path)
        counts = pd.read_parquet(counts_path)
        frame = sample_region(api, region, counts, settings)
        write_atomic(frame, path)
        strata = frame["stratum"].value_counts().to_dict()
        logger.info(f"{path.name}: {strata}")


def merge(settings: NewsSettings) -> pd.DataFrame:
    """
    Склеить выборку регионов и убрать дубли.

    Parameters:
        - settings: каталог выборки.

    Returns:
        Выборка с весом страты у каждой истории.

    Raises:
        FileNotFoundError: выборка ещё не собрана.
    """

    paths = sorted((settings.sample_dir / "parts").glob("*.parquet"))
    if not paths:
        msg = f"В {settings.sample_dir / 'parts'} нет выборки, сначала sample"
        raise FileNotFoundError(msg)
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["url_key"] = frame["url"].map(normalize_url)
    sample = (
        frame.drop_duplicates("story_id")
        .drop_duplicates("url_key")
        .drop(columns="url_key")
        .reset_index(drop=True)
    )
    path = settings.sample_dir / "stories.parquet"
    write_atomic(sample, path)
    logger.info(f"{path}: {len(sample)} из {len(frame)} историй")
    return sample


def gold(sample: pd.DataFrame, size: int, seed: int = 0) -> pd.DataFrame:
    """
    Отобрать gold-набор для ручной проверки, поровну по языкам.

    Языки, где историй меньше доли, берутся целиком; их недобор не
    перераспределяется.

    Parameters:
        - sample: выборка из `merge`;
        - size: желаемый размер набора;
        - seed: зерно случайного отбора.

    Returns:
        Строки `sample`, отсортированные по языку.
    """

    languages = sample["language"].fillna("unknown")
    share = -(-size // languages.nunique())
    picked = [
        group.sample(n=min(share, len(group)), random_state=seed).index
        for _, group in sample.groupby(languages)
    ]
    gold_set = sample.loc[np.concatenate(picked)]
    return gold_set.sort_values(["language", "region", "year"]).reset_index(drop=True)
