import logging
import re

import numpy as np
import pandas as pd

from news.fulltext import junk_mask, read_articles, text_fingerprint
from news.settings import NewsSettings

logger = logging.getLogger(__name__)


def url_date(url: str) -> pd.Timestamp | None:
    """
    Дата публикации из адреса статьи.

    Понимает `/2017/05/12/`, `/20170512/`, `/2017-05-12/` и `/12/05/2017/`.

    Parameters:
        - url: адрес статьи.

    Returns:
        Дата или None, если в адресе её нет.
    """

    if match := re.search(
        r"(20[12]\d)[/_-]?(0[1-9]|1[0-2])[/_-]?(0[1-9]|[12]\d|3[01])(?!\d)",
        url,
    ):
        year, month, day = match.groups()
    elif match := re.search(r"/(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(20[12]\d)/", url):
        day, month, year = match.groups()
    else:
        return None
    stamp = pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")
    return None if pd.isna(stamp) else stamp


def date_clusters(dataset: pd.DataFrame, settings: NewsSettings) -> pd.Series:
    """
    Отметить статьи из кластеров дат Media Cloud.

    Кластер - день, на который у источника пришлось в `date_cluster_factor` раз
    больше статей, чем в его обычный день: так выглядят даты импорта, а не
    публикации (rbc.ru 2016-12-22, profile.ru 2026-01-01).

    Parameters:
        - dataset: статьи с `media_name` и `publish_date`;
        - settings: пороги кластера.

    Returns:
        Булева маска по строкам `dataset`.
    """

    day = dataset["publish_date"].dt.normalize()
    daily = dataset.groupby([dataset["media_name"], day]).size()
    typical = daily.groupby(level=0).median()
    threshold = typical.reindex(daily.index.get_level_values(0)).to_numpy()
    clusters = daily[
        (daily >= settings.date_cluster_min)
        & (daily >= threshold * settings.date_cluster_factor)
    ]
    keys = pd.MultiIndex.from_arrays([dataset["media_name"], day])
    return pd.Series(keys.isin(clusters.index), index=dataset.index)


def resolve_dates(dataset: pd.DataFrame, settings: NewsSettings) -> pd.DataFrame:
    """
    Выбрать дату публикации из адреса, Media Cloud или страницы.

    Порядок: дата из адреса; если её нет, а дата Media Cloud попала в кластер,
    - дата со страницы, кроме 1 января (так trafilatura отдаёт год без дня);
    иначе - дата Media Cloud.

    Parameters:
        - dataset: статьи с `url`, `publish_date`, `extracted_date`;
        - settings: пороги кластера дат.

    Returns:
        Копия `dataset` с колонками `date` и `date_source`
        (url, page, media_cloud или unreliable).
    """

    from_url = pd.to_datetime(dataset["url"].map(url_date))
    from_page = pd.to_datetime(dataset["extracted_date"], errors="coerce")
    has_url = from_url.notna()
    in_cluster = date_clusters(dataset, settings) & ~has_url
    page_ok = (
        in_cluster
        & from_page.notna()
        & ~((from_page.dt.month == 1) & (from_page.dt.day == 1))
    )
    dataset = dataset.copy()
    dataset["date"] = (
        dataset["publish_date"].where(~page_ok, from_page).where(~has_url, from_url)
    )
    dataset["date_source"] = np.select(
        [has_url, page_ok, in_cluster],
        ["url", "page", "unreliable"],
        default="media_cloud",
    )
    return dataset


def assemble(settings: NewsSettings) -> pd.DataFrame:
    """
    Собрать датасет: индекс Media Cloud и тексты статей.

    Шаблоны сайта и заголовки-шум (`title_exclude`) отбрасываются, дата
    уточняется (`resolve_dates`). Перепечатки одного текста остаются, у всех,
    кроме самой ранней, заполнено `duplicate_of`.

    Parameters:
        - settings: пути индекса, part-файлов и результата, фильтры.

    Returns:
        Статьи с текстом.
    """

    stories = pd.read_parquet(settings.stories_path)
    noise = stories["title"].str.contains(
        settings.title_exclude,
        case=False,
        regex=True,
        na=False,
    )
    stories = stories[~noise]
    articles = read_articles(settings)
    junk = junk_mask(articles, settings.junk_repeats)
    articles = articles[(articles["text_source"] != "none") & ~junk]
    articles = articles.drop(columns="url")
    dataset = resolve_dates(
        stories.merge(articles, on="story_id", how="inner", validate="one_to_one"),
        settings,
    )
    dataset = dataset.sort_values(["date", "story_id"], kind="stable").reset_index(
        drop=True,
    )
    dataset["text_hash"] = dataset["text"].map(text_fingerprint)
    first = dataset.groupby("text_hash")["story_id"].transform("first")
    dataset["duplicate_of"] = first.where(first != dataset["story_id"])
    settings.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(settings.dataset_path, index=False)
    sources = dataset["date_source"].value_counts().to_dict()
    logger.info(
        f"{settings.dataset_path}: {len(dataset)} статей, "
        f"шум по заголовку {noise.sum()}, "
        f"перепечаток {dataset['duplicate_of'].notna().sum()}, даты: {sources}",
    )
    return dataset
