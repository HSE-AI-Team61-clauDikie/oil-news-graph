import logging

import numpy as np
import pandas as pd

from news.classifier import OilClassifier
from news.settings import NewsSettings
from news.storage import write_atomic

logger = logging.getLogger(__name__)


def decide(scores: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """
    Решение по вероятности: отбросить, пограничное или оставить.

    Parameters:
        - scores: вероятности «ядро или связано»;
        - thresholds: пороги `drop` и `keep` модели.

    Returns:
        Массив `drop`, `borderline`, `keep`.
    """

    return np.select(
        [scores < thresholds["drop"], scores < thresholds["keep"]],
        ["drop", "borderline"],
        default="keep",
    )


def filter_index(settings: NewsSettings, model: OilClassifier) -> None:
    """
    Оценить заголовки индекса региона помесячно.

    Готовые месяцы пропускаются: прогон можно прерывать.

    Parameters:
        - settings: каталоги индекса и оценок;
        - model: классификатор заголовков.

    Raises:
        FileNotFoundError: индекс ещё не скачан.
    """

    paths = sorted(settings.index_dir.glob("stories_*.parquet"))
    if not paths:
        msg = f"В {settings.index_dir} нет файлов индекса, сначала download"
        raise FileNotFoundError(msg)
    for path in paths:
        out = settings.filtered_dir / path.name
        if out.exists():
            continue
        stories = pd.read_parquet(path, columns=["story_id", "title"])
        scores = model.predict(stories["title"].fillna("").tolist())
        frame = pd.DataFrame(
            {
                "story_id": stories["story_id"],
                "probability": scores,
                "decision": decide(scores, model.meta["thresholds"]),
            },
        )
        write_atomic(frame, out)
        logger.info(f"{out.name}: {frame['decision'].value_counts().to_dict()}")


def candidates(settings: NewsSettings) -> pd.DataFrame:
    """
    Собрать истории, прошедшие фильтр, из склеенного индекса.

    Parameters:
        - settings: склеенный индекс, каталог оценок и путь результата.

    Returns:
        Истории индекса с `probability` и `decision`, кроме `drop`.

    Raises:
        FileNotFoundError: заголовки ещё не оценены.
    """

    paths = sorted(settings.filtered_dir.glob("stories_*.parquet"))
    if not paths:
        msg = f"В {settings.filtered_dir} нет оценок, сначала filter"
        raise FileNotFoundError(msg)
    scores = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    ).drop_duplicates("story_id")
    stories = pd.read_parquet(settings.stories_path)
    kept = stories.merge(scores, on="story_id", how="inner")
    kept = kept[kept["decision"] != "drop"].reset_index(drop=True)
    write_atomic(kept, settings.candidates_path)
    logger.info(
        f"{settings.candidates_path}: {len(kept)} из {len(stories)} историй, "
        f"{kept['decision'].value_counts().to_dict()}",
    )
    return kept
