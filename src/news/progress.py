from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from news.mediacloud import downloaded_months, month_windows
from news.regions import Region
from news.settings import NewsSettings


def rows(path: Path) -> int | None:
    """
    Число строк parquet-файла по метаданным, без чтения данных.

    Parameters:
        - path: файл.

    Returns:
        Число строк; None, если файла нет.
    """

    return pq.ParquetFile(path).metadata.num_rows if path.exists() else None


def region_status(settings: NewsSettings, name: str) -> dict:
    """
    Состояние прогона одного региона по файлам на диске.

    Parameters:
        - settings: пути региона;
        - name: имя региона.

    Returns:
        Строка таблицы `status`.
    """

    total = len(month_windows(settings.start_date, settings.end_date))
    index = sorted(settings.index_dir.glob("stories_*.parquet"))
    filtered = sorted(settings.filtered_dir.glob("stories_*.parquet"))
    texts = sorted(settings.fulltext_dir.glob("part-*.parquet"))
    singles = [settings.stories_path, settings.candidates_path, settings.dataset_path]
    files = [*index, *filtered, *texts, *(path for path in singles if path.exists())]
    updated = max((path.stat().st_mtime for path in files), default=None)
    # Локальное время машины: его сравнивают с часами на той же машине
    stamp = datetime.fromtimestamp(updated) if updated else None  # noqa: DTZ006
    return {
        "region": name,
        "months": f"{len(downloaded_months(settings.index_dir))}/{total}",
        "index": rows(settings.stories_path),
        "filtered": f"{len(filtered)}/{len(index)}" if index else None,
        "candidates": rows(settings.candidates_path),
        "texts": sum(rows(path) or 0 for path in texts) if texts else None,
        "dataset": rows(settings.dataset_path),
        "updated": f"{stamp:%Y-%m-%d %H:%M}" if stamp else None,
    }


def status(base: NewsSettings, regions: list[Region]) -> pd.DataFrame:
    """
    Таблица прогона по регионам: что скачано, отфильтровано и собрано.

    Parameters:
        - base: общие настройки;
        - regions: регионы по порядку.

    Returns:
        Строка на регион; пустая ячейка - этап ещё не начат.
    """

    # object: иначе колонки с пропусками станут float и 7921 выведется как 7921.0
    table = pd.DataFrame(
        [region_status(region.settings(base), region.name) for region in regions],
        dtype=object,
    )
    return table.fillna("")
