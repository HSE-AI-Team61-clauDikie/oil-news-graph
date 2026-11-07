from pathlib import Path

import pandas as pd
import pytest

from news.progress import status
from news.regions import Region
from news.settings import NewsSettings


def make_region(name: str) -> Region:
    """
    Регион с одной коллекцией.

    Parameters:
        - name: имя региона.

    Returns:
        Регион для тестов.
    """

    return Region(
        name=name,
        countries=["US"],
        collection_ids=[1],
        languages=["en"],
        terms=["oil"],
    )


def test_status_counts_stages_from_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = make_region("us").settings(NewsSettings(_env_file=None))
    frame = pd.DataFrame({"story_id": ["a", "b", "c"]})
    for path in (
        settings.index_dir / "stories_2026-09_2026-10.parquet",
        settings.stories_path,
        settings.filtered_dir / "stories_2026-09_2026-10.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    frame.head(2).to_parquet(settings.candidates_path)

    table = status(NewsSettings(_env_file=None), [make_region("us"), make_region("sa")])

    us, sa = table.to_dict("records")
    assert us["months"] == "2/142"
    assert us["index"] == 3
    assert us["filtered"] == "1/1"
    assert us["candidates"] == 2
    assert us["texts"] == ""
    assert us["updated"] != ""
    assert sa == {
        "region": "sa",
        "months": "0/142",
        "index": "",
        "filtered": "",
        "candidates": "",
        "texts": "",
        "dataset": "",
        "updated": "",
    }
