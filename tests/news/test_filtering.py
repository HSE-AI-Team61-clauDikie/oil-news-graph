from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news.filtering import candidates, decide, filter_index
from news.settings import NewsSettings


class FakeModel:
    """Вероятность по слову в заголовке: oil - 0.9, gas - 0.6, иначе 0.1."""

    def __init__(self) -> None:
        self.meta = {"thresholds": {"drop": 0.5, "keep": 0.8}}
        self.calls = 0

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Вероятности заголовков.

        Returns:
            Массив вероятностей.
        """

        self.calls += 1
        return np.array(
            [0.9 if "oil" in text else 0.6 if "gas" in text else 0.1 for text in texts],
        )


@pytest.fixture
def settings(tmp_path: Path) -> NewsSettings:
    """Индекс региона из двух месяцев и склеенный индекс."""

    settings = NewsSettings(
        _env_file=None,
        index_dir=tmp_path / "index",
        filtered_dir=tmp_path / "filtered",
        stories_path=tmp_path / "stories.parquet",
        candidates_path=tmp_path / "candidates.parquet",
    )
    settings.index_dir.mkdir()
    months = {
        "2025-01": ["oil up", "football"],
        "2025-02": ["gas deal", None, "oil down"],
    }
    frames = []
    for month, titles in months.items():
        frame = pd.DataFrame(
            {
                "story_id": [f"{month}-{i}" for i in range(len(titles))],
                "title": titles,
                "url": [f"https://x.com/{month}/{i}" for i in range(len(titles))],
            },
        )
        frame.to_parquet(settings.index_dir / f"stories_{month}.parquet")
        frames.append(frame)
    pd.concat(frames).to_parquet(settings.stories_path)
    return settings


def test_decide_by_thresholds() -> None:
    scores = np.array([0.1, 0.5, 0.79, 0.8])

    decisions = decide(scores, {"drop": 0.5, "keep": 0.8})

    assert decisions.tolist() == ["drop", "borderline", "borderline", "keep"]


def test_filter_index_skips_done_months(settings: NewsSettings) -> None:
    model = FakeModel()

    filter_index(settings, model)
    filter_index(settings, model)

    assert model.calls == 2
    february = pd.read_parquet(settings.filtered_dir / "stories_2025-02.parquet")
    assert february["decision"].tolist() == ["borderline", "drop", "keep"]


def test_candidates_drop_rejected(settings: NewsSettings) -> None:
    filter_index(settings, FakeModel())

    kept = candidates(settings)

    assert sorted(kept["title"]) == ["gas deal", "oil down", "oil up"]
    assert {"probability", "decision", "url"} <= set(kept.columns)
    assert settings.candidates_path.exists()


def test_filter_and_candidates_need_inputs(tmp_path: Path) -> None:
    settings = NewsSettings(
        _env_file=None,
        index_dir=tmp_path / "index",
        filtered_dir=tmp_path / "filtered",
    )

    with pytest.raises(FileNotFoundError):
        filter_index(settings, FakeModel())
    with pytest.raises(FileNotFoundError):
        candidates(settings)
