import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news import __main__ as cli
from news.fulltext import Article
from news.regions import Region
from news.settings import NewsSettings
from news.storage import write_atomic


def make_region(name: str) -> Region:
    """Регион с тестовой коллекцией."""

    return Region(
        name=name,
        countries=["US"],
        collection_ids=[1],
        languages=["en"],
        terms=["oil"],
    )


def stories(ids: list[str]) -> pd.DataFrame:
    """Индекс с разными адресами и датами публикации."""

    return pd.DataFrame(
        {
            "story_id": ids,
            "title": [f"oil {key}" for key in ids],
            "url": [f"https://example.org/{key}" for key in ids],
            "publish_date": pd.to_datetime(["2026-10-01"] * len(ids)),
            "media_name": "example.org",
        }
    )


class FakeModel:
    """Оставить все заголовки; записать вызовы энкодера."""

    def __init__(self) -> None:
        self.meta = {"thresholds": {"drop": 0.5, "keep": 0.8}}
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.full(len(texts), 0.9)


def test_filter_partial_index_then_only_new_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    base = NewsSettings(_env_file=None, filter_device="cpu", filter_batch_size=16)
    us, absent = make_region("us"), make_region("sa")
    settings = us.settings(base)
    write_atomic(stories(["a"]), settings.index_dir / "stories_2026-09.parquet")
    model = FakeModel()
    options = []

    def load(*_, **kwargs) -> FakeModel:
        options.append(kwargs)
        return model

    monkeypatch.setattr("news.classifier.OilClassifier", load)
    cli.filter_regions(base, [us, absent], {})
    assert pd.read_parquet(settings.candidates_path)["story_id"].tolist() == ["a"]
    assert not absent.settings(base).stories_path.exists()

    write_atomic(stories(["b"]), settings.index_dir / "stories_2026-10.parquet")
    cli.filter_regions(base, [us], {})
    assert model.calls == [["oil a"], ["oil b"]]
    assert options == [{"device": "cpu", "batch_size": 16}] * 2
    assert pd.read_parquet(settings.stories_path)["story_id"].tolist() == ["a", "b"]
    assert pd.read_parquet(settings.candidates_path)["story_id"].tolist() == ["a", "b"]

    def unexpected(*_, **__) -> None:
        pytest.fail("completed filter must not load the encoder")

    monkeypatch.setattr("news.classifier.OilClassifier", unexpected)
    candidate_stamp = settings.candidates_path.stat().st_mtime_ns
    cli.filter_regions(base, [us], {})
    assert settings.candidates_path.stat().st_mtime_ns == candidate_stamp


def save_candidates(settings: NewsSettings) -> None:
    """Сохранить кандидата, у которого уже оценён заголовок."""

    write_atomic(
        stories(["a"]).assign(probability=0.9, decision="keep"),
        settings.candidates_path,
    )


def save_text(settings: NewsSettings) -> None:
    """Сохранить успешную попытку загрузки одной статьи."""

    article = Article(
        "a",
        "https://example.org/a",
        text="Brent prices rose.",
        text_source="live",
    )
    write_atomic(
        pd.DataFrame([asdict(article)]),
        settings.fulltext_dir / "part-00000.parquet",
    )


def test_assemble_stage_uses_saved_texts_without_api_or_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    base = NewsSettings(_env_file=None)
    us, empty = make_region("us"), make_region("sa")
    settings = us.settings(base)
    save_candidates(settings)
    save_text(settings)
    save_candidates(empty.settings(base))

    def unexpected(*_, **__) -> None:
        pytest.fail("assemble must not access Media Cloud or load an encoder")

    monkeypatch.setattr(cli, "SearchApi", unexpected)
    monkeypatch.setattr("news.classifier.OilClassifier", unexpected)
    cli.run_regions(
        base,
        {"us": us, "sa": empty},
        argparse.Namespace(only=None, start=None, end=None, stage="assemble"),
    )

    result = pd.read_parquet(settings.dataset_path)
    assert result["story_id"].tolist() == ["a"]
    assert result["probability"].tolist() == [0.9]
    assert result["text"].tolist() == ["Brent prices rose."]
    assert not empty.settings(base).dataset_path.exists()
    dataset_stamp = settings.dataset_path.stat().st_mtime_ns
    cli.assemble_regions(base, [us], {})
    assert settings.dataset_path.stat().st_mtime_ns == dataset_stamp

    save_text(settings)
    cli.assemble_regions(base, [us], {})
    assert settings.dataset_path.stat().st_mtime_ns > dataset_stamp


def test_fulltext_failure_still_assembles_saved_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    base = NewsSettings(_env_file=None)
    region = make_region("us")
    settings = region.settings(base)
    save_candidates(settings)

    async def interrupted(kept: NewsSettings, **_) -> None:
        save_text(kept)
        msg = "connection lost after saving a part"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli.fulltext, "run", interrupted)
    with pytest.raises(RuntimeError, match="connection lost"):
        cli.fulltext_regions(base, [region], {})

    assert pd.read_parquet(settings.dataset_path)["story_id"].tolist() == ["a"]
