import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news.classifier import (
    OilClassifier,
    evaluate,
    recall_at_precision,
    thresholds,
    training_frame,
)
from news.settings import ClassifierSettings


def make_frame(size: int = 200) -> pd.DataFrame:
    """
    Таблица из `training_frame`: половина нефтяных, два языка, 2020-2026.

    Parameters:
        - size: число историй.

    Returns:
        Таблица с метками и весами.
    """

    rng = np.random.default_rng(0)
    target = np.arange(size) % 2
    return pd.DataFrame(
        {
            "story_id": [str(i) for i in range(size)],
            "language": np.where(np.arange(size) % 4 < 2, "en", "ar"),
            "year": rng.integers(2020, 2027, size),
            "weight": 1.0,
            "relevance": np.where(target == 1, "core", "none"),
            "target": target,
        },
    )


def test_training_frame_prefers_review_and_drops_unlabeled() -> None:
    inputs = pd.DataFrame(
        {
            "story_id": ["a", "b", "c"],
            "message": [
                json.dumps({"title": "Oil up", "lead": "Brent rose."}),
                json.dumps({"title": "Olive oil", "lead": None}),
                json.dumps({"title": "Gas", "lead": None}),
            ],
        },
    )
    sample = pd.DataFrame(
        {
            "story_id": ["a", "b", "c"],
            "region": "us",
            "year": 2025,
            "language": ["en", None, "en"],
            "stratum": "candidate",
            "weight": 1.0,
        },
    )
    primary = pd.DataFrame(
        {"story_id": ["a", "b", "c"], "relevance": ["related", "none", None]},
    )
    review = pd.DataFrame({"story_id": ["a"], "relevance": ["core"]})

    frame = training_frame(inputs, sample, primary, review).set_index("story_id")

    assert list(frame.index) == ["a", "b"]
    assert frame.loc["a", "relevance"] == "core"
    assert frame.loc["a", "title_lead"] == "Oil up\nBrent rose."
    assert frame.loc["b", "title_lead"] == "Olive oil"
    assert frame.loc["b", "language"] == "unknown"
    assert frame["target"].tolist() == [1, 0]


def test_recall_at_precision() -> None:
    target = np.array([1, 1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])

    assert recall_at_precision(target, scores, 1.0) == pytest.approx(2 / 3)
    assert recall_at_precision(target, scores, 0.75) == 1.0


def test_evaluate_separable_features_score_high() -> None:
    frame = make_frame()
    rng = np.random.default_rng(1)
    features = rng.normal(size=(len(frame), 8))
    features[:, 0] += frame["target"].to_numpy() * 4

    metrics = evaluate(
        features,
        frame,
        ClassifierSettings(_env_file=None, min_language_positives=5),
    )

    assert metrics["pr_auc"] > 0.95
    assert metrics["languages_in_macro"] == 2
    assert metrics["pr_auc_temporal"] > 0.9


def test_thresholds_keep_core_recall() -> None:
    settings = ClassifierSettings(_env_file=None, target_core_recall=0.9)
    target = np.array([1] * 10 + [0] * 10)
    core = target.copy()
    scores = np.concatenate([np.linspace(0.3, 0.9, 10), np.linspace(0.0, 0.5, 10)])

    cuts = thresholds(target, core, scores, settings)

    kept = (scores[core == 1] >= cuts["drop"]).mean()
    assert kept >= 0.9
    assert cuts["keep"] >= cuts["drop"]


class FakeEncoder:
    """Эмбеддинг - длина текста и единица."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str], **_) -> np.ndarray:
        """
        Векторы текстов.

        Returns:
            Матрица (число текстов, 2).
        """

        self.texts += texts
        return np.array([[len(text), 1.0] for text in texts])


def test_oil_classifier_applies_prefix_and_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = {
        "embedder": "fake",
        "prefix": "q: ",
        "max_length": 8,
        "coef": [1.0, -5.0],
        "intercept": 0.0,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    encoder = FakeEncoder()
    monkeypatch.setattr("news.classifier.load_encoder", lambda *_: encoder)

    scores = OilClassifier(tmp_path).predict(["ab", "abcdefg"])

    assert encoder.texts == ["q: ab", "q: abcdefg"]
    # 5 - 5 = 0 -> 0.5; 10 - 5 = 5 -> sigmoid(5)
    assert scores == pytest.approx([0.5, 1 / (1 + np.exp(-5))])
