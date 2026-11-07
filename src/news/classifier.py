import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from news.settings import ClassifierSettings

logger = logging.getLogger(__name__)


def load_encoder(model_name: str, max_length: int) -> SentenceTransformer:
    """
    Загрузить модель эмбеддингов.

    transformers 5 ищет только safetensors; у части моделей (BAAI/bge-m3)
    веса лежат лишь в `pytorch_model.bin`, их грузит второй попыткой.

    Parameters:
        - model_name: модель sentence-transformers;
        - max_length: токенов текста на эмбеддинг.

    Returns:
        Модель с ограничением длины.
    """

    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except OSError:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"use_safetensors": False},
        )
    model.max_seq_length = max_length
    return model


def training_frame(
    inputs: pd.DataFrame,
    sample: pd.DataFrame,
    primary: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    """
    Собрать обучающую таблицу: тексты, язык, год и итоговая метка.

    Метка второго мнения заменяет основную; истории без валидной метки
    отбрасываются.

    Parameters:
        - inputs: вход модели разметки (`labeling.build_inputs`);
        - sample: выборка с регионом, годом, стратой и весом;
        - primary: ответы основной модели;
        - review: ответы модели второго мнения.

    Returns:
        Таблица с `title`, `title_lead`, `relevance`, `target`.
    """

    messages = pd.DataFrame([json.loads(message) for message in inputs["message"]])
    texts = pd.DataFrame(
        {
            "story_id": inputs["story_id"].to_numpy(),
            "title": messages["title"].fillna("").to_numpy(),
            "lead": messages["lead"].to_numpy(),
        },
    )
    texts["title_lead"] = [
        f"{title}\n{lead}" if isinstance(lead, str) else title
        for title, lead in zip(texts["title"], texts["lead"], strict=True)
    ]
    labels = pd.concat(
        [primary.assign(source="primary"), review.assign(source="review")],
        ignore_index=True,
    )
    labels = labels[labels["relevance"].notna()]
    labels = labels.sort_values("source").drop_duplicates("story_id", keep="last")
    frame = (
        sample[["story_id", "region", "year", "language", "stratum", "weight"]]
        .merge(texts, on="story_id")
        .merge(labels[["story_id", "relevance", "source"]], on="story_id")
    )
    frame["language"] = frame["language"].fillna("unknown")
    frame["target"] = (frame["relevance"] != "none").astype(int)
    return frame.reset_index(drop=True)


def embed(
    model_name: str,
    frame: pd.DataFrame,
    field: str,
    settings: ClassifierSettings,
) -> tuple[np.ndarray, float]:
    """
    Эмбеддинги текстов одной моделью, с кэшем на диске.

    Parameters:
        - model_name: модель sentence-transformers;
        - frame: таблица с `story_id` и полем текста;
        - field: `title` или `title_lead`;
        - settings: префиксы, размер батча, каталог кэша.

    Returns:
        Матрица в порядке строк `frame` и секунд на 1000 текстов.
    """

    slug = model_name.replace("/", "__")
    path = settings.models_dir / "embeddings" / f"{slug}-{field}.parquet"
    speed_path = path.with_suffix(".json")
    cached = (
        pd.read_parquet(path)
        if path.exists()
        else pd.DataFrame(columns=["story_id", "vector"])
    )
    todo = frame[~frame["story_id"].isin(cached["story_id"])]
    seconds_per_1000 = (
        json.loads(speed_path.read_text())["seconds_per_1000"]
        if speed_path.exists()
        else float("nan")
    )
    if len(todo):
        model = load_encoder(model_name, settings.max_length)
        texts = [settings.embedders[model_name] + text for text in todo[field]]
        start = time.monotonic()
        vectors = model.encode(
            texts,
            batch_size=settings.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        seconds_per_1000 = (time.monotonic() - start) / len(texts) * 1000
        fresh = pd.DataFrame({"story_id": todo["story_id"], "vector": list(vectors)})
        cached = pd.concat([cached, fresh], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path, index=False)
        speed_path.write_text(json.dumps({"seconds_per_1000": seconds_per_1000}))
    vectors = cached.set_index("story_id").loc[frame["story_id"], "vector"]
    return np.stack(vectors.to_numpy()), seconds_per_1000


def recall_at_precision(
    target: np.ndarray, scores: np.ndarray, precision: float
) -> float:
    """
    Наибольший recall при precision не ниже заданной.

    Parameters:
        - target: 0/1;
        - scores: вероятности класса 1;
        - precision: нижняя граница precision.

    Returns:
        Recall; 0, если precision недостижима.
    """

    precisions, recalls, _ = precision_recall_curve(target, scores)
    reachable = recalls[precisions >= precision]
    return float(reachable.max()) if len(reachable) else 0.0


def out_of_fold(
    features: np.ndarray,
    target: np.ndarray,
    settings: ClassifierSettings,
) -> np.ndarray:
    """
    Вероятности кросс-валидации: каждая история оценена моделью без неё.

    Parameters:
        - features: эмбеддинги;
        - target: 0/1;
        - settings: число фолдов.

    Returns:
        Вероятности класса 1.
    """

    scores = np.zeros(len(target))
    folds = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=0)
    for train, test in folds.split(features, target):
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(features[train], target[train])
        scores[test] = model.predict_proba(features[test])[:, 1]
    return scores


def evaluate(
    features: np.ndarray,
    frame: pd.DataFrame,
    settings: ClassifierSettings,
) -> dict[str, float]:
    """
    Метрики одной модели эмбеддингов на кросс-валидации и временном сплите.

    Parameters:
        - features: эмбеддинги в порядке `frame`;
        - frame: таблица из `training_frame`;
        - settings: фолды, границы precision и временного сплита.

    Returns:
        PR-AUC, recall при целевой precision, macro PR-AUC по языкам,
        PR-AUC взвешенная по стратам и на временном сплите.
    """

    target = frame["target"].to_numpy()
    scores = out_of_fold(features, target, settings)
    per_language = [
        average_precision_score(group["target"], scores[group.index])
        for _, group in frame.groupby("language")
        if group["target"].sum() >= settings.min_language_positives
        and group["target"].sum() < len(group)
    ]
    train = (frame["year"] < settings.test_from_year).to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(features[train], target[train])
    temporal = model.predict_proba(features[~train])[:, 1]
    return {
        "pr_auc": average_precision_score(target, scores),
        "recall_at_precision": recall_at_precision(
            target,
            scores,
            settings.target_precision,
        ),
        "pr_auc_macro_language": float(np.mean(per_language)),
        "languages_in_macro": len(per_language),
        "pr_auc_weighted": average_precision_score(
            target,
            scores,
            sample_weight=frame["weight"].fillna(1),
        ),
        "pr_auc_temporal": average_precision_score(target[~train], temporal),
    }


def benchmark(
    frame: pd.DataFrame,
    field: str,
    settings: ClassifierSettings,
) -> pd.DataFrame:
    """
    Сравнить модели эмбеддингов: замороженный энкодер + логистическая регрессия.

    Parameters:
        - frame: таблица из `training_frame`;
        - field: `title` или `title_lead`;
        - settings: модели и параметры оценки.

    Returns:
        Метрики и скорость по моделям, лучшие сверху.
    """

    rows = []
    for model_name in settings.embedders:
        try:
            features, seconds = embed(model_name, frame, field, settings)
        except OSError as error:
            logger.warning(f"{model_name} не загрузилась, пропущена: {error}")
            continue
        metrics = evaluate(features, frame, settings)
        rows.append({"model": model_name, **metrics, "seconds_per_1000": seconds})
        logger.info(f"{model_name}: {metrics}")
    table = pd.DataFrame(rows).sort_values("pr_auc_macro_language", ascending=False)
    path = settings.models_dir / f"benchmark-{field}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table.reset_index(drop=True)


def thresholds(
    target: np.ndarray,
    core: np.ndarray,
    scores: np.ndarray,
    settings: ClassifierSettings,
) -> dict[str, float]:
    """
    Пороги отсева и уверенного «да» по вероятностям кросс-валидации.

    Ниже `drop` история отбрасывается: порог держит recall ядра не ниже
    целевого. Выше `keep` precision не ниже целевой. Между ними - пограничная
    полоса для модели «заголовок + лид».

    Parameters:
        - target: 0/1 «ядро или связано»;
        - core: 0/1 «ядро»;
        - scores: вероятности кросс-валидации;
        - settings: целевые recall ядра и precision.

    Returns:
        Пороги `drop` и `keep`.
    """

    core_scores = np.sort(scores[core == 1])
    drop_index = int(np.floor((1 - settings.target_core_recall) * len(core_scores)))
    drop = float(core_scores[drop_index]) if len(core_scores) else 0.5
    precisions, _, cuts = precision_recall_curve(target, scores)
    above = cuts[precisions[:-1] >= settings.target_precision]
    keep = float(above.min()) if len(above) else 1.0
    return {"drop": drop, "keep": max(keep, drop)}


def fit(
    model_name: str,
    frame: pd.DataFrame,
    field: str,
    settings: ClassifierSettings,
) -> Path:
    """
    Обучить итоговый классификатор на всей разметке и сохранить его.

    Parameters:
        - model_name: модель эмбеддингов;
        - frame: таблица из `training_frame`;
        - field: `title` или `title_lead`;
        - settings: параметры оценки и каталог моделей.

    Returns:
        Каталог модели: `model.joblib` и `meta.json`.
    """

    features, seconds = embed(model_name, frame, field, settings)
    target = frame["target"].to_numpy()
    scores = out_of_fold(features, target, settings)
    core = (frame["relevance"] == "core").to_numpy().astype(int)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(features, target)

    out_dir = settings.models_dir / field
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    meta = {
        "embedder": model_name,
        "prefix": settings.embedders[model_name],
        "field": field,
        "max_length": settings.max_length,
        "thresholds": thresholds(target, core, scores, settings),
        "metrics": evaluate(features, frame, settings),
        "seconds_per_1000": seconds,
        "trained_on": len(frame),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    logger.info(f"{out_dir}: {meta['thresholds']}, {meta['metrics']}")
    return out_dir


def predict(texts: list[str], model_dir: Path, batch_size: int = 64) -> np.ndarray:
    """
    Вероятность «ядро или связано» для текстов.

    Parameters:
        - texts: заголовки или «заголовок + лид», как при обучении;
        - model_dir: каталог из `fit`;
        - batch_size: текстов в батче энкодера.

    Returns:
        Вероятности в порядке `texts`.
    """

    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    encoder = load_encoder(meta["embedder"], meta["max_length"])
    vectors = encoder.encode(
        [meta["prefix"] + text for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    model: LogisticRegression = joblib.load(model_dir / "model.joblib")
    return model.predict_proba(vectors)[:, 1]
