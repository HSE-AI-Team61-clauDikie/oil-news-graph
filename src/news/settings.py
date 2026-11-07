from datetime import date
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class NewsSettings(BaseSettings):
    """
    Настройки сбора русскоязычных новостей о нефти.

    Значения берутся из окружения и `.env` в корне репозитория.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mediacloud_api_key: SecretStr = SecretStr("")
    # ОПЕК без *: «ОПЕК» не склоняется, а ОПЕК* ловит «опекуна» и «опеку»
    query: str = Field(
        default=(
            "language:ru AND article_title:(нефт* OR НПЗ OR ОПЕК OR OPEC "
            "OR Brent OR WTI OR баррел* OR нефтепровод* OR нефтепродукт* "
            "OR Роснефт* OR Лукойл* OR Транснефт* OR Газпромнефт*) "
            "AND NOT article_title:(Нефтехимик* OR Нефтчи)"
        ),
        description="Lucene-запрос Media Cloud",
    )
    # Без коллекций или источников Media Cloud отвечает 422
    collection_ids: list[int] = Field(
        default=[34412232, 38380780, 38380781],
        min_length=1,
        description="Коллекции Media Cloud; по умолчанию - российские",
    )
    start_date: date = date(2015, 2, 19)
    end_date: date = date(2026, 10, 31)
    page_size: int = 1000

    concurrency: int = Field(default=32, description="Одновременных загрузок")
    per_host_concurrency: int = Field(
        default=4,
        description="Одновременных загрузок с одного сайта",
    )
    request_timeout: float = 20.0
    accept_language: str = "ru,en;q=0.8"
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )
    min_text_chars: int = Field(
        default=300,
        description="Короче - страница не дала статью, идём в Wayback",
    )
    wayback_interval: float = Field(
        default=4.0,
        description="Секунд между запросами к archive.org: чаще он отвечает 429",
    )
    junk_repeats: int = Field(
        default=3,
        description="Столько одинаковых текстов с одного сайта - шаблон, не статья",
    )
    batch_size: int = Field(default=500, description="Статей в одном part-файле")
    wayback_batch_size: int = Field(
        default=50,
        description="Статей в part-файле этапа Wayback: он медленный, сохраняем чаще",
    )

    use_wayback: bool = Field(
        default=True,
        description="Искать в Wayback статьи, не давшие текста на сайте",
    )
    title_exclude: str | None = Field(
        default=r"нефтехимик|нефтчи|опек(?!\b|\+)",
        description="Регулярка заголовков-шума: спортклубы, опека и опекуны",
    )
    date_cluster_min: int = Field(
        default=30,
        description="Меньше статей источника за день - не кластер дат",
    )
    date_cluster_factor: int = Field(
        default=20,
        description="Во сколько раз день выше медианы источника, чтобы быть кластером",
    )

    sample_candidates: int = Field(
        default=25,
        description="Историй recall-запроса на год региона",
    )
    sample_background: int = Field(
        default=10,
        description="Историй вне recall-запроса на год региона: оценка его пропусков",
    )
    sample_background_days: int = Field(
        default=6,
        description="Случайных дней, из которых берётся фон региона",
    )
    sample_min_days: int = Field(
        default=3,
        description="Не меньше стольких дней на год в выборке кандидатов",
    )
    sample_max_days: int = Field(
        default=15,
        description="Не больше стольких дней на год в выборке кандидатов",
    )
    sample_query_bytes: int = Field(
        default=4600,
        description="Длина запроса после URL-кодирования, дальше Media Cloud - 400",
    )
    gold_size: int = Field(default=600, description="Историй в gold-наборе")
    sample_dir: Path = Path("data/interim/news/sample")

    index_dir: Path = Path("data/raw/news/mediacloud")
    labels_dir: Path = Path("data/interim/news/labels")
    stories_path: Path = Path("data/interim/news/stories.parquet")
    fulltext_dir: Path = Path("data/raw/news/fulltext")
    dataset_path: Path = Path("data/processed/news/ru_oil_news.parquet")


class LabelingSettings(BaseSettings):
    """
    Настройки разметки новостей через DeepSeek.

    Значения берутся из окружения и `.env` в корне репозитория.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.deepseek.com"
    model: str = Field(default="deepseek-flash", description="Основная разметка")
    review_model: str = Field(
        default="deepseek-v4-pro",
        description="Второе мнение: gold, неуверенные и контрольная доля",
    )
    review_below: float = Field(
        default=0.6,
        description="Уверенность основной модели, ниже которой нужен review",
    )
    review_share: float = Field(
        default=0.05,
        description="Случайная доля выборки на контрольный review",
    )
    concurrency: int = Field(default=32, description="Одновременных запросов")
    request_timeout: float = 120.0
    max_tokens: int = 300
    attempts: int = Field(default=5, description="Попыток на запрос")
    lead_chars: int = Field(default=500, description="Символов текста в лиде")
    batch_size: int = Field(default=500, description="Ответов в одном part-файле")


class ClassifierSettings(BaseSettings):
    """Настройки классификатора нефтяных новостей."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedders: dict[str, str] = Field(
        default={
            "intfloat/multilingual-e5-base": "query: ",
            "intfloat/multilingual-e5-large": "query: ",
            "BAAI/bge-m3": "",
            "sentence-transformers/LaBSE": "",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "",
        },
        description="Модели эмбеддингов и префикс текста, которого они ждут",
    )
    batch_size: int = 64
    max_length: int = Field(default=256, description="Токенов текста на эмбеддинг")
    folds: int = 5
    test_from_year: int = Field(
        default=2024,
        description="Временной сплит: обучение до этого года, тест с него",
    )
    min_language_positives: int = Field(
        default=10,
        description="Меньше положительных - язык не входит в macro-среднее",
    )
    target_precision: float = 0.9
    target_core_recall: float = Field(
        default=0.95,
        description="Recall ядра, под который подбирается порог отсева",
    )
    models_dir: Path = Path("data/models/news")
