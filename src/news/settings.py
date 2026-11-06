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

    index_dir: Path = Path("data/raw/news/mediacloud")
    stories_path: Path = Path("data/interim/news/stories.parquet")
    fulltext_dir: Path = Path("data/raw/news/fulltext")
    dataset_path: Path = Path("data/processed/news/ru_oil_news.parquet")
