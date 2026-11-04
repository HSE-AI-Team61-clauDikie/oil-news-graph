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
    query: str = Field(
        default=(
            "language:ru AND article_title:(нефт* OR НПЗ OR ОПЕК* OR OPEC "
            "OR Brent OR WTI OR баррел* OR нефтепровод* OR нефтепродукт* "
            "OR Роснефт* OR Лукойл* OR Транснефт* OR Газпромнефт*)"
        ),
        description="Lucene-запрос Media Cloud",
    )
    collection_ids: list[int] = Field(
        default_factory=list,
        description="Коллекции Media Cloud; пусто - все источники",
    )
    start_date: date = date(2015, 2, 19)
    end_date: date = date(2026, 10, 31)
    page_size: int = 1000

    concurrency: int = Field(default=32, description="Одновременных загрузок")
    per_host_concurrency: int = Field(
        default=2,
        description="Одновременных загрузок с одного сайта",
    )
    request_timeout: float = 20.0
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
    batch_size: int = Field(default=500, description="Статей в одном part-файле")

    index_dir: Path = Path("data/raw/news/mediacloud")
    stories_path: Path = Path("data/interim/news/stories.parquet")
    fulltext_dir: Path = Path("data/raw/news/fulltext")
    dataset_path: Path = Path("data/processed/news/ru_oil_news.parquet")
