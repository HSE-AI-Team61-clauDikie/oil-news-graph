import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from news.settings import NewsSettings


class Region(BaseModel):
    """
    Регион сбора: страны, коллекции Media Cloud и запрос к ним.

    Пустой `terms` - качается весь поток коллекций на языках `languages`;
    иначе только истории, где встречается хотя бы один термин.
    """

    name: str
    countries: list[str] = Field(min_length=1)
    collection_ids: list[int] = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    terms: list[str] = Field(default_factory=list)

    def query(self) -> str:
        """
        Lucene-запрос Media Cloud для региона.

        Returns:
            Фильтр по языкам и, если заданы, по терминам.
        """

        query = f"language:({' OR '.join(self.languages)})"
        if self.terms:
            query += f" AND ({' OR '.join(self.terms)})"
        return query

    def settings(self, base: NewsSettings) -> NewsSettings:
        """
        Настройки пайплайна для региона.

        Parameters:
            - base: общие настройки: ключ, период, лимиты.

        Returns:
            Копия `base` с запросом, коллекциями и путями региона, без Wayback
            и без русского фильтра заголовков.
        """

        return base.model_copy(
            update={
                "query": self.query(),
                "collection_ids": self.collection_ids,
                "title_exclude": None,
                "use_wayback": False,
                "accept_language": ",".join(self.languages),
                "index_dir": Path("data/raw/news") / self.name / "mediacloud",
                "stories_path": Path("data/interim/news")
                / self.name
                / "stories.parquet",
                "fulltext_dir": Path("data/raw/news") / self.name / "fulltext",
                "dataset_path": (
                    Path("data/processed/news") / self.name / "oil_news.parquet"
                ),
            },
        )


def load_regions(path: Path | None = None) -> dict[str, Region]:
    """
    Прочитать описание регионов.

    Parameters:
        - path: TOML, одна таблица на регион; по умолчанию `regions.toml` рядом.

    Returns:
        Регионы по имени.
    """

    path = path or Path(__file__).with_name("regions.toml")
    tables = tomllib.loads(path.read_text(encoding="utf-8"))
    return {name: Region(name=name, **fields) for name, fields in tables.items()}
