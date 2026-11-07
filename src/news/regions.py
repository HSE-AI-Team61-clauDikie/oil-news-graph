import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from news.settings import NewsSettings


class Region(BaseModel):
    """
    Регион сбора: страны, коллекции Media Cloud и запрос к ним.

    `terms` - recall-запрос по заголовку: с `full_stream` индекс качается
    целиком, но выборка для разметки всё равно делится по нему на кандидатов
    и фон.
    """

    name: str
    countries: list[str] = Field(min_length=1)
    collection_ids: list[int] = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    terms: list[str] = Field(min_length=1)
    full_stream: bool = False

    def language_filter(self) -> str:
        """
        Фильтр Media Cloud по языкам региона.

        Returns:
            Lucene-условие на поле `language`.
        """

        return f"language:({' OR '.join(self.languages)})"

    def recall_query(self) -> str:
        """
        Истории на языках региона, в заголовке которых есть хотя бы один термин.

        Returns:
            Lucene-запрос кандидатов.
        """

        terms = " OR ".join(self.terms)
        return f"{self.language_filter()} AND article_title:({terms})"

    def background_query(self) -> str:
        """
        Истории на языках региона, которые recall-запрос не находит.

        Returns:
            Lucene-запрос дополнения к кандидатам.
        """

        terms = " OR ".join(self.terms)
        return f"{self.language_filter()} AND NOT article_title:({terms})"

    def query(self) -> str:
        """
        Запрос выкачки индекса.

        Returns:
            Весь поток на языках региона при `full_stream`, иначе кандидаты.
        """

        return self.language_filter() if self.full_stream else self.recall_query()

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
                "filtered_dir": Path("data/interim/news") / self.name / "filtered",
                "candidates_path": Path("data/interim/news")
                / self.name
                / "candidates.parquet",
                "fulltext_dir": Path("data/raw/news") / self.name / "fulltext",
                "dataset_path": (
                    Path("data/processed/news") / self.name / "oil_news.parquet"
                ),
            },
        )


def load_regions(path: Path | None = None) -> dict[str, Region]:
    """
    Прочитать описание регионов.

    Термины региона - это `terms.common`, термины его языков из `terms` и
    `extra_terms` самого региона, без повторов.

    Parameters:
        - path: TOML с таблицами `terms` и `regions`; по умолчанию
            `regions.toml` рядом.

    Returns:
        Регионы по имени.
    """

    path = path or Path(__file__).with_name("regions.toml")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    vocabulary: dict[str, list[str]] = config["terms"]
    regions: dict[str, Region] = {}
    for name, fields in config["regions"].items():
        terms = vocabulary.get("common", []).copy()
        for language in fields["languages"]:
            terms += vocabulary.get(language, [])
        terms += fields.pop("extra_terms", [])
        regions[name] = Region(name=name, terms=list(dict.fromkeys(terms)), **fields)
    return regions
