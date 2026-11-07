from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from news.regions import Region, load_regions
from news.settings import NewsSettings


def make_region(*, full_stream: bool = False) -> Region:
    """
    Регион США с двумя терминами.

    Parameters:
        - full_stream: качать ли поток целиком.

    Returns:
        Регион для тестов.
    """

    return Region(
        name="us",
        countries=["US"],
        collection_ids=[34412234],
        languages=["en", "es"],
        terms=["oil", '"natural gas"'],
        full_stream=full_stream,
    )


def test_recall_and_background_split_by_title_terms() -> None:
    region = make_region()

    assert region.recall_query() == (
        'language:(en OR es) AND article_title:(oil OR "natural gas")'
    )
    assert region.background_query() == (
        'language:(en OR es) AND NOT article_title:(oil OR "natural gas")'
    )


def test_query_is_recall_unless_full_stream() -> None:
    assert make_region().query() == make_region().recall_query()
    assert make_region(full_stream=True).query() == "language:(en OR es)"


def test_settings_use_region_paths_and_skip_wayback() -> None:
    settings = make_region(full_stream=True).settings(NewsSettings(_env_file=None))

    assert settings.collection_ids == [34412234]
    assert settings.start_date == date(2015, 1, 1)
    assert settings.candidates_path == Path("data/interim/news/us/candidates.parquet")
    assert settings.query == "language:(en OR es)"
    assert settings.use_wayback is False
    assert settings.title_exclude is None
    assert settings.accept_language == "en,es"
    assert settings.index_dir == Path("data/raw/news/us/mediacloud")
    assert settings.dataset_path == Path("data/processed/news/us/oil_news.parquet")


def test_region_needs_collections_and_terms() -> None:
    with pytest.raises(ValidationError):
        Region(
            name="x",
            countries=["X"],
            collection_ids=[],
            languages=["en"],
            terms=["oil"],
        )
    with pytest.raises(ValidationError):
        Region(name="x", countries=["X"], collection_ids=[1], languages=["en"])


def test_load_regions_joins_common_language_and_extra_terms(tmp_path: Path) -> None:
    path = tmp_path / "regions.toml"
    path.write_text(
        "[terms]\n"
        'common = ["OPEC", "Brent"]\n'
        'es = ["petróleo", "OPEC"]\n'
        'en = ["oil"]\n'
        "[regions.mx]\n"
        'countries = ["MX"]\n'
        "collection_ids = [34412427]\n"
        'languages = ["es"]\n'
        'extra_terms = ["Pemex"]\n',
        encoding="utf-8",
    )

    regions = load_regions(path)

    assert list(regions) == ["mx"]
    assert regions["mx"].terms == ["OPEC", "Brent", "petróleo", "Pemex"]
    assert regions["mx"].full_stream is False


def test_repository_regions_load() -> None:
    regions = load_regions()

    assert regions
    assert all(region.terms for region in regions.values())
