from pathlib import Path

import pytest
from pydantic import ValidationError

from news.regions import Region, load_regions
from news.settings import NewsSettings


def test_query_without_terms_takes_whole_stream() -> None:
    region = Region(
        name="saudi",
        countries=["Saudi Arabia"],
        collection_ids=[34412050],
        languages=["ar", "en"],
    )

    assert region.query() == "language:(ar OR en)"


def test_query_with_terms_filters_by_them() -> None:
    region = Region(
        name="us",
        countries=["United States"],
        collection_ids=[34412234],
        languages=["en"],
        terms=["oil", "crude", '"natural gas"'],
    )

    assert region.query() == 'language:(en) AND (oil OR crude OR "natural gas")'


def test_settings_use_region_paths_and_skip_wayback() -> None:
    region = Region(
        name="us",
        countries=["United States"],
        collection_ids=[34412234],
        languages=["en", "es"],
    )

    settings = region.settings(NewsSettings(_env_file=None))

    assert settings.collection_ids == [34412234]
    assert settings.query == "language:(en OR es)"
    assert settings.use_wayback is False
    assert settings.title_exclude is None
    assert settings.accept_language == "en,es"
    assert settings.index_dir == Path("data/raw/news/us/mediacloud")
    assert settings.dataset_path == Path("data/processed/news/us/oil_news.parquet")


def test_region_needs_collections() -> None:
    with pytest.raises(ValidationError):
        Region(name="x", countries=["X"], collection_ids=[], languages=["en"])


def test_load_regions_reads_toml(tmp_path: Path) -> None:
    path = tmp_path / "regions.toml"
    path.write_text(
        "[mexico]\n"
        'countries = ["Mexico"]\n'
        "collection_ids = [34412427]\n"
        'languages = ["es"]\n'
        'terms = ["petróleo", "Pemex"]\n',
        encoding="utf-8",
    )

    regions = load_regions(path)

    assert list(regions) == ["mexico"]
    assert regions["mexico"].query() == "language:(es) AND (petróleo OR Pemex)"
