from pathlib import Path

import pandas as pd
import pytest

from news.dataset import assemble, resolve_dates, url_date
from news.settings import NewsSettings

TEXT = (
    "Цены на нефть марки Brent выросли после сообщений о сокращении добычи "
    "странами ОПЕК+. Аналитики ожидают снижения коммерческих запасов в США. "
) * 6


@pytest.fixture
def settings(tmp_path: Path) -> NewsSettings:
    return NewsSettings(
        _env_file=None,
        stories_path=tmp_path / "stories.parquet",
        fulltext_dir=tmp_path / "fulltext",
        dataset_path=tmp_path / "dataset.parquet",
        date_cluster_min=5,
        date_cluster_factor=5,
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://lenta.ru/news/2026/11/03/fall/", "2026-11-03"),
        ("http://www.ria.ru/economy/20150605/1068386708.html", "2015-06-05"),
        ("https://russian.rt.com/inotv/2022-04-12/Welt-v-ES", "2022-04-12"),
        ("https://www.rbc.ru/business/22/12/2016/585b6a2c9a79", "2016-12-22"),
        ("https://www.rbc.ru/rbcfreenews/5927a8fa9a7947aa5f9c9b63", None),
        ("https://www.interfax.ru/business/1126785", None),
        ("https://example.ru/2019/02/30/bad-day/", None),
    ],
)
def test_url_date(url: str, expected: str | None) -> None:
    result = url_date(url)

    assert result == (pd.Timestamp(expected) if expected else None)


def test_resolve_dates_fixes_import_clusters(settings: NewsSettings) -> None:
    # rbc.ru: обычный день - одна статья, а на 2016-12-22 Media Cloud свалил шесть
    usual = [
        {
            "media_name": "rbc.ru",
            "url": f"https://www.rbc.ru/rbcfreenews/usual{number}",
            "publish_date": pd.Timestamp("2016-11-01") + pd.Timedelta(days=number),
            "extracted_date": None,
        }
        for number in range(10)
    ]
    cluster = [
        {
            "media_name": "rbc.ru",
            "url": f"https://www.rbc.ru/rbcfreenews/import{number}",
            "publish_date": pd.Timestamp("2016-12-22"),
            "extracted_date": extracted,
        }
        for number, extracted in enumerate(
            [
                "2017-05-12",
                "2017-06-01",
                "2018-01-01",
                None,
                "2017-03-03",
                "2017-08-08",
            ],
        )
    ]
    dated_url = {
        "media_name": "lenta.ru",
        "url": "https://lenta.ru/news/2017/02/03/oil/",
        "publish_date": pd.Timestamp("2016-12-22"),
        "extracted_date": "2016-12-22",
    }
    dataset = pd.DataFrame([*usual, *cluster, dated_url])
    dataset["story_id"] = dataset.index.astype(str)

    dates = resolve_dates(dataset, settings)

    assert dates["date_source"].tolist() == [
        *["media_cloud"] * 10,
        "page",
        "page",
        "unreliable",
        "unreliable",
        "page",
        "page",
        "url",
    ]
    assert dates.loc[10, "date"] == pd.Timestamp("2017-05-12")
    assert dates.loc[12, "date"] == pd.Timestamp("2016-12-22")
    assert dates.loc[16, "date"] == pd.Timestamp("2017-02-03")


def article(story_id: str, url: str, text: str | None, **fields) -> dict:
    """
    Строка part-файла с текстом статьи.

    Parameters:
        - story_id: id истории;
        - url: адрес;
        - text: текст или None;
        - fields: остальные поля, перекрывают значения по умолчанию.

    Returns:
        Словарь полей статьи.
    """

    return {
        "story_id": story_id,
        "url": url,
        "final_url": url,
        "text": text,
        "text_source": "live" if text else "none",
        "extracted_date": None,
        "fetched_at": pd.Timestamp("2026-11-01", tz="UTC"),
    } | fields


def test_assemble_drops_noise_and_junk_and_marks_reprints(
    settings: NewsSettings,
) -> None:
    pd.DataFrame(
        {
            "story_id": ["tass", "region", "other", "empty", "homepage", "hockey"],
            "url": [
                "https://tass.ru/1",
                "https://r.ru/1",
                "https://o.ru/1",
                "https://e.ru/1",
                "http://www.vesti.ru/doc.html?id=1",
                "https://sports.ru/1",
            ],
            "publish_date": pd.to_datetime(
                [
                    "2026-10-01",
                    "2026-10-02",
                    "2026-10-02",
                    "2026-10-02",
                    "2015-08-06",
                    "2026-10-03",
                ],
            ),
            "media_name": ["tass.ru", "r.ru", "o.ru", "e.ru", "vesti.ru", "sports.ru"],
            "title": [
                "Нефть дорожает",
                "Нефть дорожает",
                "НПЗ остановлен",
                "Нефть",
                "Нефть",
                "«Нефтехимик» обыграл «Ладу»",
            ],
        },
    ).to_parquet(settings.stories_path)
    settings.fulltext_dir.mkdir()
    pd.DataFrame(
        [
            article("region", "https://r.ru/1", TEXT.upper()),
            article("tass", "https://tass.ru/1", TEXT, text_source="wayback"),
            article("other", "https://o.ru/1", "Другой текст про НПЗ"),
            article("empty", "https://e.ru/1", None),
            article(
                "homepage",
                "http://www.vesti.ru/doc.html?id=1",
                "Главные новости дня",
                final_url="https://www.vesti.ru/",
            ),
            article("hockey", "https://sports.ru/1", "Хоккейный матч"),
        ],
    ).to_parquet(settings.fulltext_dir / "part-00000.parquet")

    dataset = assemble(settings).set_index("story_id")

    assert sorted(dataset.index) == ["other", "region", "tass"]
    assert dataset.loc["region", "duplicate_of"] == "tass"
    assert pd.isna(dataset.loc["tass", "duplicate_of"])
    assert pd.isna(dataset.loc["other", "duplicate_of"])
    assert dataset.loc["tass", "url"] == "https://tass.ru/1"
    assert set(dataset["date_source"]) == {"media_cloud"}
