import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from tenacity import wait_none

from news.labeling import (
    DeepSeekLabeler,
    build_inputs,
    read_labels,
    review_ids,
    run,
    system_prompt,
)
from news.regions import Region
from news.settings import LabelingSettings, NewsSettings

GOOD = {
    "relevance": "core",
    "commodity": ["oil"],
    "topics": ["inventories"],
    "entities": ["EIA"],
    "country": "US",
    "confidence": 0.95,
}


def completion(content: str) -> dict:
    """
    Ответ chat completions с одним сообщением.

    Parameters:
        - content: текст ответа модели.

    Returns:
        Тело ответа API.
    """

    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_cache_hit_tokens": 100,
            "prompt_cache_miss_tokens": 10,
            "completion_tokens": 40,
        },
    }


def make_client(handler) -> httpx.AsyncClient:
    """
    Клиент DeepSeek на подменённом транспорте.

    Parameters:
        - handler: функция запрос -> ответ.

    Returns:
        Асинхронный клиент.
    """

    return httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture
def labeling_settings() -> LabelingSettings:
    """Настройки без пауз между повторами."""

    return LabelingSettings(_env_file=None, attempts=4, batch_size=2)


def test_system_prompt_joins_instructions_and_reference() -> None:
    prompt = system_prompt()

    assert prompt.startswith("# Oil and gas news relevance labeling")
    assert "\n# Reference" in prompt
    assert "json" in prompt.lower()


async def test_label_parses_answer_and_usage(
    labeling_settings: LabelingSettings,
) -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(json.dumps(GOOD)))

    async with make_client(handler) as client:
        labeler = DeepSeekLabeler(client, labeling_settings, "deepseek-flash")
        answer = await labeler.label("s1", '{"title": "x"}')

    assert answer.relevance == "core"
    assert answer.entities == ["EIA"]
    assert answer.error is None
    assert answer.cache_hit_tokens == 100
    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert bodies[0]["response_format"] == {"type": "json_object"}


async def test_label_retries_blank_and_server_errors(
    labeling_settings: LabelingSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("news.labeling.wait_exponential", lambda **_: wait_none())
    replies = iter(
        [
            httpx.Response(503),
            httpx.Response(200, json=completion("   ")),
            httpx.Response(200, json={"choices": [{"message": None}], "usage": None}),
            httpx.Response(200, json=completion(json.dumps(GOOD))),
        ],
    )

    async with make_client(lambda _: next(replies)) as client:
        labeler = DeepSeekLabeler(client, labeling_settings, "deepseek-flash")
        answer = await labeler.label("s1", "{}")

    assert answer.relevance == "core"


async def test_label_reports_invalid_schema(
    labeling_settings: LabelingSettings,
) -> None:
    bad = {**GOOD, "relevance": "maybe"}

    async with make_client(
        lambda _: httpx.Response(200, json=completion(json.dumps(bad))),
    ) as client:
        labeler = DeepSeekLabeler(client, labeling_settings, "deepseek-flash")
        answer = await labeler.label("s1", "{}")

    assert answer.relevance is None
    assert answer.error.startswith("invalid")


async def test_run_writes_parts_and_skips_done(
    tmp_path: Path,
    labeling_settings: LabelingSettings,
) -> None:
    news = NewsSettings(_env_file=None, labels_dir=tmp_path / "labels")
    inputs = pd.DataFrame({"story_id": ["a", "b", "c"], "message": ["{}"] * 3})
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, json=completion(json.dumps(GOOD)))

    transport = httpx.MockTransport(handler)

    await run(inputs, news, labeling_settings, "deepseek-flash", transport)
    await run(inputs, news, labeling_settings, "deepseek-flash", transport)

    labels = read_labels(news, "deepseek-flash")
    assert sorted(labels["story_id"]) == ["a", "b", "c"]
    assert len(requests) == 3
    assert len(list((tmp_path / "labels" / "deepseek-flash").glob("part-*"))) == 2


def test_build_inputs_adds_lead_and_source_country(tmp_path: Path) -> None:
    news = NewsSettings(_env_file=None, fulltext_dir=tmp_path / "fulltext")
    news.fulltext_dir.mkdir()
    pd.DataFrame(
        {
            "story_id": ["a"],
            "url": ["https://example.com/a"],
            "final_url": ["https://example.com/a"],
            "text": ["Crude stocks fell. " * 50],
            "text_source": ["live"],
            "error": [None],
            "fetched_at": ["2026-01-01"],
        },
    ).to_parquet(news.fulltext_dir / "part-00000.parquet")
    sample = pd.DataFrame(
        {
            "story_id": ["a", "b"],
            "region": ["us", "us"],
            "publish_date": pd.to_datetime(["2025-07-23", None]),
            "language": ["en", "en"],
            "title": ["Crude stocks fall", "No text"],
        },
    )
    regions = {
        "us": Region(
            name="us",
            countries=["US"],
            collection_ids=[1],
            languages=["en"],
            terms=["oil"],
        ),
    }

    inputs = build_inputs(sample, regions, news, LabelingSettings(_env_file=None))

    first, second = (json.loads(message) for message in inputs["message"])
    assert first["date"] == "2025-07-23"
    assert first["source_country"] == "US"
    assert len(first["lead"]) == 500
    assert second["lead"] is None
    assert second["date"] is None


def test_review_ids_take_gold_unsure_and_control() -> None:
    sample = pd.DataFrame({"story_id": [str(i) for i in range(100)]})
    gold_set = pd.DataFrame({"story_id": ["0", "1"]})
    primary = pd.DataFrame(
        {
            "story_id": ["2", "3", "4"],
            "confidence": [0.4, 0.9, None],
        },
    )

    ids = review_ids(sample, gold_set, primary, LabelingSettings(_env_file=None))

    assert {"0", "1", "2", "4"} <= set(ids)
    assert ids.is_unique
    assert len(ids) <= 4 + 5
