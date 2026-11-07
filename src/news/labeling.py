import asyncio
import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from news.fulltext import junk_mask, read_articles
from news.regions import Region
from news.settings import LabelingSettings, NewsSettings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).with_name("prompts")


class Label(BaseModel):
    """Ответ модели по одной новости."""

    relevance: Literal["core", "related", "none"]
    commodity: list[Literal["oil", "products", "gas", "petrochemicals", "macro"]]
    topics: list[
        Literal[
            "production",
            "refining",
            "transport_logistics",
            "prices_market",
            "inventories",
            "sanctions_policy",
            "opec_plus",
            "outages_disruptions",
            "deals_corporate",
            "macro",
        ]
    ] = Field(max_length=3)
    entities: list[str] = Field(max_length=8)
    country: str | None
    confidence: float = Field(ge=0, le=1)


@dataclass
class Answer:
    """Результат одного запроса к модели: разметка или причина её отсутствия."""

    story_id: str
    model: str
    relevance: str | None = None
    commodity: list[str] | None = None
    topics: list[str] | None = None
    entities: list[str] | None = None
    country: str | None = None
    confidence: float | None = None
    raw: str | None = None
    error: str | None = None
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0
    labeled_at: str = ""


class BlankAnswerError(Exception):
    """Модель вернула пустой ответ: в JSON-режиме так выглядит невалидный JSON."""


def system_prompt() -> str:
    """
    Системный промпт разметки: инструкции и справочник.

    Returns:
        Текст, одинаковый для всех запросов: на нём работает кэш DeepSeek.
    """

    instructions = (PROMPTS_DIR / "oil_relevance.md").read_text(encoding="utf-8")
    reference = (PROMPTS_DIR / "oil_reference.md").read_text(encoding="utf-8")
    return f"{instructions}\n\n{reference}"


def build_inputs(
    sample: pd.DataFrame,
    regions: dict[str, Region],
    news: NewsSettings,
    labeling: LabelingSettings,
) -> pd.DataFrame:
    """
    Подготовить вход модели: заголовок, лид, дата, язык и страна источника.

    Лид - начало скачанного текста; тексты-шаблоны и короткие не берутся.

    Parameters:
        - sample: выборка из `sampling.merge`;
        - regions: регионы по имени, из них страна источника;
        - news: каталог текстов выборки;
        - labeling: длина лида.

    Returns:
        Таблица `story_id` и `message` - JSON для модели.
    """

    articles = read_articles(news)
    good = articles[
        (articles["text_source"] != "none") & ~junk_mask(articles, news.junk_repeats)
    ]
    leads = good.set_index("story_id")["text"].str.slice(0, labeling.lead_chars)
    messages = [
        json.dumps(
            {
                "date": f"{row.publish_date:%Y-%m-%d}"
                if pd.notna(row.publish_date)
                else None,
                "language": row.language,
                "source_country": regions[row.region].countries[0]
                if row.region in regions
                else None,
                "title": row.title,
                "lead": leads.get(row.story_id),
            },
            ensure_ascii=False,
        )
        for row in sample.itertuples()
    ]
    return pd.DataFrame({"story_id": sample["story_id"], "message": messages})


def is_transient(error: BaseException) -> bool:
    """
    Ошибка, после которой запрос к DeepSeek стоит повторить.

    Parameters:
        - error: исключение запроса.

    Returns:
        True для сети, 429, 5xx и пустого ответа.
    """

    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.TransportError | BlankAnswerError)


class DeepSeekLabeler:
    """Размечает новости одной моделью DeepSeek."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: LabelingSettings,
        model: str,
    ) -> None:
        self.client = client
        self.settings = settings
        self.model = model
        self.system = system_prompt()

    async def request(self, message: str) -> tuple[str, dict]:
        """
        Один запрос к chat completions с повторами.

        Пустой или неполный ответ повторяется: так API отдаёт и невалидный JSON
        модели, и сбой на своей стороне.

        Parameters:
            - message: JSON новости.

        Returns:
            Текст ответа модели и учёт токенов.
        """

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": message},
            ],
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(is_transient),
            wait=wait_exponential(multiplier=2, max=60),
            stop=stop_after_attempt(self.settings.attempts),
            reraise=True,
        ):
            with attempt:
                response = await self.client.post("/chat/completions", json=body)
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as error:
                    raise BlankAnswerError from error
                choices = data.get("choices") or [{}]
                content = (choices[0].get("message") or {}).get("content") or ""
                if not content.strip():
                    raise BlankAnswerError
        return content, data.get("usage") or {}

    async def label(self, story_id: str, message: str) -> Answer:
        """
        Разметить одну новость.

        Parameters:
            - story_id: id истории;
            - message: JSON новости.

        Returns:
            Разметка или ошибка; исключения не выходят наружу.
        """

        answer = Answer(
            story_id=story_id,
            model=self.model,
            labeled_at=datetime.now(UTC).isoformat(),
        )
        try:
            answer.raw, usage = await self.request(message)
        except Exception as error:  # noqa: BLE001
            # Исключение отменило бы всю пачку уже оплаченных ответов
            answer.error = f"{type(error).__name__}: {error}"[:500]
            return answer
        answer.cache_hit_tokens = usage.get("prompt_cache_hit_tokens") or 0
        answer.cache_miss_tokens = usage.get("prompt_cache_miss_tokens") or 0
        answer.output_tokens = usage.get("completion_tokens") or 0
        try:
            label = Label.model_validate_json(answer.raw)
        except ValidationError as error:
            answer.error = f"invalid: {error.errors()[0]['msg']}"[:500]
            return answer
        for name, value in label.model_dump().items():
            setattr(answer, name, value)
        return answer


def read_labels(settings: NewsSettings, model: str) -> pd.DataFrame:
    """
    Прочитать ответы модели, по каждой истории - последний.

    Parameters:
        - settings: каталог разметки;
        - model: имя модели DeepSeek.

    Returns:
        Таблица ответов; пустая, если модель ещё ничего не разметила.
    """

    paths = sorted((settings.labels_dir / model).glob("part-*.parquet"))
    if not paths:
        return pd.DataFrame(columns=[field.name for field in fields(Answer)])
    labels = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return (
        labels.sort_values("labeled_at", kind="stable")
        .drop_duplicates("story_id", keep="last")
        .reset_index(drop=True)
    )


async def run(
    inputs: pd.DataFrame,
    news: NewsSettings,
    labeling: LabelingSettings,
    model: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """
    Разметить новости пачками в part-файлы модели.

    Уже размеченные без ошибки пропускаются, поэтому прогон можно прерывать.

    Parameters:
        - inputs: таблица из `build_inputs`;
        - news: каталог разметки;
        - labeling: ключ, лимиты и размер пачки;
        - model: имя модели DeepSeek;
        - transport: транспорт httpx вместо сети.
    """

    done = read_labels(news, model)
    done = done[done["error"].isna()]
    todo = inputs[~inputs["story_id"].isin(done["story_id"])]
    logger.info(f"{model}: {len(todo)} из {len(inputs)} к разметке")
    out_dir = news.labels_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    part = len(list(out_dir.glob("part-*.parquet")))

    async with httpx.AsyncClient(
        base_url=labeling.base_url,
        headers={
            "Authorization": f"Bearer {labeling.deepseek_api_key.get_secret_value()}",
        },
        timeout=labeling.request_timeout,
        limits=httpx.Limits(max_connections=labeling.concurrency),
        transport=transport,
    ) as client:
        labeler = DeepSeekLabeler(client, labeling, model)
        for start in range(0, len(todo), labeling.batch_size):
            batch = todo.iloc[start : start + labeling.batch_size]
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(labeler.label(row.story_id, row.message))
                    for row in batch.itertuples()
                ]
            answers = pd.DataFrame([asdict(task.result()) for task in tasks])
            path = out_dir / f"part-{part:05d}.parquet"
            tmp = path.with_suffix(".tmp")
            answers.to_parquet(tmp, index=False)
            tmp.replace(path)
            part += 1
            errors = int(answers["error"].notna().sum())
            hit = answers["cache_hit_tokens"].sum() / max(
                1,
                answers[["cache_hit_tokens", "cache_miss_tokens"]].to_numpy().sum(),
            )
            left = len(todo) - start - len(batch)
            logger.info(
                f"{path.name}: {answers['relevance'].value_counts().to_dict()}, "
                f"ошибок {errors}, кэш {hit:.0%}, осталось {left}",
            )


def review_ids(
    sample: pd.DataFrame,
    gold: pd.DataFrame,
    primary: pd.DataFrame,
    settings: LabelingSettings,
) -> pd.Series:
    """
    Истории для второго мнения: gold, неуверенные и случайная контрольная доля.

    Parameters:
        - sample: выборка;
        - gold: gold-набор;
        - primary: ответы основной модели;
        - settings: порог уверенности и контрольная доля.

    Returns:
        Уникальные `story_id`.
    """

    unsure = primary.loc[
        primary["confidence"].isna() | (primary["confidence"] < settings.review_below),
        "story_id",
    ]
    control = sample["story_id"].sample(frac=settings.review_share, random_state=0)
    ids = pd.concat([gold["story_id"], unsure, control], ignore_index=True)
    return ids.drop_duplicates().reset_index(drop=True)
