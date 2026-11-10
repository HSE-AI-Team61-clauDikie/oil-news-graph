import argparse
import asyncio
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from mediacloud.api import DirectoryApi, SearchApi

from news import dataset, fulltext, labeling, mediacloud, progress, sampling
from news.regions import Region, load_regions
from news.settings import LabelingSettings, NewsSettings

logger = logging.getLogger("news")


def parse_args() -> argparse.Namespace:
    """
    Разобрать команду и её параметры.

    Returns:
        Аргументы командной строки.
    """

    period = argparse.ArgumentParser(add_help=False)
    period.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD")
    period.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD")

    parser = argparse.ArgumentParser(
        prog="python -m news",
        description="Новости о нефти: Media Cloud и полный текст",
    )
    parser.add_argument(
        "--region",
        help="регион из regions.toml; без него - российские новости",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="fulltext по выборке на разметку, а не по индексу",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "count",
        parents=[period],
        help="число историй и запросов по месяцам",
    )
    commands.add_parser("download", parents=[period], help="скачать индекс")
    commands.add_parser("merge", help="склеить индекс без дублей")
    collections = commands.add_parser("collections", help="найти коллекции")
    collections.add_argument("--name", required=True, help="часть названия")
    texts = commands.add_parser("fulltext", help="скачать тексты статей")
    texts.add_argument(
        "--retry-failed",
        action="store_true",
        help="повторить статьи без текста",
    )
    texts.add_argument(
        "--newest-first",
        action="store_true",
        help="идти от свежих статей к старым",
    )
    commands.add_parser("assemble", help="собрать итоговый датасет")
    commands.add_parser(
        "filter",
        help="оценить заголовки индекса региона классификатором (нужен --region)",
    )
    run = commands.add_parser(
        "run",
        parents=[period],
        help="прогнать регионы по порядку regions.toml",
    )
    run.add_argument(
        "--stage",
        choices=["download", "filter", "fulltext", "assemble", "process"],
        required=True,
        help=(
            "download: индекс; filter: заголовки; fulltext: тексты; "
            "assemble: сохранённые тексты; process: все этапы обработки"
        ),
    )
    run.add_argument("--only", help="регионы через запятую, например us,sa")
    commands.add_parser(
        "status",
        help="таблица прогона по регионам: скачано, отфильтровано, собрано",
    )
    commands.add_parser(
        "sample",
        parents=[period],
        help="выборка регион × год на разметку; с --region - один регион",
    )
    commands.add_parser("gold", help="отобрать gold-набор из выборки")
    label = commands.add_parser("label", help="разметить выборку через DeepSeek")
    label.add_argument(
        "--review",
        action="store_true",
        help="второе мнение: gold, неуверенные и контрольная доля",
    )
    for name, text in (
        ("benchmark", "сравнить модели эмбеддингов на разметке"),
        ("train", "обучить итоговый классификатор"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument(
            "--field",
            choices=["title", "title_lead"],
            default="title",
            help="вход: заголовок или заголовок + лид",
        )
        if name == "train":
            command.add_argument("--embedder", required=True, help="модель эмбеддингов")
    return parser.parse_args()


def api_key(settings: NewsSettings) -> str:
    """
    Ключ Media Cloud из настроек.

    Raises:
        SystemExit: ключ не задан.
    """

    key = settings.mediacloud_api_key.get_secret_value()
    if not key:
        msg = (
            "Нет MEDIACLOUD_API_KEY в .env: ключ выдаётся после регистрации "
            "на https://search.mediacloud.org"
        )
        raise SystemExit(msg)
    return key


def log_table(frame: pd.DataFrame) -> None:
    """Вывести таблицу в лог целиком."""

    logger.info(f"\n{frame.to_string(index=False)}")


def collect_sample(
    settings: NewsSettings,
    regions: dict[str, Region],
    name: str | None,
) -> None:
    """
    Собрать выборку на разметку и склеить её.

    Parameters:
        - settings: период, размеры выборки и каталог;
        - regions: все регионы;
        - name: только этот регион, если задан.
    """

    chosen = [regions[name]] if name else list(regions.values())
    sampling.collect(SearchApi(api_key(settings)), chosen, settings)
    sample = sampling.merge(settings)
    log_table(
        sample.groupby(["region", "stratum"])
        .size()
        .unstack(fill_value=0)
        .reset_index(),
    )


def select_gold(settings: NewsSettings) -> None:
    """
    Отобрать gold-набор и сохранить рядом с выборкой.

    Parameters:
        - settings: каталог выборки и размер набора.
    """

    sample = pd.read_parquet(settings.sample_dir / "stories.parquet")
    picked = sampling.gold(sample, settings.gold_size)
    path = settings.sample_dir / "gold.parquet"
    sampling.write_atomic(picked, path)
    logger.info(f"{path}: {len(picked)} историй, языков {picked['language'].nunique()}")


def label_sample(
    settings: NewsSettings,
    regions: dict[str, Region],
    *,
    review: bool,
) -> None:
    """
    Разметить выборку основной моделью или дать второе мнение.

    Parameters:
        - settings: каталоги выборки, текстов и разметки;
        - regions: страны источников;
        - review: второе мнение вместо основной разметки.

    Raises:
        SystemExit: ключ DeepSeek не задан.
    """

    labeling_settings = LabelingSettings()
    if not labeling_settings.deepseek_api_key.get_secret_value():
        msg = "Нет DEEPSEEK_API_KEY в .env"
        raise SystemExit(msg)
    sample = pd.read_parquet(settings.sample_dir / "stories.parquet")
    inputs = labeling.build_inputs(
        sample,
        regions,
        sampling.sample_settings(settings),
        labeling_settings,
    )
    model = labeling_settings.model
    if review:
        gold = pd.read_parquet(settings.sample_dir / "gold.parquet")
        primary = labeling.read_labels(settings, labeling_settings.model)
        ids = labeling.review_ids(sample, gold, primary, labeling_settings)
        inputs = inputs[inputs["story_id"].isin(ids)]
        model = labeling_settings.review_model
    asyncio.run(labeling.run(inputs, settings, labeling_settings, model))


def classifier_command(
    settings: NewsSettings,
    regions: dict[str, Region],
    args: argparse.Namespace,
) -> None:
    """
    Сравнить модели эмбеддингов или обучить итоговый классификатор.

    Parameters:
        - settings: каталоги выборки, текстов и разметки;
        - regions: страны источников;
        - args: команда, поле текста и модель эмбеддингов.
    """

    # torch и sentence-transformers - в группе ml: без неё сбор и разметка работают
    from news import classifier
    from news.settings import ClassifierSettings

    labeling_settings = LabelingSettings()
    sample = pd.read_parquet(settings.sample_dir / "stories.parquet")
    inputs = labeling.build_inputs(
        sample,
        regions,
        sampling.sample_settings(settings),
        labeling_settings,
    )
    frame = classifier.training_frame(
        inputs,
        sample,
        labeling.read_labels(settings, labeling_settings.model),
        labeling.read_labels(settings, labeling_settings.review_model),
    )
    logger.info(f"Обучающих историй: {len(frame)}, нефтяных {frame['target'].sum()}")
    classifier_settings = ClassifierSettings()
    if args.command == "benchmark":
        log_table(classifier.benchmark(frame, args.field, classifier_settings))
    else:
        classifier.fit(args.embedder, frame, args.field, classifier_settings)


def filter_region(settings: NewsSettings, region: str | None) -> None:
    """
    Оценить заголовки индекса региона и собрать прошедшие фильтр.

    Parameters:
        - settings: пути региона и модель заголовков;
        - region: имя региона.

    Raises:
        SystemExit: регион не задан.
    """

    if not region:
        msg = "filter работает по региону: python -m news --region <имя> filter"
        raise SystemExit(msg)
    # torch и sentence-transformers - в группе ml: без неё сбор и разметка работают
    from news import filtering
    from news.classifier import OilClassifier

    model = OilClassifier(
        settings.title_model_dir,
        device=settings.filter_device,
        batch_size=settings.filter_batch_size,
    )
    filtering.filter_index(settings, model)
    filtering.candidates(settings)


def download_regions(base: NewsSettings, regions: list[Region], period: dict) -> None:
    """
    Скачать и склеить индекс регионов по очереди.

    Исчерпанная квота Media Cloud останавливает прогон; повторный запуск
    продолжит с первого недокачанного месяца.

    Parameters:
        - base: общие настройки и ключ;
        - regions: регионы по порядку;
        - period: `start_date`, `end_date`, если заданы.
    """

    for region in regions:
        settings = region.settings(base).model_copy(update=period)
        logger.info(f"{region.name}: выкачка индекса")
        mediacloud.download(SearchApi(api_key(settings)), settings)
        mediacloud.merge(settings)


def filter_regions(base: NewsSettings, regions: list[Region], period: dict) -> None:
    """
    Склеить доступные окна индекса и оценить заголовки всех прибывших регионов.

    Неполные регионы тоже обрабатываются. После следующей загрузки новые окна
    попадут в склейку и фильтр, уже оценённые окна повторно не считаются.

    Parameters:
        - base: общие настройки и модель заголовков;
        - regions: регионы по порядку;
        - period: `start_date`, `end_date`, если заданы.
    """

    # torch и sentence-transformers - в группе ml: без неё сбор и разметка работают
    from news import filtering
    from news.classifier import OilClassifier

    model = None
    for region in regions:
        settings = region.settings(base).model_copy(update=period)
        paths = sorted(settings.index_dir.glob("stories_*.parquet"))
        if not paths:
            logger.info(f"{region.name}: нет скачанных окон индекса, пропущен")
            continue
        logger.info(f"{region.name}: фильтр доступных окон индекса")
        if not settings.stories_path.exists() or any(
            path.stat().st_mtime_ns > settings.stories_path.stat().st_mtime_ns
            for path in paths
        ):
            mediacloud.merge(settings)
        pending = [
            path for path in paths if not (settings.filtered_dir / path.name).exists()
        ]
        if pending:
            if model is None:
                model = OilClassifier(
                    base.title_model_dir,
                    device=base.filter_device,
                    batch_size=base.filter_batch_size,
                )
            filtering.filter_index(settings, model)
        scores = list(settings.filtered_dir.glob("stories_*.parquet"))
        if not settings.candidates_path.exists() or any(
            path.stat().st_mtime_ns > settings.candidates_path.stat().st_mtime_ns
            for path in [settings.stories_path, *scores]
        ):
            filtering.candidates(settings)


def assemble_regions(base: NewsSettings, regions: list[Region], period: dict) -> None:
    """Собрать датасеты из сохранённых текстов, без сети и загрузки энкодера."""

    for region in regions:
        settings = region.settings(base).model_copy(update=period)
        parts = list(settings.fulltext_dir.glob("part-*.parquet"))
        if not settings.candidates_path.exists() or not parts:
            continue
        if settings.dataset_path.exists() and all(
            path.stat().st_mtime_ns <= settings.dataset_path.stat().st_mtime_ns
            for path in [settings.candidates_path, *parts]
        ):
            continue
        logger.info(f"{region.name}: датасет из сохранённых текстов")
        kept = settings.model_copy(update={"stories_path": settings.candidates_path})
        dataset.assemble(kept)


def fulltext_regions(base: NewsSettings, regions: list[Region], period: dict) -> None:
    """Скачать тексты готовых кандидатов без классификации; собрать датасеты."""

    assemble_regions(base, regions, period)
    for region in regions:
        settings = region.settings(base).model_copy(update=period)
        if not settings.candidates_path.exists():
            continue
        logger.info(f"{region.name}: тексты и датасет")
        kept = settings.model_copy(update={"stories_path": settings.candidates_path})
        try:
            asyncio.run(
                fulltext.run(
                    kept,
                    newest_first=True,
                    max_articles=base.fulltext_region_limit,
                ),
            )
        finally:
            assemble_regions(base, [region], period)


def process_regions(base: NewsSettings, regions: list[Region], period: dict) -> None:
    """Собрать сохранённое, оценить все прибывшие регионы, затем скачать тексты."""

    assemble_regions(base, regions, period)
    filter_regions(base, regions, period)
    fulltext_regions(base, regions, period)


def run_regions(
    base: NewsSettings,
    regions: dict[str, Region],
    args: argparse.Namespace,
) -> None:
    """
    Прогнать регионы по порядку `regions.toml`: выкачку или обработку.

    Parameters:
        - base: общие настройки;
        - regions: все регионы;
        - args: стадия, список регионов и период.

    Raises:
        SystemExit: в `--only` неизвестный регион.
    """

    names = args.only.split(",") if args.only else list(regions)
    unknown = [name for name in names if name not in regions]
    if unknown:
        msg = f"Нет регионов {', '.join(unknown)}, есть: {', '.join(regions)}"
        raise SystemExit(msg)
    chosen = [regions[name] for name in names]
    period = {
        name: value
        for name, value in {"start_date": args.start, "end_date": args.end}.items()
        if value
    }
    stages = {
        "download": download_regions,
        "filter": filter_regions,
        "fulltext": fulltext_regions,
        "assemble": assemble_regions,
        "process": process_regions,
    }
    stages[args.stage](base, chosen, period)


def main() -> None:
    """Выполнить команду CLI."""

    # Копия лога в файл: прогон идёт сутками, терминал могут закрыть
    log_path = Path("data/logs/news.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    # httpx пишет в INFO каждый запрос, trafilatura - каждую страницу без
    # статьи: на сотнях тысяч статей это шум, причина и так попадает в error
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    args = parse_args()
    settings = NewsSettings()
    regions = load_regions()
    if args.region and args.region not in regions:
        msg = f"Нет региона {args.region}, есть: {', '.join(regions)}"
        raise SystemExit(msg)
    if args.sample:
        settings = sampling.sample_settings(settings)
    elif args.region and args.command not in {"sample", "run", "status"}:
        settings = regions[args.region].settings(settings)
        if args.command in {"fulltext", "assemble"}:
            # В регионе тексты качаются только для прошедших фильтр заголовков
            settings = settings.model_copy(
                update={"stories_path": settings.candidates_path},
            )
    period = {
        "start_date": getattr(args, "start", None),
        "end_date": getattr(args, "end", None),
    }
    settings = settings.model_copy(
        update={name: value for name, value in period.items() if value},
    )

    commands = {
        "count": lambda: log_table(
            mediacloud.count(SearchApi(api_key(settings)), settings),
        ),
        "download": lambda: mediacloud.download(SearchApi(api_key(settings)), settings),
        "merge": lambda: mediacloud.merge(settings),
        "collections": lambda: log_table(
            mediacloud.find_collections(DirectoryApi(api_key(settings)), args.name),
        ),
        "fulltext": lambda: asyncio.run(
            fulltext.run(
                settings,
                retry_failed=args.retry_failed,
                newest_first=args.newest_first,
            ),
        ),
        "assemble": lambda: dataset.assemble(settings),
        "sample": lambda: collect_sample(settings, regions, args.region),
        "gold": lambda: select_gold(settings),
        "label": lambda: label_sample(settings, regions, review=args.review),
        "benchmark": lambda: classifier_command(settings, regions, args),
        "train": lambda: classifier_command(settings, regions, args),
        "filter": lambda: filter_region(settings, args.region),
        "run": lambda: run_regions(settings, regions, args),
        "status": lambda: log_table(progress.status(settings, list(regions.values()))),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
