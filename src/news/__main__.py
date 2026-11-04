import argparse
import asyncio
import logging
from datetime import date

import pandas as pd
from mediacloud.api import DirectoryApi, SearchApi

from news import fulltext, mediacloud
from news.settings import NewsSettings

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
        description="Русскоязычные новости о нефти: Media Cloud и полный текст",
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
    commands.add_parser("assemble", help="собрать итоговый датасет")
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


def main() -> None:
    """Выполнить команду CLI."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx пишет в INFO каждый запрос: на сотнях тысяч статей это шум
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    settings = NewsSettings()
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
            fulltext.run(settings, retry_failed=args.retry_failed),
        ),
        "assemble": lambda: fulltext.assemble(settings),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
