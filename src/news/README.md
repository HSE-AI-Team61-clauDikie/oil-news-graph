# Русскоязычные новости о нефти: Media Cloud + полный текст

Пайплайн собирает русскоязычные новости о нефтяном рынке за 2015-02-19 –
2026-10-31 вместе с полным текстом статей:

```text
Media Cloud (индекс: url, дата, источник, заголовок)
  → живая страница издания  ─┐
  → снимок Wayback Machine  ─┴→ trafilatura → текст
  → датасет с отметками перепечаток
```

## Почему Media Cloud, а не GDELT

| | GDELT DOC 2.0 API | GDELT raw GKG translingual | **Media Cloud Search API** |
|---|---|---|---|
| Глубина | только последние 3 месяца | с 2015-02-19 | архив переобработан примерно за 10 лет; у каждого источника своё начало |
| Объём 2015–2026 | — | zip раз в 15 минут по 10–16 МБ, ~4 ТБ | ~1000 историй на запрос, 4000 запросов в неделю |
| Русский текст | поиск по английскому машинному переводу | 300–500 строк `srclc:rus` на файл | заголовок в оригинале, `language:ru`, запрос на русском |
| Дубликаты | много перепечаток | одна статья в нескольких выгрузках | URL нормализуются при индексации |
| Доступ | без ключа, жёсткий rate limit | без ключа | бесплатный аккаунт и API key |
| Полный текст | нет | нет | нет: текст есть только у администраторов |

Media Cloud выбран как индекс: один пагинируемый API, русские заголовки и
меньше всего дублей. Полный текст не отдаёт ни один агрегатор, поэтому он
скачивается по URL.

### Откуда берётся текст

Проверено на реальных URL русскоязычных статей:

| Сэмпл | Живая страница | Wayback Machine | Common Crawl index |
|---|---|---|---|
| 25 статей 2026 года | 25/25 с текстом | — | — |
| 25 статей 2019 года | 16/25: 404, 403, закрытые сайты | 9/10 с текстом, включая страницы, которые сейчас отдают 404 | 504 на каждый запрос |

`fulltext` работает в два этапа:

1. **Живые страницы** всех статей: до `CONCURRENCY` запросов одновременно,
   до `PER_HOST_CONCURRENCY` на один сайт. Это быстро.
2. **Wayback** для статей, которые не дали текста: страница недоступна, текст
   короче `MIN_TEXT_CHARS` или это не статья. Берётся ближайший к дате
   публикации снимок (CDX API + `id_`, то есть исходный HTML без панели
   Wayback). Это медленно, но первый этап уже не ждёт второй.

«Не статья» — это редирект на главную сайта или один и тот же текст на
`JUNK_REPEATS` и больше страницах одного сайта: так старые адреса vesti.ru
отдают главную, а bashinform.ru — подвал вместо статьи. Такие тексты уходят в
Wayback, а в датасет не попадают. Текст и метаданные извлекает trafilatura.

Источники:
[Media Cloud Search API](https://www.mediacloud.org/documentation/search-api-guide),
[Query Guide](https://www.mediacloud.org/documentation/query-guide),
[FAQ](https://www.mediacloud.org/documentation/faqs),
[mediacloud/api-client](https://github.com/mediacloud/api-client),
[GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/),
[GDELT Web NGrams](https://blog.gdeltproject.org/using-the-new-web-ngrams-dataset-to-find-relevant-coverage/),
[Wayback CDX API](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server),
[trafilatura](https://trafilatura.readthedocs.io/).

## Запуск

Ключ выдаётся после регистрации на <https://search.mediacloud.org>; его нужно
положить в `.env` в корне репозитория:

```text
MEDIACLOUD_API_KEY=...
```

Команды запускаются из корня репозитория:

```bash
uv run python -m news count                  # объём по месяцам и расход квоты
uv run python -m news download               # индекс помесячно, можно прерывать
uv run python -m news merge                  # индекс без дублей
uv run python -m news fulltext               # тексты, можно прерывать
uv run python -m news fulltext --newest-first  # от 2026 к 2015
uv run python -m news fulltext --retry-failed
uv run python -m news assemble               # итоговый датасет
```

`count` и `download` принимают `--start` и `--end` (`YYYY-MM-DD`).

Поиск без коллекций Media Cloud не выполняет (422 `Must have at least one of
'ss' or 'cs'`). По умолчанию берутся российские коллекции:

| id | Коллекция | Источников |
|---|---|---|
| 34412232 | Russian Federation - National | 242 |
| 38380780 | Russia - State & Local | 36 |
| 38380781 | Moskva, Russia - State & Local | 36 |

Другие коллекции ищет `collections --name Kazakhstan`; набор задаётся через
`COLLECTION_IDS='[id, ...]'` (JSON-список).

Любое поле `NewsSettings` (`settings.py`) переопределяется переменной окружения
с тем же именем: `QUERY`, `CONCURRENCY`, `WAYBACK_INTERVAL` и т.д.

## Файлы

| Путь | Что лежит |
|---|---|
| `data/raw/news/mediacloud/stories_YYYY-MM.parquet` | индекс Media Cloud за месяц |
| `data/interim/news/stories.parquet` | склеенный индекс без дублей |
| `data/raw/news/fulltext/part-NNNNN.parquet` | попытки загрузки текста |
| `data/processed/news/ru_oil_news.parquet` | итоговый датасет |

### Как собирается датасет

`assemble` (`news/dataset.py`) склеивает индекс и тексты и чистит их:

- **Шум по заголовку.** Отбрасываются заголовки, подходящие под `TITLE_EXCLUDE`:
  спортклубы «Нефтехимик» и «Нефтчи», «опека» и «опекун». Запрос Media Cloud
  исключает их и сам, но индекс, скачанный старым запросом, тоже чистится.
- **Шаблоны вместо статьи** — те же, что идут в Wayback: редирект на главную
  или один текст на многих страницах сайта.
- **Даты.** Media Cloud иногда ставит дату импорта вместо даты публикации:
  сотни статей источника на один день (rbc.ru 2016-12-22, profile.ru 2026-01-01).
  Поэтому `date` берётся из адреса статьи (`/2017/05/12/`, `/20170512/`), если
  его нет и дата Media Cloud попала в такой кластер — со страницы (кроме
  1 января: так trafilatura отдаёт год без дня), иначе — из Media Cloud. Если
  кластер есть, а другой даты нет, `date_source = unreliable`.
- **Перепечатки** отмечаются в `duplicate_of`: самая ранняя публикация текста
  остаётся оригиналом.

Колонки итогового датасета:

| Колонка | Смысл |
|---|---|
| `story_id`, `publish_date`, `indexed_date`, `media_name`, `media_url`, `title`, `url`, `language` | индекс Media Cloud |
| `date` | дата публикации после проверки; для анализа брать её, а не `publish_date` |
| `date_source` | откуда `date`: `url`, `page`, `media_cloud` или `unreliable` |
| `text` | текст статьи |
| `text_source` | `live` или `wayback` |
| `final_url`, `http_status`, `wayback_timestamp` | откуда взят текст |
| `extracted_title`, `extracted_date` | заголовок и дата по версии trafilatura |
| `text_hash` | sha1 текста без регистра и пунктуации |
| `duplicate_of` | `story_id` самой ранней публикации того же текста |

## Ограничения

- Квота Media Cloud: 4000 запросов в неделю, ~1000 историй на запрос. Перед
  выкачкой стоит запустить `count`.
- archive.org отвечает 429 при запросах чаще ~1 в секунду. По умолчанию запросы
  к нему идут раз в 4 секунды, а после 429 вся очередь ждёт минуту. На статью
  из архива уходит два запроса, поэтому этап Wayback — самый долгий. Он
  сохраняется пачками по `WAYBACK_BATCH_SIZE`, и его можно прерывать: датасет
  из уже скачанного собирается `assemble` в любой момент.
- Часть сайтов закрыта антиботом (403), стеной авторизации (gazeta.ru
  перенаправляет на `/auth/sso.shtml`) или закрылась сама. Если их нет
  и в Wayback, у статьи остаётся `text_source = none` и причина в `error`.
- Покрытие источников в Media Cloud неравномерно по годам: для каждого источника
  данные начинаются с момента, когда его RSS стали собирать.
