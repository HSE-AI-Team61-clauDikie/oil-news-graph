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

## Регионы: выборка, разметка DeepSeek и классификатор

Новости о нефти по США, Ближнему Востоку, ЕС-27, Казахстану, Азербайджану,
Венесуэле, Мексике, Китаю и Индии собираются на языках оригинала. Отбирает
нефтяные не регулярка, а классификатор, обученный на разметке DeepSeek.

```text
regions.toml → sample: регион × год, кандидаты recall-запроса + фон вне его
  → --sample fulltext: лид - первые 500 символов живой страницы
  → label: DeepSeek Flash по промпту prompts/oil_relevance.md
  → gold + label --review: V4-Pro на gold, неуверенных и 5% контроля
  → benchmark / train: эмбеддинги + логистическая регрессия
```

```bash
uv run python -m news sample --start 2015-01-01 --end 2026-10-31
uv run python -m news --sample fulltext
uv run python -m news gold
uv run python -m news label
uv run python -m news label --review
uv run --group ml python -m news benchmark --field title
uv run --group ml python -m news train --field title --embedder intfloat/multilingual-e5-base
```

Прогон региона (`us` — имя из `regions.toml`); каждая команда пропускает
сделанное, поэтому её можно прерывать и перезапускать:

```bash
uv run python -m news --region us count
uv run python -m news --region us download
uv run python -m news --region us merge
uv run --group ml python -m news --region us filter
uv run python -m news --region us fulltext --newest-first
uv run python -m news --region us assemble
```

`filter` оценивает каждый месячный файл индекса моделью заголовков
(`data/models/news/title/meta.json`, лежит в git; энкодер скачивается с Hugging
Face) и собирает `candidates.parquet` — истории с `decision` `keep` или
`borderline` и вероятностью `probability`. `fulltext` и `assemble` региона
работают только с ними. В итоговом датасете вероятность сохраняется: порог
можно ужесточить без нового прогона.

- **`regions.toml`** — страна на регион: коллекции Media Cloud (National и
  State & Local), языки потока с долей от 1% и `full_stream`. Recall-термины
  заголовка заданы по языкам в `[terms]`; регион получает `common`, термины
  своих языков и `extra_terms`.
- **Выборка** (`sampling.py`): кандидаты — до `SAMPLE_CANDIDATES` историй
  recall-запроса на год региона, фон — `SAMPLE_BACKGROUND` историй вне его
  на год, из `SAMPLE_BACKGROUND_DAYS` дней на весь регион. Фон оценивает
  пропуски запроса: доля нефтяных в фоне × объём фона.
  - `story_sample` Media Cloud для этого не годится: он детерминированно
    отдаёт самые релевантные истории, а случайная выдача `story_list`
    закрыта (403). Поэтому дни выбираются случайно с вероятностью по объёму,
    их истории берутся `story_list` (без ранжирования) фильтром
    `publication_date`, и уже из них — случайные.
  - Вес истории — объём её страты (регион × год для кандидатов, регион для
    фона), делённый на число взятых.
  - Media Cloud пускает 2 запроса поиска в минуту: выборка по всем регионам
    идёт несколько часов, готовые регионы при перезапуске пропускаются.
  - Gold-набор (`GOLD_SIZE`) — поровну по языкам.
- **Разметка** (`labeling.py`): модель `deepseek-flash` без режима рассуждений,
  JSON-режим, `temperature=0`. Системный промпт — `prompts/oil_relevance.md` и
  справочник `prompts/oil_reference.md`; он одинаков для всех запросов, поэтому
  кэшируется (≈ 50 тыс. токенов, ≈ $0.0004 на историю). Ответы проверяются по
  схеме и пишутся пачками в part-файлы; прогон можно прерывать. Ключ —
  `DEEPSEEK_API_KEY` в `.env`.
- **Классификатор** (`classifier.py`, группа зависимостей `ml`): цель —
  «ядро или связано» против «нет». `benchmark` сравнивает модели эмбеддингов из
  `ClassifierSettings.embedders` на одной кросс-валидации: PR-AUC, recall при
  precision ≥ 0,9, macro PR-AUC по языкам, PR-AUC с весами страт, временной
  сплит (обучение до 2024 года) и скорость. `train` сохраняет модель и пороги:
  ниже `drop` история отбрасывается (recall ядра ≥ 0,95), выше `keep` — precision
  ≥ 0,9, между ними — модель «заголовок + лид».
  Итоговая модель заголовков — `intfloat/multilingual-e5-base`: e5-large точнее
  (recall при precision ≥ 0,9 — 0,963 против 0,930), но в 4 раза медленнее, а
  индекс регионов — около 9,5 млн заголовков (≈ 55 ч против ≈ 13 ч на MPS).
  На кросс-валидации при пороге 0,50: precision 0,915, recall 0,908, recall ядра
  0,950, recall «связано» 0,855; отсеивается 47,5% кандидатов выборки.
  `Alibaba-NLP/gte-multilingual-base` не входит в список: её код с Hugging Face
  несовместим с transformers 5 (`get_extended_attention_mask`), а
  `jinaai/jina-embeddings-v3` — под некоммерческой лицензией CC BY-NC.

Регионы `regions.toml` за 2015-01-01 – 2026-10-31 (объёмы — Media Cloud,
`count_over_time`, recall-запрос — после расширения терминов не пересчитан):

| Страна | Коллекции | Языки (доля потока) | Историй на языках региона, млн | Кандидатов recall, тыс. | Доля | Выкачка |
|---|---|---|---|---|---|---|
| US | 38379429, 34412234 | en 95%, es 4% | 184.72 | 1931 | 1.0% | recall |
| SA | 34412050, 38380804 | ar 79%, en 13% | 2.33 | 44 | 1.9% | recall |
| AE | 34412114, 38381108 | ar 82%, en 18% | 3.51 | 66 | 1.9% | recall |
| IQ | 34412423, 38380060 | ar 93%, en 3%, fa 1% | 0.73 | 20 | 2.7% | весь поток |
| IR | 34412284, 38380055 | fa 93%, en 4%, fr 2% | 12.36 | 250 | 2.0% | recall |
| KW | 34412071 | ar 93%, en 7% | 3.05 | 60 | 2.0% | recall |
| QA | 34412242, 38380692 | ml 67%, en 19%, ar 13% | 1.10 | 9 | 0.8% | recall |
| OM | 34412083 | ar 57%, en 42% | 0.46 | 12 | 2.6% | весь поток |
| BH | 34412049 | ml 48%, ar 44%, en 6% | 1.81 | 15 | 0.9% | recall |
| IL | 34412391 | he 65%, en 30%, fr 2%, ru 2% | 3.92 | 30 | 0.8% | recall |
| JO | 34412072, 38380252 | ar 86%, en 10% | 2.82 | 38 | 1.3% | recall |
| LB | 34412343 | ar 71%, en 22%, fr 3% | 3.62 | 62 | 1.7% | recall |
| SY | 34412453, 38380999 | ar 81%, fr 8%, en 6% | 0.45 | 8 | 1.8% | весь поток |
| YE | 34412100, 38381672 | ar 92%, en 2% | 0.73 | 14 | 1.9% | весь поток |
| EG | 34412471 | ar 97%, en 3% | 9.18 | 98 | 1.1% | recall |
| TR | 34412131, 38381036 | tr 96%, en 3% | 18.85 | 141 | 0.7% | recall |
| AT | 34412245, 38378035 | de 99% | 6.43 | 45 | 0.7% | recall |
| BE | 34412298, 38379181 | fr 53%, nl 38%, en 8%, de 1% | 9.47 | 46 | 0.5% | recall |
| BG | 34412233, 38379319 | bg 87%, en 2% | 6.80 | 96 | 1.4% | recall |
| HR | 34412323 | hr 98%, bs 2% | 6.68 | 59 | 0.9% | recall |
| CY | 34412239 | el 41%, tr 29%, ru 19%, en 11% | 3.08 | 30 | 1.0% | recall |
| CZ | 34412292 | cs 95%, en 2% | 4.34 | 31 | 0.7% | recall |
| DK | 34412412, 38379645 | da 98% | 5.33 | 21 | 0.4% | recall |
| EE | 34412338, 38379735 | et 73%, ru 21%, en 6% | 1.72 | 18 | 1.0% | recall |
| FI | 34412208, 38379781 | fi 89%, sv 10% | 6.29 | 27 | 0.4% | recall |
| FR | 34412146, 38379799 | fr 98% | 24.09 | 137 | 0.6% | recall |
| DE | 38379816, 34412409 | de 91%, ru 6% | 55.62 | 425 | 0.8% | recall |
| GR | 34412477, 38379845 | el 98%, en 2% | 11.40 | 83 | 0.7% | recall |
| HU | 34412252, 38379904 | hu 94%, en 1% | 9.26 | 79 | 0.9% | recall |
| IE | 34412271, 38380065 | en 99% | 3.04 | 19 | 0.6% | recall |
| IT | 34412372, 38380117 | it 95%, en 4%, de 1% | 21.04 | 196 | 0.9% | recall |
| LV | 34412437, 38380266 | lv 75%, ru 23%, en 1% | 2.12 | 23 | 1.1% | recall |
| LT | 38379746 | lt 92%, ru 2%, en 1% | 2.87 | 29 | 1.0% | recall |
| LU | 38380287, 38379192 | fr 72%, de 20%, en 3%, pt 3%, lb 2% | 1.77 | 10 | 0.5% | recall |
| MT | 34412381 | en 64%, mt 35%, it 2% | 0.56 | 4 | 0.7% | recall |
| NL | 34412382, 38380454 | nl 99% | 12.97 | 69 | 0.5% | recall |
| PL | 34412416, 38380631 | pl 100% | 9.82 | 88 | 0.9% | recall |
| PT | 34412337, 38380661 | pt 99% | 7.81 | 78 | 1.0% | recall |
| RO | 34412235, 38380695 | ro 97%, hu 2%, en 1% | 8.94 | 78 | 0.9% | recall |
| SK | 34412152, 38380860 | sk 91%, hu 5%, en 4% | 3.89 | 36 | 0.9% | recall |
| SI | 34412061 | sl 99% | 2.58 | 22 | 0.9% | recall |
| ES | 34412356, 38002034 | es 93%, ca 5%, en 1% | 54.97 | 307 | 0.6% | recall |
| SE | 34412223, 38380910 | sv 99% | 16.24 | 62 | 0.4% | recall |
| KZ | 34412415, 38380255 | ru 82%, kk 10%, en 4% | 3.82 | 55 | 1.4% | recall |
| AZ | 205925810, 34412123 | az 65%, ru 26%, en 6% | 10.89 | 373 | 3.4% | recall |
| VE | 34412387, 38380333 | es 97% | 5.11 | 131 | 2.6% | recall |
| MX | 34412427, 38380322 | es 99% | 30.87 | 398 | 1.3% | recall |
| CN | 34412193, 38379438 | zh 57%, en 14%, ru 4% | 4.89 | 33 | 0.7% | recall |
| IN | 38379954, 34412118 | en 43%, hi 42%, ta 5%, ml 3%, mr 2%, gu 2%, kn 1% | 69.53 | 741 | 1.1% | recall |

| Путь | Что лежит |
|---|---|
| `data/interim/news/sample/parts/{регион}_{год}.parquet` | выборка страты |
| `data/interim/news/sample/counts/{регион}.parquet` | объёмы страт по годам |
| `data/interim/news/sample/stories.parquet` | выборка без дублей, с весами |
| `data/interim/news/sample/gold.parquet` | gold-набор для ручной проверки |
| `data/raw/news/sample/fulltext/part-NNNNN.parquet` | тексты выборки |
| `data/interim/news/labels/{модель}/part-NNNNN.parquet` | ответы DeepSeek |
| `data/interim/news/{регион}/filtered/stories_YYYY-MM.parquet` | вероятность и решение по каждой истории месяца |
| `data/interim/news/{регион}/candidates.parquet` | прошедшие фильтр истории индекса |
| `data/models/news/benchmark-{поле}.csv` | сравнение моделей эмбеддингов |
| `data/models/news/{поле}/meta.json` | энкодер, веса регрессии, пороги и метрики; `title` лежит в git |

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
