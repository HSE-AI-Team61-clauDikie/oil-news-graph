# Источники данных: новостной граф нефтяного рынка → прогноз запасов → календарные спреды WTI

**Основной период исследования:** **19.02.2015 – 31.10.2026**  
**Частота основной выборки:** **1 наблюдение = 1 релиз EIA Weekly Petroleum Status Report (обычно раз в неделю)**

---

# 1. EIA Weekly Petroleum Status Report — главный таргет и фундаментальные признаки

## Источники
- Weekly Petroleum Status Report: https://www.eia.gov/petroleum/supply/weekly/
- Календарь релизов: https://www.eia.gov/petroleum/supply/weekly/schedule.php
- EIA Open Data: https://www.eia.gov/opendata/
- Документация API v2: https://www.eia.gov/opendata/documentation.php
- Petroleum & Other Liquids Data: https://www.eia.gov/petroleum/data.php

## Для чего
Это центральный источник проекта. Из него берём:
- фактическое изменение коммерческих запасов нефти;
- уровень запасов;
- Cushing stocks;
- PADD 1–5;
- добычу;
- импорт;
- экспорт;
- refinery inputs;
- refinery utilization;
- gasoline stocks;
- distillate stocks;
- SPR отдельно.

## Период
Основной период: **19.02.2015–31.10.2026**.

При этом EIA имеет гораздо более длинную историю, поэтому полную доступную историю стоит скачать отдельно для лагов и sanity-check.

## Поля
```text
week_ending
release_datetime
commercial_crude_stocks_ex_spr
delta_commercial_crude_stocks
cushing_stocks
padd1_stocks ... padd5_stocks
domestic_crude_production
crude_imports
crude_exports
refinery_inputs
refinery_utilization
gasoline_stocks
distillate_stocks
spr_stocks
```

## Как парсить
Предпочтительно через EIA API v2:

```python
import requests
import pandas as pd

url = "https://api.eia.gov/v2/..."
params = {
    "api_key": "...",
    "frequency": "weekly",
    "start": "2015-02-19",
    "end": "2026-10-31"
}
data = requests.get(url, params=params).json()
df = pd.DataFrame(data["response"]["data"])
```

Также сохранять архивные релизы WPSR через `Previous Issues`, потому что для event-study важна информация именно в том виде, в каком она была известна в день релиза.

Рекомендуемые файлы:
```text
eia_current_history.parquet
eia_release_vintages.parquet
```

---

# 2. Рыночный consensus перед EIA

## Основной источник
Trading Economics Economic Calendar:
- https://docs.tradingeconomics.com/economic_calendar/
- https://docs.tradingeconomics.com/economic_calendar/snapshot/
- https://docs.tradingeconomics.com/

## Для чего
Нам нужен не просто прогноз запасов, а расхождение модели с тем, что уже ожидал рынок:

`Inventory Surprise = Actual - Consensus`

## Поля
```text
release_datetime
event
actual
forecast
previous
revised
unit
source
last_update
```

В Trading Economics:
- `Forecast` = survey consensus;
- `TEForecast` = собственный прогноз Trading Economics.

Для проекта нужен именно `Forecast`.

## Период
Желательно **19.02.2015–31.10.2026**.

Фактическая глубина истории может зависеть от тарифа/API-доступа, поэтому это нужно проверить первым.

## Как парсить
Через официальный API:

```python
import requests

url = "https://api.tradingeconomics.com/calendar/country/united states"
params = {
    "c": "YOUR_API_KEY",
    "d1": "2015-02-19",
    "d2": "2026-10-31"
}
data = requests.get(url, params=params).json()
```

Дальше фильтровать событие вида:
```text
EIA Crude Oil Inventories
Crude Oil Stocks Change
```

Все значения привести к одной шкале: **million barrels**.

Примеры:
```text
-700K → -0.700
2.969M → 2.969
```

<span style="color:red; font-weight:bold">ПЛАТНЫЙ API. ПРОБНАЯ НЕДЕЛЯ - 49$</span>

## Альтернативы
### Investing.com
https://www.investing.com/economic-calendar/eia-crude-oil-inventories-75

Главный минус: не известно в какой момент предоставляется прогноз, и могут ли они его менять на истории

### Reuters polls
Пример публикации: https://www.reuters.com/business/energy/us-crude-gasoline-inventories-likely-fell-last-week-distillate-stocks-rose-2026-07-28/

Здесь по дате публикации легко понять момент прогноза + подмечают, что собирают до выхода API (предварительные результаты, подробнее ниже)
Минус: парсить ленту вручную и заплатить 1$

---

# 3. GDELT 2.0 — основной поток новостей

## Источники
- Data: https://www.gdeltproject.org/data.html
- GDELT 2.0 overview: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
- GKG Codebook: https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.pdf

## Для чего
GDELT нужен как основной исторический индекс новостей и источник сущностей:
- организации;
- люди;
- география;
- темы;
- тональность;
- URL публикаций;
- timestamp.

## Период
GDELT 2.0 начинается **19 февраля 2015 года**, поэтому именно эта дата задаёт нижнюю границу объединённого датасета.

## Минимальные поля
```text
gkg_id
publication_datetime
source_common_name
document_identifier
organizations
locations
persons
themes
tone
```

После собственного NLP:
```text
article_id
event_id
event_type
company_id
facility_id
region_id
capacity_affected
event_direction
event_severity
expected_duration
text_embedding
```

## Стартовые поисковые темы
```text
crude oil
WTI
refinery
refinery outage
pipeline
pipeline outage
Cushing
Gulf Coast
offshore production
oil production
oil export
oil import
storage
terminal
port
maintenance
hurricane
storm
OPEC
production cut
production increase
```

## Как парсить
Raw GKG имеет расширение `.csv`, но формат tab-separated:

```python
df = pd.read_csv(
    "file.gkgv2.csv",
    sep="\t",
    header=None,
    low_memory=False
)
```

Для больших объёмов использовать chunks и сразу сохранять отфильтрованный результат в Parquet.


## Полный текст
GDELT использовать как индекс. Для NLP полного текста:
```text
GDELT URL → публичная страница → HTML → Trafilatura/BeautifulSoup → clean text
```

---

# 4. Media Cloud — дополнительный новостной источник

## Источники
- Search API: https://www.mediacloud.org/documentation/search-api-guide
- Документация: https://www.mediacloud.org/documentation
- Search: https://www.mediacloud.org/media-cloud-search

## Для чего
Использовать для:
- проверки полноты GDELT;
- поиска тематических статей;
- анализа media attention;
- альтернативной выборки источников.

## Поля
```text
story_id
publish_date
indexed_date
media_name
title
url
language
```

---

# 5. EIA Refinery Capacity Report — НПЗ и компании

## Источник
https://www.eia.gov/petroleum/refinerycapacity/

## Для чего
Чтобы граф отражал реальную инфраструктуру.

Узлы:
```text
Company
Refinery
State
PADD
```

Рёбра:
```text
Company --operates--> Refinery
Refinery --located_in--> State
Refinery --located_in--> PADD
```

## Период
Собирать annual refinery-level snapshots за **2015–2026**.

Важно использовать историческую версию конкретного года, потому что меняются:
- собственники;
- мощности;
- статусы;
- закрытия/открытия.

## Поля
```text
refinery_name
company/operator
state
padd
capacity_bpd
status
valid_year
```

## Как парсить
EIA публикует XLSX:

```python
df = pd.read_excel("refinery_capacity_2026.xlsx")
```

После загрузки:
1. snake_case;
2. `refinery_id`;
3. нормализация названий компаний;
4. `valid_from/valid_to`.

---

# 6. BSEE — offshore production и платформы

## Источники
- Production by Platform: https://www.data.bsee.gov/Production/ProdbyPlatform/
- Production information: https://www.data.bsee.gov/Main/Production.aspx
- Platform/Rig information: https://www.data.bsee.gov/Main/Platform.aspx
- Data Center: https://www.data.bsee.gov/

## Для чего
Нужно знать:
- какие платформы существуют;
- кто оператор;
- где они находятся;
- сколько нефти производят;
- какую долю production может затронуть событие.

## Период
**2015–2026**, плюс более ранняя история при необходимости.

## Поля
```text
complex_id
structure_number
area_code
block_number
structure_name
production_month
producing_wells
oil_rate_bpd
gas_rate_mcfpd
operator_name
operator_number
lease_number
install_date
removal_date
region
```

## Как парсить
Предпочтительно raw downloads:
```python
pd.read_csv(...)
```

Для fixed-width:
```python
pd.read_fwf(...)
```

## Граф
```text
Company --operates--> Platform
Platform --located_in--> OffshoreArea
Platform --produces--> CrudeOil
```

---

# 7. BOEM/BSEE GIS — платформы и трубопроводы

## Источники
- Maps and GIS: https://www.boem.gov/oil-gas-energy/mapping-and-data
- Gulf GIS: https://www.boem.gov/oil-gas-energy/mapping-and-data/goar-geographic-information-system-gis-data-and-maps
- Downloads: https://www.data.boem.gov/Main/Mapping.aspx

## Для чего
Для пространственных связей:
```text
Platform ↔ Pipeline
Pipeline ↔ Region
Platform ↔ Hurricane
Infrastructure ↔ Gulf area
```

## Что брать
- pipelines;
- platforms;
- blocks;
- lease polygons;
- planning areas.

## Как парсить
```python
import geopandas as gpd
gdf = gpd.read_file("platforms.shp")
```

## Важно
Часть Gulf GIS использует NAD27. Перед spatial join привести всё к общей CRS, например:
```python
gdf = gdf.to_crs("EPSG:4326")
```

---

# 8. NOAA / NHC HURDAT2 — ураганы

## Источник
https://www.nhc.noaa.gov/data/

## Для чего
Ураганы могут одновременно влиять на:
- offshore production;
- платформы;
- порты;
- трубопроводы;
- НПЗ;
- импорт/экспорт.

## Период
Для проекта: **2015–2026**.

## Поля
```text
storm_id
storm_name
datetime
status
latitude
longitude
max_wind
central_pressure
```

HURDAT2 содержит шестичасовые точки трека.

## Как парсить
Текстовый comma-delimited формат организован блоками:
1. header шторма;
2. N строк трека;
3. перевод N/S/E/W в signed coordinates;
4. нормальная long-table.

## Как связать с графом
Spatial join / buffers:
```text
storm ↔ platforms within 100/200/300 km
storm ↔ refineries
storm ↔ ports
```

Полезные производные признаки:
```text
oil_capacity_exposed_bpd
refinery_capacity_exposed_bpd
number_of_platforms_exposed
```

---

# 9. Global Energy Monitor — нефтепроводы

## Источник
https://globalenergymonitor.org/research/explore-global-oil-infrastructure-tracker

## Для чего
Дополнительная глобальная инфраструктура нефтепроводов.

## Поля
```text
pipeline
owner
start_point
end_point
status
capacity
country
```

## Период
Использовать status/date поля, доступные в tracker. Для ML — привести к состоянию на каждую неделю 2015–2026.

## Парсинг
```python
pd.read_excel(...)
pd.read_csv(...)
```

Для GIS:
```python
gpd.read_file(...)
```

## Граф
```text
Company --owns--> Pipeline
Pipeline --connects--> Region A
Pipeline --connects--> Region B
```

---

# 10. OpenStreetMap / Overpass — заполнение дыр

## Источники
- Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
- APIs overview: https://wiki.openstreetmap.org/wiki/APIs
- Overpass Turbo: https://overpass-turbo.eu/

## Для чего
Только как дополнительный источник для:
- ports;
- terminals;
- storage;
- pipelines;
- refineries.

Не использовать как главный authoritative source.

## Парсинг
Overpass QL → JSON → pandas/geopandas.

---

# 11. API Weekly Statistical Bulletin — pre-EIA сигнал

**API = American Petroleum Institute.**

## Источник
https://www.api.org/energy-insights/statistics/wsb

## Для чего
API WSB обычно публикуется во вторник около 16:30 ET, то есть раньше EIA.

Можно использовать:
- crude stocks;
- gasoline stocks;
- distillate stocks;
- refinery operations;
- production/imports.

## Период
Желательно **2015–2026**, если получится получить историю.

## Доступ
Платный через официальных дистрибьюторов:
- LSEG Data & Analytics;
- ICE.

Не понятно, сколько стоит, надо оформлять заявку

## Как использовать в моделях
Model A:
```text
fundamentals + consensus + news graph
```

Model B:
```text
fundamentals + consensus + API WSB + news graph
```

Если graph помогает после API WSB — это сильный результат.

---

# 12. CME/NYMEX WTI futures — spread и backtest

## Источники
- WTI overview: https://www.cmegroup.com/education/courses/introduction-to-energy/introduction-to-crude-oil/wti-overview
- CME DataMine: https://www.cmegroup.com/datamine.html
- Settlement methodology: https://www.cmegroup.com/trading/energy/files/NYMEX_Energy_Futures_Daily_Settlement_Procedure.pdf

## Для чего
Нужны конкретные контрактные месяцы:
```text
CL front month
CL second month
CL third month
...
```

Строим:
```text
spread_1_3 = F1 - F3
spread_1_6 = F1 - F6
```

## Период
**19.02.2015–31.10.2026**

## Поля
```text
trade_datetime
contract_month
open
high
low
close
settlement
volume
open_interest
```

## Для EDA
Daily settlement достаточно.

Проверки:
```text
Inventory Surprise vs Δ(F1-F3)
Inventory Surprise vs Δ(F1-F6)
```


<span style="color:red; font-weight:bold">ПЛАТНЫЙ API. ПОЛНАЯ ИСТОРИЯ - 49$</span>

Из альтернативных источников можно собрать бесплатно нужный период


# 13. FRED / ALFRED — control variables

## Источник
https://fred.stlouisfed.org/docs/api/fred/overview.html

## Для чего
Контрольные признаки:
- broad USD index;
- VIX;
- Fed Funds;
- Treasury yields;
- другие risk/macro proxies.

Цель — проверить, содержит ли news graph дополнительную информацию сверх обычного финансового фона.

## Период
**2015–2026**

## Как парсить
REST API → JSON/XML.

```python
url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "...",
    "api_key": "...",
    "file_type": "json"
}
```

Для vintage-sensitive данных использовать ALFRED.

---

# 14. Приоритеты источников

## Обязательные до полноценной GNN
| Источник | Роль |
|---|---|
| EIA WPSR | таргет + fundamentals |
| Trading Economics / Bloomberg / LSEG | market consensus |
| CME WTI futures | calendar spreads + backtest |
| GDELT 2.0 | новости |
| EIA Refinery Capacity | НПЗ + компании + PADD |

## Второй этап
| Источник | Роль |
|---|---|
| BSEE | offshore production + operators |
| BOEM/BSEE GIS | platforms + pipelines + geography |
| NOAA HURDAT2 | hurricane shocks |
| Global Energy Monitor | pipelines |

## Дополнительные
| Источник | Роль |
|---|---|
| API WSB | pre-EIA benchmark |
| FRED / ALFRED | macro controls |
| Media Cloud | альтернативный news archive |
| OSM / Overpass | infrastructure gap-filling |

---

# 15. Master dataset

Одна строка = один релиз EIA.

```text
release_id
week_ending
release_datetime

# EIA history known before release
stocks_lag1
stocks_lag2
cushing_lag1
production_lag1
imports_lag1
exports_lag1
refinery_inputs_lag1
...

# market expectation
consensus
api_wsb_if_available

# news / graph
news_count
event_count
refinery_outage_count
production_outage_count
pipeline_event_count
graph_embedding_1
...
graph_embedding_k

# market state before release
f1_pre
f3_pre
f6_pre
spread_1_3_pre
spread_1_6_pre

# target
actual_delta_stocks
inventory_surprise

# future, only for backtest
f1_after
f3_after
f6_after
spread_1_3_after
spread_1_6_after
```

---

# 16. Короткий список финальных источников

1. EIA WPSR  
   https://www.eia.gov/petroleum/supply/weekly/

2. EIA Open Data API  
   https://www.eia.gov/opendata/documentation.php

3. Trading Economics Economic Calendar  
   https://docs.tradingeconomics.com/economic_calendar/

4. GDELT 2.0 / GKG  
   https://www.gdeltproject.org/data.html

5. EIA Refinery Capacity  
   https://www.eia.gov/petroleum/refinerycapacity/

6. CME DataMine / WTI  
   https://www.cmegroup.com/datamine.html

7. BSEE Production by Platform  
   https://www.data.bsee.gov/Production/ProdbyPlatform/

8. BOEM/BSEE GIS  
   https://www.data.boem.gov/Main/Mapping.aspx

9. NOAA/NHC HURDAT2  
   https://www.nhc.noaa.gov/data/

10. Global Energy Monitor  
    https://globalenergymonitor.org/research/explore-global-oil-infrastructure-tracker

11. API Weekly Statistical Bulletin  
    https://www.api.org/energy-insights/statistics/wsb

12. FRED / ALFRED  
    https://fred.stlouisfed.org/docs/api/fred/overview.html

13. Media Cloud  
    https://www.mediacloud.org/documentation/search-api-guide

14. OpenStreetMap Overpass  
    https://wiki.openstreetmap.org/wiki/Overpass_API

---
