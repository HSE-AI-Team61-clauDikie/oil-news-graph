"""Автономный разведочный анализ многоязычного новостного корпуса.

Run from the repository root: .venv/bin/python eda/news_eda.py
Таблицы и графики сохраняются в eda/, основной отчёт — в docs/news_eda.md.
Для расчёта достаточно локальных файлов. Страна обозначает коллекцию источников.
"""

from __future__ import annotations

# Long multilingual regexes and report paragraphs are kept readable as literals.
# ruff: noqa: E501
import argparse
import base64
import hashlib
import html
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import eda_utils as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-10-31")
MONTHS = pd.period_range(START, END, freq="M")
DERIVED = ROOT / "eda/data/derived/news"
REPORT = ROOT / "eda/reports/news"
SEED = 6107

# Transparent title screens, not event extraction or validated relevance labels.
# Coverage varies by language. Keep overlaps; never infer absence of an event.
TOPICS = {
    "refining": r"refiner|raffin|rafiner|finomít|διυλισ|рафин|مصفا|مصاف|تكرير|jalostam",
    "pipelines": r"pipeline|oleoduct|oléoduc|oleodott|ropovod|naftovod|pijpleid|rurociąg|Druzhba|Družba|трубопровод|أنابيب|انابيب|αγωγ|torujuhe",
    "outages": r"outage|shutdown|shut down|shut[s]?\b|disrupt|explosion|fire\b|maintenance|panne|incend|fermeture|Wartung|Explosion|Brand\b|chiusur|interru|parada|انفجار|حريق|توقف|إغلاق|اغلاق|صيانة|έκρηξ|пожар|авари",
    "production": r"\boutput\b|\bproduction\b|drilling|oilfield|Produktion|Förder|producci|produzio|doby|termel|إنتاج|انتاج|حفر|добыч|производ|παραγωγ",
    "opec": r"\bOPEC\b|\bOPEP\b|أوبك|اوبك|ОПЕК",
    "weather": r"hurricane|cyclone|tropical storm|flood|freeze|storm|ouragan|tempête|Hurrikan|uragan|huracán|إعصار|اعصار|عاصفة|فيضانات|ураган|шторм|torm|hurrik",
    "shipping_trade": r"tanker|shipping|export|import|freight|Hormuz|Suez|Red Sea|صادرات|واردات|تصدير|استيراد|ناقل|هرمز|البحر الأحمر|экспорт|импорт|танкер",
    "geopolitics": r"sanction|sanzi|Sanktion|санкц|عقوب|war\b|guerre|Krieg|guerra|حرب|attack|attaque|Angriff|атака|هجوم",
    "price_commentary": r"\bprice|\bprices|\bprix|\bPreis|prezz|precio|цены|цен[ауы]|أسعار|اسعار|سعر|árak|cijena|ceny|τιμ",
    "inventory_reporting": r"\bEIA\b|\bAPI\b.{0,45}(?:crude|inventor|stock)|(?:crude|inventor|stock).{0,45}\bAPI\b|petroleum status|crude.{0,25}inventor|oil.{0,25}stockpile|stocks? de brut|reservas de crudo|scorte.{0,20}greggio|Rohöllager|مخزون.{0,30}(?:النفط|الخام)|(?:إدارة|ادارة) معلومات الطاقة|معهد البترول الأمريكي|запас.{0,25}нефт",
}
ENTITIES = {
    "Cushing": r"\bCushing\b|Кушинг|كوشينغ",
    "Permian": r"\bPermian\b|Пермск|بيرميان",
    "Gulf_of_Mexico": r"Gulf of Mexico|Gulf of America|golfe du Mexique|golfo de M[eé]xico|Мексиканск.{0,8}залив|خليج المكسيك",
    "US_oil_institutions": r"\bEIA\b|\bWTI\b|\bSPR\b|American Petroleum|إدارة معلومات الطاقة|معهد البترول الأمريكي",
    "Exxon_Chevron": r"\bExxon|\bChevron|إكسون|اكسون|شيفرون",
    "Aramco_ADNOC": r"\bAramco\b|\bADNOC\b|أرامكو|ارامكو|أدنوك|ادنوك",
    "Pemex_PDVSA": r"\bPemex\b|\bPDVSA\b|بيميكس",
}


TOPIC_NAMES = {
    "refining": "Переработка",
    "pipelines": "Трубопроводы",
    "outages": "Остановки и аварии",
    "production": "Добыча и производство",
    "opec": "ОПЕК",
    "weather": "Погода",
    "shipping_trade": "Перевозки и торговля",
    "geopolitics": "Геополитика",
    "price_commentary": "Цены",
    "inventory_reporting": "Отчётность о запасах",
}
ENTITY_NAMES = {
    "Cushing": "Cushing",
    "Permian": "Permian",
    "Gulf_of_Mexico": "Мексиканский залив",
    "US_oil_institutions": "EIA, WTI, SPR и др.",
    "Exxon_Chevron": "Exxon / Chevron",
    "Aramco_ADNOC": "Aramco / ADNOC",
    "Pemex_PDVSA": "Pemex / PDVSA",
}
FAILURE_NAMES = {
    "timeout": "Тайм-аут",
    "network/TLS": "Сеть / TLS",
    "extraction/short": "Извлечение / короткий текст",
    "other": "Прочие",
}


def covered_months(path: Path) -> set[str]:
    """Read packed YYYY-MM_YYYY-MM windows, including zero-story months."""
    bounds = path.stem.removeprefix("stories_").split("_")
    return set(pd.period_range(bounds[0], bounds[-1], freq="M").astype(str))


def aggregate_month(frame: pd.DataFrame, date: str, column: str) -> pd.Series:
    """Count rows by publication month; invalid/out-of-range dates stay in QA."""
    return frame.groupby(frame[date].dt.to_period("M")).size().rename(column)


def assign_release(dates: pd.Series, releases: pd.DatetimeIndex) -> pd.Series:
    """Next observed release, strictly after the article's publication day.

    Day-only timestamps cannot establish pre-release intraday availability.
    Same-day articles go to the following release. This is a retrospective
    publication-date screen, not proof of point-in-time text availability.
    """
    day = pd.to_datetime(dates).dt.normalize()
    pos = releases.searchsorted(day, side="right")
    result = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = day.notna() & (pos < len(releases))
    result.loc[ok] = releases.take(pos[ok]).to_numpy()
    return result


def concentration(counts: pd.Series) -> dict[str, float]:
    """Source concentration within the observed sample."""
    share = counts.sort_values(ascending=False) / counts.sum()
    return {
        "sources": len(counts),
        "top1_share": share.iloc[0],
        "top5_share": share.head(5).sum(),
        "effective_sources": 1 / share.pow(2).sum(),
    }


def article_metadata(path: Path) -> pd.DataFrame:
    """Вычислить признаки порциями, освобождая полный текст после каждой порции."""
    parquet = pq.ParquetFile(path)
    omitted = {
        "final_url",
        "error",
        "extracted_title",
        "wayback_timestamp",
        "http_status",
    }
    columns = [name for name in parquet.schema_arrow.names if name not in omitted]
    parts = []
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        data = batch.to_pandas()
        data["chars"] = data.text.str.len().fillna(0).astype(int)
        data["words"] = data.text.str.count(r"\S+").fillna(0).astype(int)
        data["excerpt"] = data.text.str.slice(0, 600)
        data["title_normalized"] = (
            data.title.fillna("")
            .str.casefold()
            .str.replace(r"[^\w]+", " ", regex=True)
            .str.strip()
        )
        data["host"] = data.url.str.extract(
            r"^https?://(?:www\.)?([^/:]+)", expand=False
        ).str.casefold()
        data["date_shift_days"] = (
            data.date.dt.normalize() - data.publish_date.dt.normalize()
        ).dt.days
        data["date_ok"] = data.date.between(START, END) & ~data.date_source.eq(
            "unreliable"
        )
        data["length_ok"] = data.chars.between(300, 100_000)
        data["quality_ok"] = (
            data.date_ok
            & data.length_ok
            & data.title_normalized.ne("")
            & data.text_hash.notna()
        )
        data["exact_time_present"] = data.date.notna() & data.date.ne(
            data.date.dt.normalize()
        )
        page_date = (
            pd.to_datetime(data.extracted_date, errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        data["page_date_shift_days"] = (page_date - data.date.dt.normalize()).dt.days
        for topic, pattern in TOPICS.items():
            data["topic_" + topic] = data.title.str.contains(
                pattern, case=False, regex=True, na=False
            )
        for entity, pattern in ENTITIES.items():
            data["entity_" + entity] = data.title.str.contains(
                pattern, case=False, regex=True, na=False
            )
        parts.append(data.drop(columns=["text", "extracted_date"]))
    return pd.concat(parts, ignore_index=True)


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": pq.read_metadata(path).num_rows,
    }


def collect() -> dict[str, pd.DataFrame]:  # noqa: C901 - sequential stage accounting
    """Scan each region separately so the full text corpus never fills RAM."""
    config = tomllib.loads((ROOT / "src/news/regions.toml").read_text())
    rows, monthly, articles, sources, failures, scores, weeks = ([] for _ in range(7))
    files = [
        file_record(p)
        for p in [
            u.DATA / "consensus/myfxbook_eia_consensus.parquet",
            u.DATA / "consensus/trendonify_api_crude_full_history.parquet",
            u.DATA / "eia/eia_weekly_features.parquet",
        ]
    ]
    releases = pd.DatetimeIndex(
        u.release_frame()["release_date"].unique()
    ).sort_values()
    for region, spec in config["regions"].items():
        raw = ROOT / "data/raw/news" / region
        interim = ROOT / "data/interim/news" / region
        processed = ROOT / "data/processed/news" / region / "oil_news.parquet"
        paths = sorted((raw / "mediacloud").glob("stories_*.parquet"))
        all_paths = [
            *paths,
            *sorted((interim / "filtered").glob("*.parquet")),
            *sorted((raw / "fulltext").glob("part-*.parquet")),
        ]
        all_paths += [
            p
            for p in [
                interim / "stories.parquet",
                interim / "candidates.parquet",
                processed,
            ]
            if p.exists()
        ]
        files.extend(file_record(path) for path in all_paths)
        covered = set().union(*(covered_months(p) for p in paths)) if paths else set()
        scored_paths = sorted((interim / "filtered").glob("stories_*.parquet"))
        scored_months = (
            set().union(*(covered_months(p) for p in scored_paths))
            if scored_paths
            else set()
        )
        month = pd.DataFrame(index=MONTHS)
        month["index_downloaded"] = month.index.astype(str).isin(covered)
        month["index_scored"] = month.index.astype(str).isin(scored_months)
        row = {
            "region": region,
            "mode": "full_stream" if spec.get("full_stream") else "recall",
            "expected_languages": ",".join(spec["languages"]),
            "months_downloaded": len(covered),
            "months_scored": len(scored_months),
            "raw_rows": sum(pq.read_metadata(p).num_rows for p in paths),
        }
        for key in [
            "index_rows",
            "candidates",
            "attempt_rows",
            "attempted",
            "attempted_candidates",
            "extracted",
            "assembled",
        ]:
            row[key] = 0
        weekly = pd.DataFrame(index=releases)
        index = None
        if (interim / "stories.parquet").exists():
            index = pd.read_parquet(
                interim / "stories.parquet",
                columns=["publish_date"],
            )
            row["index_rows"] = len(index)
            month = month.join(aggregate_month(index, "publish_date", "indexed"))
            weekly["indexed"] = index.groupby(
                assign_release(index.publish_date, releases)
            ).size()
            row["index_first"] = index.publish_date.min()
            row["index_last"] = index.publish_date.max()
        candidates = None
        if (interim / "candidates.parquet").exists():
            candidates = pd.read_parquet(
                interim / "candidates.parquet",
                columns=[
                    "story_id",
                    "publish_date",
                    "media_name",
                    "language",
                ],
            )
            row["candidates"] = len(candidates)
            month = month.join(
                aggregate_month(candidates, "publish_date", "candidates")
            )
            weekly["candidates"] = candidates.groupby(
                assign_release(candidates.publish_date, releases)
            ).size()
        if scored_paths:
            sc = pd.concat(
                [
                    pd.read_parquet(p, columns=["probability", "decision"])
                    for p in scored_paths
                ],
                ignore_index=True,
            )
            bins = pd.cut(sc.probability, np.linspace(0, 1, 21), include_lowest=True)
            hist = sc.groupby(bins, observed=False).size()
            for interval, count in hist.items():
                scores.append(
                    {"region": region, "bin_mid": interval.mid, "count": count}
                )
            row["scored_rows"] = len(sc)
            row["borderline"] = int(sc.decision.eq("borderline").sum())
            del sc
        attempt_paths = sorted((raw / "fulltext").glob("part-*.parquet"))
        if attempt_paths:
            at = pd.concat(
                [
                    pd.read_parquet(
                        p,
                        columns=[
                            "story_id",
                            "text_source",
                            "http_status",
                            "error",
                            "fetched_at",
                        ],
                    )
                    for p in attempt_paths
                ],
                ignore_index=True,
            )
            row["attempt_rows"] = len(at)
            at = at.sort_values("fetched_at", kind="stable").drop_duplicates(
                "story_id", keep="last"
            )
            row["attempted"] = len(at)
            row["extracted"] = int(at.text_source.ne("none").sum())
            row["fetched_last"] = at.fetched_at.max()
            fail = at[at.text_source.eq("none")].copy()
            error = fail.error.fillna("unspecified")
            fail["failure"] = np.select(
                [
                    fail.http_status.notna(),
                    error.str.contains("Timeout", case=False),
                    error.str.contains(
                        "SSL|Connect|Network|RemoteProtocol", case=False
                    ),
                    error.str.contains("короче|текст|article|extract", case=False),
                ],
                [
                    "HTTP " + fail.http_status.fillna(0).astype(int).astype(str),
                    "timeout",
                    "network/TLS",
                    "extraction/short",
                ],
                default="other",
            )
            for name, count in fail.failure.value_counts().items():
                failures.append({"region": region, "failure": name, "count": count})
            if candidates is not None:
                joined = candidates.merge(
                    at[["story_id", "text_source"]],
                    on="story_id",
                    how="inner",
                    validate="one_to_one",
                )
                row["attempted_candidates"] = len(joined)
                month = month.join(aggregate_month(joined, "publish_date", "attempted"))
                weekly["attempted"] = joined.groupby(
                    assign_release(joined.publish_date, releases)
                ).size()
                month = month.join(
                    aggregate_month(
                        joined[joined.text_source.ne("none")],
                        "publish_date",
                        "extracted",
                    )
                )
                for stage, data in [("candidate", candidates), ("attempted", joined)]:
                    for (name, lang), count in (
                        data.groupby(["media_name", "language"]).size().items()
                    ):
                        sources.append(
                            {
                                "region": region,
                                "stage": stage,
                                "source": name,
                                "language": lang,
                                "count": count,
                            }
                        )
            del at
        if processed.exists() and pq.read_metadata(processed).num_rows:
            data = article_metadata(processed)
            row["assembled"] = len(data)
            data["region"] = region
            month = month.join(aggregate_month(data, "date", "assembled"))
            weekly["assembled"] = (
                data[data.date_ok]
                .groupby(assign_release(data.loc[data.date_ok, "date"], releases))
                .size()
            )
            conc = concentration(data.groupby("host").size())
            row.update(conc)
            row["first_text"] = data.date.min()
            row["last_text"] = data.date.max()
            row["invalid_dates"] = int((~data.date_ok).sum())
            row["date_changed"] = int(data.date_shift_days.ne(0).sum())
            row["local_duplicate_rows"] = int(data.duplicate_of.notna().sum())
            row["median_chars"] = data.chars.median()
            for (name, lang), count in (
                data.groupby(["media_name", "language"]).size().items()
            ):
                sources.append(
                    {
                        "region": region,
                        "stage": "assembled",
                        "source": name,
                        "language": lang,
                        "count": count,
                    }
                )
            articles.append(data)
        for column in ["indexed", "candidates", "attempted", "extracted", "assembled"]:
            if column not in month:
                month[column] = 0
            month[column] = month[column].fillna(0).astype(int)
        month["region"] = region
        monthly.append(month.rename_axis("month").reset_index())
        weekly["region"] = region
        for col in ["indexed", "candidates", "attempted", "assembled"]:
            if col not in weekly:
                weekly[col] = 0
            weekly[col] = weekly[col].fillna(0).astype(int)
        weeks.append(weekly.rename_axis("release_date").reset_index())
        rows.append(row)
        print(  # noqa: T201 - CLI progress
            f"{region}: {row['index_rows']:,} indexed / {row['candidates']:,} selected / {row['assembled']:,} assembled",
            flush=True,
        )
    return {
        "regions": pd.DataFrame(rows),
        "monthly": pd.concat(monthly, ignore_index=True),
        "articles": pd.concat(articles, ignore_index=True),
        "sources": pd.DataFrame(sources),
        "failures": pd.DataFrame(failures),
        "score_hist": pd.DataFrame(scores),
        "weekly": pd.concat(weeks, ignore_index=True),
        "manifest": pd.DataFrame(files),
    }


def full_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """Полные интервалы [релиз − 7 дней, релиз) внутри периода выгрузки."""
    end = weekly.release_date
    return weekly.loc[
        weekly.window_days.eq(7)
        & (end - pd.Timedelta(days=7)).ge(START)
        & (end - pd.Timedelta(days=1)).le(END)
    ]


def count_correlations(weekly: pd.DataFrame) -> pd.DataFrame:
    """Сравнить страны и учесть каждую различную выборку окон один раз."""
    rows = []
    for region, group in (
        full_weeks(weekly)
        .dropna(subset=["assembled", "surprise_mmbbl"])
        .groupby("region")
    ):
        any_attempt = group.attempted.gt(0)
        covered = group.attempt_fraction.ge(0.8)
        masks = (
            [("both", covered)]
            if any_attempt.equals(covered)
            else [
                ("any_attempt", any_attempt),
                ("80pct_attempted", covered),
            ]
        )
        for name, mask in masks:
            d = group.loc[mask]
            if (
                len(d) >= 30
                and d.assembled.nunique() > 1
                and d.surprise_mmbbl.nunique() > 1
            ):
                rho, p = stats.spearmanr(d.assembled, d.surprise_mmbbl)
                rows.append(
                    {"region": region, "screen": name, "n": len(d), "rho": rho, "p": p}
                )
    cor = pd.DataFrame(rows, columns=["region", "screen", "n", "rho", "p"])
    cor["p_bh"] = (
        stats.false_discovery_control(cor.p, method="bh") if len(cor) else np.nan
    )
    return cor


def volume_benchmark(weekly: pd.DataFrame) -> pd.DataFrame:
    """Сравнить базовую модель API и два счётчика новостей на общих датах."""
    d = (
        full_weeks(weekly)
        .groupby("release_date")
        .agg(
            attempted=("attempted", "sum"),
            candidates=("candidates", "sum"),
            texts=("unique_screened", "sum"),
            physical=("physical_candidates", "sum"),
            api=("api_mmbbl", "first"),
            consensus=("consensus_mmbbl", "first"),
            target=("surprise_mmbbl", "first"),
        )
    )
    d = (
        d.loc[d.attempted.eq(d.candidates) & d.candidates.gt(0) & d.texts.gt(0)]
        .dropna()
        .copy()
    )
    d["api_gap"] = d.api - d.consensus
    d["log_texts"] = np.log1p(d.texts)
    d["physical_share"] = d.physical / d.texts
    train = d.index < pd.Timestamp("2023-01-01")
    rows = []
    for name, cols in [
        ("api", ["api_gap"]),
        ("api_news", ["api_gap", "log_texts", "physical_share"]),
    ]:
        if train.sum() < 4 or train.all():
            break
        x = np.column_stack([np.ones(len(d)), d[cols]])
        beta = np.linalg.lstsq(x[train], d.target[train], rcond=None)[0]
        residual = d.target[~train] - x[~train] @ beta
        rows.append(
            {
                "model": name,
                "train_weeks": int(train.sum()),
                "test_weeks": int((~train).sum()),
                "mae_mmbbl": residual.abs().mean(),
                "rmse_mmbbl": np.sqrt(residual.pow(2).mean()),
            }
        )
    return pd.DataFrame(
        rows, columns=["model", "train_weeks", "test_weeks", "mae_mmbbl", "rmse_mmbbl"]
    )


def summarize(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Global duplicate audit, quality screens, and coverage-aware release panels."""
    a = tables["articles"]
    cluster_qa = a.groupby("text_hash").agg(
        rows=("story_id", "size"),
        distinct_titles=("title_normalized", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )
    cluster_qa["span_days"] = (cluster_qa["last"] - cluster_qa["first"]).dt.days
    suspicious = cluster_qa.query(
        "rows >= 10 and distinct_titles >= 5 and span_days > 30"
    )
    a["template_cluster_screen"] = a.text_hash.isin(suspicious.index)
    a["consent_screen"] = a.excerpt.str.contains(
        r"iDNES a reklama|Before you continue to|Please enable JavaScript|Access Denied|subscribe to continue|consent to (?:the use|cookies)|Zustimmung zur|consentement.{0,30}cookies",
        case=False,
        regex=True,
        na=False,
    )
    a["content_warning"] = a.template_cluster_screen | a.consent_screen
    a["date_disagreement_gt7d"] = a.date_shift_days.abs().gt(7)
    a["quality_ok"] = a.quality_ok & ~a.content_warning
    tables["template_clusters"] = suspicious.sort_values(
        "rows", ascending=False
    ).reset_index()
    canonical = a.loc[
        a.quality_ok, ["date", "story_id", "region", "text_hash"]
    ].sort_values(["date", "story_id", "region"], kind="stable")
    canonical = canonical.drop_duplicates("text_hash")
    a["global_original"] = a.index.isin(canonical.index)
    a["screened"] = a.quality_ok & a.global_original
    a["physical_event_screen"] = a[
        [
            "topic_" + k
            for k in [
                "refining",
                "pipelines",
                "outages",
                "production",
                "weather",
                "shipping_trade",
            ]
        ]
    ].any(axis=1)
    a["candidate_event"] = (
        a.screened & a.physical_event_screen & ~a.topic_inventory_reporting
    )
    g = a.groupby("text_hash").agg(
        rows=("region", "size"),
        regions=("region", "nunique"),
        hosts=("host", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )
    tables["duplicate_clusters"] = (
        g[g.rows.gt(1)].sort_values("rows", ascending=False).reset_index()
    )
    tables["quality"] = (
        a.groupby("region")
        .agg(
            assembled=("story_id", "size"),
            valid_dates=("date_ok", "sum"),
            quality_pass=("quality_ok", "sum"),
            unique_screened=("screened", "sum"),
            physical_candidates=("candidate_event", "sum"),
            direct_reporting_flags=("topic_inventory_reporting", "sum"),
            exact_timestamps=("exact_time_present", "sum"),
            very_long=("chars", lambda x: x.gt(100_000).sum()),
            content_warnings=("content_warning", "sum"),
            date_disagreements=("date_disagreement_gt7d", "sum"),
        )
        .reset_index()
    )
    tables["languages"] = (
        a.groupby("language")
        .agg(
            rows=("story_id", "size"),
            regions=("region", "nunique"),
            median_chars=("chars", "median"),
            screened=("screened", "sum"),
            mean_score=("probability", "mean"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    tables["source_year"] = (
        a.loc[a.date_ok, ["region", "host"]]
        .groupby(["region", a.date.dt.year.rename("year"), "host"])
        .size()
        .rename("rows")
        .reset_index()
    )
    tables["topics"] = (
        a.groupby("region")[["topic_" + k for k in TOPICS]].mean().reset_index()
    )
    tables["entities"] = (
        a.groupby("region")[["entity_" + k for k in ENTITIES]].sum().reset_index()
    )
    release = u.release_frame()[
        ["release_date", "period", "surprise_mmbbl", "api_mmbbl", "consensus_mmbbl"]
    ]
    dates = pd.DatetimeIndex(release.release_date.unique()).sort_values()
    a["next_release"] = assign_release(a.date, dates)
    weekly = tables["weekly"].merge(
        release, on="release_date", how="left", validate="many_to_one"
    )
    count = (
        a.loc[a.screened, ["region", "next_release", "candidate_event"]]
        .groupby(["region", "next_release"])
        .candidate_event.agg(unique_screened="size", physical_candidates="sum")
    )
    weekly = weekly.merge(
        count, left_on=["region", "release_date"], right_index=True, how="left"
    )
    count_columns = ["unique_screened", "physical_candidates"]
    weekly[count_columns] = weekly[count_columns].fillna(0).astype(int)
    weekly["attempt_fraction"] = weekly.attempted / weekly.candidates.replace(0, np.nan)
    weekly["observed_text_count"] = weekly.assembled.astype(float).where(
        weekly.attempted.gt(0)
    )
    previous = pd.Series(dates, index=dates).shift(1)
    weekly["window_start"] = weekly.release_date.map(previous)
    weekly["window_days"] = (weekly.release_date - weekly.window_start).dt.days
    weekly["cohort"] = np.where(
        weekly.release_date.lt("2023-01-01"), "train_2015_2022", "test_2023_2026"
    )
    weekly["coverage_state"] = np.select(
        [
            weekly.indexed.eq(0),
            weekly.candidates.eq(0),
            weekly.attempted.eq(0),
            weekly.attempt_fraction.lt(0.8),
        ],
        ["no_index_rows", "no_selected_rows", "not_attempted", "partial_attempts"],
        default="at_least_80pct_attempted",
    )
    tables["weekly"] = weekly
    tables["release_coverage"] = (
        weekly.groupby(["region", "cohort"])
        .agg(
            releases=("release_date", "size"),
            weeks_with_index=("indexed", lambda x: x.gt(0).sum()),
            weeks_attempted=("attempted", lambda x: x.gt(0).sum()),
            weeks_with_text=("assembled", lambda x: x.gt(0).sum()),
            weeks_80pct=("attempt_fraction", lambda x: x.ge(0.8).sum()),
            articles=("assembled", "sum"),
        )
        .reset_index()
    )
    tables["count_correlations"] = count_correlations(weekly)
    tables["volume_benchmark"] = volume_benchmark(weekly)
    drift = []
    for region, frame in tables["source_year"].groupby("region"):
        mix = frame.pivot_table(
            index="year", columns="host", values="rows", fill_value=0
        )
        for before, after in zip(mix.index[:-1], mix.index[1:], strict=True):
            p, q = mix.loc[before], mix.loc[after]
            p, q = p / p.sum(), q / q.sum()
            drift.append(
                {
                    "region": region,
                    "year_before": before,
                    "year_after": after,
                    "total_variation": (p - q).abs().sum() / 2,
                    "rows_before": int(frame.loc[frame.year.eq(before), "rows"].sum()),
                    "rows_after": int(frame.loc[frame.year.eq(after), "rows"].sum()),
                }
            )
    tables["source_drift"] = pd.DataFrame(drift)
    titles = (
        a.loc[
            a.title_normalized.ne(""),
            ["title_normalized", "story_id", "text_hash", "region", "date", "title"],
        ]
        .groupby("title_normalized")
        .agg(
            rows=("story_id", "size"),
            texts=("text_hash", "nunique"),
            regions=("region", "nunique"),
            first=("date", "min"),
            last=("date", "max"),
            example=("title", "first"),
        )
    )
    tables["repeated_titles"] = (
        titles[titles.rows.gt(1)].sort_values("rows", ascending=False).reset_index()
    )
    review = []
    review_flags = [
        "region",
        "probability",
        "date_ok",
        "content_warning",
        "date_disagreement_gt7d",
        "topic_inventory_reporting",
        "chars",
    ]
    for _region, frame in a[review_flags].groupby("region"):
        for reason, part in [
            ("near_threshold", frame.nsmallest(3, "probability")),
            ("high_score", frame.nlargest(2, "probability")),
            ("random", frame.sample(min(3, len(frame)), random_state=SEED)),
            ("invalid_date", frame[~frame.date_ok].head(3)),
            ("content_warning", frame[frame.content_warning].head(2)),
            ("date_disagreement", frame[frame.date_disagreement_gt7d].head(2)),
            ("inventory_screen", frame[frame.topic_inventory_reporting].head(2)),
            ("long_text", frame.nlargest(1, "chars")),
        ]:
            selected = a.loc[part.index].copy()
            selected["review_reason"] = reason
            review.append(selected)
    sample = pd.concat(review).drop_duplicates(["region", "story_id"])
    sample = sample[
        [
            "region",
            "story_id",
            "language",
            "date",
            "publish_date",
            "date_source",
            "title",
            "url",
            "probability",
            "chars",
            "review_reason",
            "excerpt",
        ]
    ].copy()
    for label in [
        "human_oil_relevance",
        "human_event_type",
        "human_date_valid",
        "human_text_quality",
        "human_notes",
    ]:
        sample[label] = ""
    tables["review_sample"] = sample
    return tables


def make_figures(tables: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    """Build publication-sized, offline figures from the measured tables."""
    u.setup_style()
    figs: list[tuple[str, str]] = []

    def save(name: str, caption: str) -> None:
        plt.savefig(REPORT / f"{name}.png", dpi=160)
        plt.close()
        figs.append((name, caption))

    r = tables["regions"].set_index("region")
    active = r[r.assembled.gt(0)].sort_values("index_rows", ascending=False)
    x = np.arange(len(active))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for col, label in [
        ("index_rows", "Индекс"),
        ("candidates", "Отобрано по заголовку"),
        ("attempted", "Есть попытка"),
        ("assembled", "В корпусе"),
    ]:
        axes[0].plot(x, active[col], "o-", ms=3, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Число строк, логарифмическая шкала")
    axes[0].legend(ncol=2)
    for col, label, color in [
        ("attempted", "Попытки / кандидаты", u.SERIES[0]),
        ("assembled", "Корпус / кандидаты", u.SERIES[1]),
    ]:
        axes[1].plot(x, active[col] / active.candidates, "o-", label=label, color=color)
    axes[1].set_xticks(x, active.index.str.upper(), rotation=45)
    axes[1].set_ylabel("Доля отобранных публикаций")
    axes[1].legend()
    axes[0].set_title("Сколько публикаций проходит каждую стадию сбора")
    fig.tight_layout()
    save(
        "01_pipeline",
        "Коллекции упорядочены по объёму индекса. При полном обходе кандидатов линии отбора и попыток совпадают. Попытки включают ошибки загрузки.",
    )

    m = tables["monthly"].copy()
    ordered = active.index.tolist()
    fig, axes = plt.subplots(3, 1, figsize=(15, 19), sharex=True, sharey=True)
    for ax, col, title in zip(
        axes,
        ["indexed", "attempted", "assembled"],
        [
            "Индекс по дате Media Cloud",
            "Попытки по дате Media Cloud",
            "Корпус по итоговой дате публикации",
        ],
        strict=True,
    ):
        mat = m.pivot(index="region", columns="month", values=col).reindex(ordered)
        image = ax.imshow(
            np.log10(mat.to_numpy() + 1),
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=np.log10(
                1 + m[["indexed", "attempted", "assembled"]].to_numpy().max()
            ),
            cmap=u.SEQUENTIAL,
        )
        ax.set_yticks(range(len(ordered)), [s.upper() for s in ordered], fontsize=7)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.018, label="log₁₀(строки + 1)")
    ticks = [i for i, month in enumerate(MONTHS) if month.month == 1]
    axes[-1].set_xticks(ticks, [str(MONTHS[i].year) for i in ticks])
    fig.tight_layout()
    save(
        "02_coverage",
        "Одинаковая цветовая шкала для трёх стадий. Светлые ячейки соответствуют месяцам с меньшим числом записей.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    failures = (
        tables["failures"].groupby("failure")["count"].sum().sort_values().tail(10)
    )
    failures = failures.rename(index=FAILURE_NAMES)
    failures.plot.barh(ax=axes[0], color=u.SERIES[1])
    axes[0].set_title("Причины неудачного извлечения")
    axes[0].set_xlabel("Число публикаций")
    ratio = active.assembled / active.attempted
    ratio.sort_values().plot.barh(ax=axes[1], color=u.SERIES[0])
    axes[1].set_title("Доля попыток, давших статью")
    axes[1].set_xlabel("Доля после очистки при сборке")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("Коллекция")
    fig.tight_layout()
    save(
        "03_fetch_failures",
        "Для каждой публикации учтена последняя сохранённая попытка. Доля справа учитывает потери при извлечении и сборке.",
    )

    a = tables["articles"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hist = tables["score_hist"].groupby("bin_mid")["count"].sum()
    axes[0].bar(hist.index, hist.values, width=0.045, color=u.SERIES[0])
    meta = json.loads((ROOT / "data/models/news/title/meta.json").read_text())
    axes[0].axvline(
        meta["thresholds"]["keep"], color=u.SERIES[1], label="Порог включения"
    )
    axes[0].set_title("Распределение оценок заголовков")
    axes[0].set_xlabel("Оценка классификатора")
    axes[0].legend()
    lang = tables["languages"].set_index("language").rows
    lang.iloc[:20].sort_values().plot.barh(ax=axes[1], color=u.SERIES[2])
    axes[1].set_title("20 крупнейших языковых групп")
    axes[1].set_ylabel("Язык")
    fig.tight_layout()
    save(
        "04_scores_languages",
        "Слева показаны оценки всех обработанных заголовков, справа — языки готовых статей.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    active.sort_values("top1_share").top1_share.plot.barh(ax=axes[0], color=u.SERIES[0])
    axes[0].set_title("Доля крупнейшего хоста внутри коллекции")
    axes[0].set_xlabel("Доля строк корпуса")
    mat = (
        tables["source_year"].groupby(["year", "host"]).rows.sum().unstack(fill_value=0)
    )
    top = mat.sum().nlargest(8).index
    mix = mat[top].div(mat.sum(axis=1), axis=0)
    mix["другие хосты"] = 1 - mix.sum(axis=1)
    mix.plot.area(ax=axes[1], linewidth=0, color=[*u.SERIES, u.NEUTRAL])
    axes[1].set_title("Доли восьми крупнейших хостов по годам")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Коллекция")
    axes[1].set_xlabel("Год")
    fig.tight_layout()
    save(
        "05_sources",
        "Состав корпуса по хостам зависит от индекса и доступности архивных страниц. В 2026 году учтены январь–октябрь.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    axes[0].hist(
        a.chars.clip(lower=1),
        bins=np.geomspace(250, max(a.chars.max(), 301), 65),
        color=u.SERIES[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Длина текста в символах")
    axes[0].axvline(300, color=u.SERIES[1])
    dup = active.local_duplicate_rows / active.assembled
    dup.sort_values().plot.barh(ax=axes[1], color=u.SERIES[1])
    axes[1].set_title("Доля локальных дублей")
    axes[1].set_ylabel("Коллекция")
    shift = a.date_shift_days.clip(-365, 365)
    axes[2].hist(shift, bins=60, color=u.SERIES[2])
    axes[2].set_yscale("log")
    axes[2].set_title("Итоговая дата минус дата индекса")
    axes[2].set_xlabel("Дни, значения обрезаны на ±365")
    fig.tight_layout()
    save(
        "06_quality",
        "Локальные дубли определены по хешу нормализованного полного текста. Сдвиг даты рассчитан по каждой собранной строке.",
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    topic = tables["topics"].set_index("region").reindex(ordered)
    image = ax.imshow(
        topic.to_numpy(), cmap=u.SEQUENTIAL, aspect="auto", vmin=0, vmax=0.5
    )
    ax.set_yticks(range(len(ordered)), [s.upper() for s in ordered])
    ax.set_xticks(
        range(len(TOPICS)), [TOPIC_NAMES[k] for k in TOPICS], rotation=35, ha="right"
    )
    ax.set_title("Какие темы встречаются в заголовках")
    fig.colorbar(image, ax=ax, label="Доля заголовков, верх шкалы 50%")
    fig.tight_layout()
    save(
        "07_topics",
        "Темы пересекаются. Доля совпадений зависит в том числе от языка словаря и состава издателей.",
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    entity = tables["entities"].set_index("region").reindex(ordered)
    image = ax.imshow(np.log10(entity.to_numpy() + 1), cmap=u.SEQUENTIAL, aspect="auto")
    ax.set_yticks(range(len(ordered)), [s.upper() for s in ordered])
    ax.set_xticks(
        range(len(ENTITIES)),
        [ENTITY_NAMES[k] for k in ENTITIES],
        rotation=30,
        ha="right",
    )
    ax.set_title("Упоминания компаний, мест и отраслевых обозначений")
    fig.colorbar(image, ax=ax, label="log₁₀(совпадения + 1)")
    fig.tight_layout()
    save(
        "08_entities",
        "Число совпадений в заголовках по стране источника. Конкретный объект и место события требуют отдельной проверки.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    cov = tables["release_coverage"]
    for ax, cohort in zip(axes, ["train_2015_2022", "test_2023_2026"], strict=True):
        d = cov[cov.cohort.eq(cohort)].set_index("region").reindex(ordered)
        yy = np.arange(len(d))
        ax.barh(
            yy,
            d.weeks_with_text / d.releases,
            color=u.SERIES[0],
            label="Есть собранная статья",
        )
        ax.barh(
            yy,
            d.weeks_80pct / d.releases,
            color=u.SERIES[2],
            height=0.35,
            label="Попытки у ≥80% кандидатов",
        )
        ax.set_yticks(yy, [s.upper() for s in ordered])
        ax.set_xlim(0, 1)
        ax.set_title("2015–2022" if cohort.startswith("train") else "2023–2026")
        ax.set_xlabel("Доля календарных окон EIA")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(
        "09_release_coverage",
        "Публикация отнесена к первому наблюдаемому релизу строго после дня публикации. Знаменатель включает все окна соответствующего периода.",
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    q = tables["quality"].set_index("region").reindex(ordered)
    for col, label in [
        ("assembled", "Собрано"),
        ("quality_pass", "Прошли проверки качества"),
        ("unique_screened", "Уникальный проверенный хеш"),
        (
            "physical_candidates",
            "Физические темы без флага отчётности",
        ),
    ]:
        ax.plot(x, q[col], "o-", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [s.upper() for s in ordered], rotation=45)
    ax.set_ylabel("Число строк, логарифмическая шкала")
    ax.set_title("Статьи после проверки качества, дедупликации и отбора по теме")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(
        "10_screening",
        "Один экземпляр текста закрепляется за самой ранней допустимой записью. Последняя линия показывает заголовки для проверки событий.",
    )
    cor = tables["count_correlations"].sort_values(["rho", "region"])
    fig, ax = plt.subplots(figsize=(14, 5))
    countries = cor.region.drop_duplicates().tolist()
    x_pos = {name: i for i, name in enumerate(countries)}
    legend = {
        "both": ("o", "Оба порога дают одни окна"),
        "any_attempt": ("x", "Есть попытка"),
        "80pct_attempted": ("+", "Попытки у ≥80%"),
    }
    for screen, rows in cor.groupby("screen"):
        marker, label = legend[screen]
        ax.scatter(
            rows.region.map(x_pos),
            rows.rho,
            marker=marker,
            label=label,
            color=u.SERIES[0],
        )
    passed = cor.loc[cor.p_bh.lt(0.05)]
    if len(passed):
        ax.scatter(
            passed.region.map(x_pos),
            passed.rho,
            s=90,
            facecolors="none",
            edgecolors=u.SERIES[1],
            label="p после поправки < 0,05",
        )
    ax.axhline(0, color=u.INK_MUTED, linewidth=1)
    ax.set_xticks(range(len(countries)), [v.upper() for v in countries], rotation=55)
    ax.set_ylabel("Корреляция Спирмена, ρ")
    ax.set_title("Число статей за неделю и сюрприз запасов США")
    if len(cor):
        ax.legend()
    fig.tight_layout()
    save(
        "11_inventory_surprise",
        "Каждая точка показывает корреляцию внутри одной коллекции по полным семидневным окнам. Порог значимости учитывает все сравнения.",
    )
    return figs


def validate_tables(t: dict[str, pd.DataFrame]) -> list[str]:
    """Check accounting and time boundaries against the complete local snapshot."""
    checks = []
    r, a, w = t["regions"], t["articles"], t["weekly"]
    month = t["monthly"]
    conditions = {
        "Число статей совпадает с суммой строк по странам": len(a) == r.assembled.sum(),
        "Каждая пара страны и story_id встречается в корпусе один раз": not a.duplicated(
            ["region", "story_id"]
        ).any(),
        "Каждая коллекция представлена всеми месяцами периода": (
            len(month) == len(r) * len(MONTHS)
            and not month.duplicated(["region", "month"]).any()
            and month.month.isin(MONTHS).all()
            and set(month.region) == set(r.region)
        ),
        "Сохранённые попытки соответствуют отобранным идентификаторам": r.attempted.eq(
            r.attempted_candidates
        ).all(),
        "Число попыток ограничено числом кандидатов": r.attempted.le(
            r.candidates
        ).all(),
        "Число статей ограничено числом успешных извлечений и попыток": (
            r.assembled.le(r.extracted) & r.extracted.le(r.attempted)
        ).all(),
        "Неудачные и успешные попытки в сумме дают все попытки": t["failures"][
            "count"
        ].sum()
        + r.extracted.sum()
        == r.attempted.sum(),
        "В каждом окне число попыток ограничено числом кандидатов": w.attempted.le(
            w.candidates
        ).all(),
        "В окнах без попыток число наблюдаемых текстов остаётся пропуском": w.loc[
            w.attempted.eq(0), "observed_text_count"
        ]
        .isna()
        .all(),
        "Уникальные тексты удовлетворяют фильтрам качества": a.loc[
            a.screened, "quality_ok"
        ].all(),
        "Хеши отобранных для анализа текстов уникальны": a.loc[
            a.screened, "text_hash"
        ].is_unique,
        "День публикации предшествует назначенному релизу": (
            a.loc[a.next_release.notna(), "date"].dt.normalize()
            < a.loc[a.next_release.notna(), "next_release"]
        ).all(),
        "Число словарных кандидатов в окне ограничено числом уникальных текстов": w.physical_candidates.le(
            w.unique_screened
        ).all(),
        "Сюрприз равен первой публикации факта минус консенсус": np.allclose(
            u.release_frame().surprise_mmbbl,
            u.release_frame().actual_mmbbl - u.release_frame().consensus_mmbbl,
        ),
    }
    for name, passed in conditions.items():
        if not passed:
            raise ValueError(name)
        checks.append(name)
    return checks


def number(value, digits: int = 0) -> str:
    """Русская запись числа для отчётов."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def pct(part, total) -> str:
    if not total:
        return "—"
    rate = 100 * part / total
    return number(rate, 3 if 0 < rate < 0.05 else 1) + "%"


def report_summary(t: dict[str, pd.DataFrame], checks: list[str]) -> dict:
    a, r, w = t["articles"], t["regions"], t["weekly"]
    active = r.loc[r.assembled.gt(0)].copy()
    months = t["monthly"].groupby("region")[["index_downloaded", "index_scored"]].all()
    by_region = r.set_index("region")
    completed = months.all(axis=1) & by_region.attempted_candidates.eq(
        by_region.candidates
    )
    valid_dates = a.loc[a.date_ok, ["text_hash", "date"]]
    sides = valid_dates.date.ge("2023-01-01").groupby(valid_dates.text_hash).nunique()
    manifest = t["manifest"].sort_values("path")
    digest = hashlib.sha256(manifest.to_json(orient="records").encode()).hexdigest()
    changed = [
        name
        for name, size, stamp in manifest[["path", "bytes", "mtime_ns"]].itertuples(
            index=False, name=None
        )
        if not (ROOT / name).exists()
        or (ROOT / name).stat().st_size != size
        or (ROOT / name).stat().st_mtime_ns != stamp
    ]
    if changed:
        msg = f"Во время EDA изменились входные файлы ({len(changed)}), начиная с {changed[0]}"
        raise ValueError(msg)
    cor = t["count_correlations"]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "scope_start": str(START.date()),
        "scope_end": str(END.date()),
        "configured_regions": len(r),
        "complete_regions": int(completed.sum()),
        "regions_with_text": len(active),
        "indexed_rows": int(r.index_rows.sum()),
        "candidate_rows": int(r.candidates.sum()),
        "attempted_rows": int(r.attempted.sum()),
        "matched_candidate_rows": int(r.attempted_candidates.sum()),
        "extracted_rows": int(r.extracted.sum()),
        "failed_rows": int(t["failures"]["count"].sum()),
        "assembled_rows": len(a),
        "quality_rows": int(a.quality_ok.sum()),
        "screened_unique_texts": int(a.screened.sum()),
        "physical_title_candidates": int(a.candidate_event.sum()),
        "languages": int(a.language.nunique()),
        "hosts": int(a.host.nunique()),
        "local_duplicate_rows": int(a.duplicate_of.notna().sum()),
        "all_duplicate_excess": int(len(a) - a.text_hash.nunique()),
        "cross_region_duplicate_clusters": int(
            t["duplicate_clusters"].regions.gt(1).sum()
        ),
        "cross_split_text_clusters": int(sides.gt(1).sum()),
        "invalid_date_rows": int((~a.date_ok).sum()),
        "future_date_rows": int(a.date.gt(END).sum()),
        "exact_time_rows": int(a.exact_time_present.sum()),
        "content_warning_rows": int(a.content_warning.sum()),
        "date_disagreement_rows": int(a.date_disagreement_gt7d.sum()),
        "inventory_title_flags": int(a.topic_inventory_reporting.sum()),
        "review_rows": len(t["review_sample"]),
        "full_seven_day_windows": full_weeks(w).release_date.nunique(),
        "country_comparisons": len(cor),
        "significant_after_bh": int(cor.p_bh.lt(0.05).sum()),
        "volume_benchmark": t["volume_benchmark"].to_dict("records"),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(
            (ROOT / "data/models/news/title/meta.json").read_bytes()
        ).hexdigest(),
        "regions_config_sha256": hashlib.sha256(
            (ROOT / "src/news/regions.toml").read_bytes()
        ).hexdigest(),
        "manifest_sha256": digest,
        "input_file_count": len(manifest),
        "inputs_changed_during_analysis": changed,
        "checks_passed": checks,
    }


def display_value(v) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, (int, np.integer)):
        return number(v)
    if isinstance(v, (float, np.floating)):
        return number(v, 4).rstrip("0").rstrip(",")
    return str(v)


class NewsReport:
    """Одна последовательность разделов для Markdown и автономного HTML."""

    def __init__(self, figs: list[tuple[str, str]]) -> None:
        self.figs = figs
        self.title = "Разведочный анализ новостного корпуса"
        self.md = [
            f"# {self.title}",
            (
                "[Ноутбук](../eda/notebooks/08_news_corpus.ipynb) · [Автономный HTML](../eda/reports/news/news_eda.html) · "
                "[Сводка JSON](../eda/reports/news/summary.json) · [Расчёты](../eda/news_eda.py) · [EDA неновостных источников](eda.md)"
            ),
        ]
        self.sections = [f"<h1>{self.title}</h1>"]

    def heading(self, label: str) -> None:
        self.md.append(f"## {label}")
        self.sections.append(f"<h2>{html.escape(label)}</h2>")

    def paragraph(self, value: str) -> None:
        self.md.append(value)
        self.sections.append(f"<p>{html.escape(value)}</p>")

    def table(self, frame: pd.DataFrame, labels: list[str]) -> None:
        grid = frame.set_axis(labels, axis=1).map(display_value)
        lines = [
            labels,
            ["---"] * len(labels),
            *grid.itertuples(index=False, name=None),
        ]
        self.md.append(
            "\n".join(
                "| "
                + " | ".join(html.escape(v).replace("|", "&#124;") for v in line)
                + " |"
                for line in lines
            )
        )
        self.sections.append(
            '<div class="table-wrap">' + grid.to_html(index=False, border=0) + "</div>"
        )

    def figure(self, index: int) -> None:
        name, caption = self.figs[index]
        self.md.append(f"![{caption}](../eda/reports/news/{name}.png)")
        data = base64.b64encode((REPORT / f"{name}.png").read_bytes()).decode()
        self.sections.append(
            f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )

    def code(self, value: str) -> None:
        self.md.append(f"```bash\n{value}\n```")
        self.sections.append(f"<pre><code>{html.escape(value)}</code></pre>")


def make_report(
    t: dict[str, pd.DataFrame], figs: list[tuple[str, str]], checks: list[str]
) -> dict:
    """Записать Markdown и автономный HTML по одним и тем же таблицам и выводам."""
    a, r, w = t["articles"], t["regions"], t["weekly"]
    active = r.loc[r.assembled.gt(0)]
    q = t["quality"]
    cor = t["count_correlations"]
    bench = t["volume_benchmark"]
    failed = (
        t["failures"].groupby("failure")["count"].sum().sort_values(ascending=False)
    )
    meta = json.loads((ROOT / "data/models/news/title/meta.json").read_text())
    summary = report_summary(t, checks)
    report = NewsReport(figs)
    indexed, selected, attempted, extracted, assembled = (
        int(r[k].sum())
        for k in ["index_rows", "candidates", "attempted", "extracted", "assembled"]
    )
    clean, unique, events = (
        summary[k]
        for k in ["quality_rows", "screened_unique_texts", "physical_title_candidates"]
    )
    report.paragraph(
        f"Период публикаций с {START:%d.%m.%Y} по {END:%d.%m.%Y}. Пересчёт от {summary['generated_at_utc'][:10]} UTC охватывает {len(active)} стран с текстами. "
        f"Объём корпуса — {number(assembled)} строк. Число уникальных текстов после проверок качества — {number(unique)}. Объём выборки для проверки физических событий по заголовкам — {number(events)}."
    )
    report.paragraph(
        "В отчёт от 10 ноября 2026 года входили 460 460 статей из 31 коллекции. Проверяем, какие архивы удалось восстановить и связан ли недельный объём публикаций с сюрпризом запасов США. "
        f"При уровне 5% после поправки на множественные сравнения статистически значимых результатов — {summary['significant_after_bh']} из {summary['country_comparisons']}."
    )

    report.heading("1. Источники и учёт стадий сбора")
    report.paragraph(
        f"Для {summary['complete_regions']} из {len(r)} настроенных коллекций есть индекс и результаты фильтра по заголовку за все {len(MONTHS)} месяца, а у каждого текущего кандидата найдена сохранённая попытка по story_id. "
        "Один материал может входить в несколько коллекций. До глобальной дедупликации строки суммируются по странам."
    )
    report.table(
        pd.DataFrame(
            [
                ("Индекс Media Cloud", indexed),
                ("Отобрано классификатором заголовков", selected),
                ("Кандидаты с сохранённой попыткой", attempted),
                ("Успешно извлечён текст", extracted),
                ("Строки собранного корпуса", assembled),
                ("Прошли проверки качества", clean),
                ("Уникальные тексты после проверок", unique),
            ]
        ),
        ["Стадия", "Число записей"],
    )
    report.paragraph(
        f"Попытки охватили {pct(attempted, selected)} кандидатов. До итоговой статьи дошли {pct(assembled, attempted)} попыток. "
        f"На извлечении потеряно {number(failed.sum())} публикаций, ещё {number(extracted - assembled)} успешно извлечённых записей отсеяны при сборке."
    )
    report.figure(0)

    report.heading("2. Географическое и временное покрытие")
    report.paragraph(
        "Код страны обозначает коллекцию издателей. Ирак, Оман, Сирия и Йемен собирались целиком в пределах выбранных коллекций Media Cloud. Для остальных стран индекс ограничивал предварительный поиск по тематическим терминам. Поэтому доля отобранных заголовков зависит и от способа сбора."
    )
    by_country = r.sort_values("index_rows", ascending=False).copy()
    by_country["yield"] = [
        pct(n, d)
        for n, d in zip(by_country.assembled, by_country.attempted, strict=True)
    ]
    report.table(
        by_country[
            ["region", "index_rows", "candidates", "attempted", "assembled", "yield"]
        ],
        ["Страна", "Индекс", "Кандидаты", "Попытки", "Статьи", "Статьи / попытки"],
    )
    m = t["monthly"]
    populated = m.loc[m.assembled.gt(0)].groupby("region").month.nunique()
    report.paragraph(
        f"Хотя бы одна статья найдена во всех {len(MONTHS)} месяцах для {int(populated.eq(len(MONTHS)).sum())} коллекций. "
        f"Из {number(len(m))} пар «страна–месяц» статьи есть в {number(m.assembled.gt(0).sum())}. "
        "Помесячный индекс основан на дате Media Cloud, корпус — на итоговой дате статьи. Границы покрытия учитывают и пустые месяцы внутри многомесячных файлов."
    )
    era = pd.cut(
        m.month.dt.year,
        bins=[2014, 2018, 2022, 2025, 2026],
        labels=["2015–2018", "2019–2022", "2023–2025", "2026, январь–октябрь"],
    )
    by_era = (
        m.groupby(era, observed=True)[["attempted", "extracted"]].sum().reset_index()
    )
    by_era["yield"] = [
        pct(n, d) for n, d in zip(by_era.extracted, by_era.attempted, strict=True)
    ]
    report.paragraph(
        "Как часто архивная ссылка даёт текст, видно при группировке по исходной дате публикации. Обе величины ниже относятся к одной и той же когорте кандидатов. Вместе с возрастом ссылок меняется состав издателей, поэтому таблица сама по себе не устанавливает причину различий."
    )
    report.table(
        by_era, ["Годы публикации", "Попытки", "Извлечены тексты", "Успех извлечения"]
    )
    report.figure(1)

    report.heading("3. Ошибки загрузки и смещение выборки")
    report.paragraph(
        "При повторной загрузке берём последний результат по времени fetched_at внутри страны. HTTP 200 в таблице ошибок означает, что сервер ответил, но извлечь пригодный текст не удалось. HTTP-статус и причина обработки учитываются для одной и той же последней попытки."
    )
    f = (
        failed.rename(index=FAILURE_NAMES)
        .head(10)
        .rename_axis("reason")
        .reset_index(name="n")
    )
    f["share"] = [pct(v, failed.sum()) for v in f.n]
    report.table(f, ["Причина", "Неудачные попытки", "Доля всех неудач"])
    sources = (
        t["sources"]
        .groupby(["region", "source", "stage"])["count"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    sources["yield"] = [
        pct(n, d) for n, d in zip(sources.assembled, sources.attempted, strict=True)
    ]
    report.paragraph(
        "Результаты десяти крупнейших издателей по числу кандидатов показывают, как различается восстановление текстов внутри коллекций. Сравнение долей учитывает дополнительную очистку после извлечения."
    )
    report.table(
        sources.nlargest(10, "candidate")[
            ["region", "source", "candidate", "attempted", "assembled", "yield"]
        ],
        [
            "Коллекция",
            "Издатель Media Cloud",
            "Кандидаты",
            "Попытки",
            "Статьи",
            "Статьи / попытки",
        ],
    )
    if "sy" in set(active.region):
        sy = active.set_index("region").loc["sy"]
        report.paragraph(
            f"У Сирии в корпус вошли {number(sy.assembled)} из {number(sy.attempted)} обработанных кандидатов ({pct(sy.assembled, sy.attempted)}). "
            "Низкое число текстов в такой коллекции особенно чувствительно к доступности сайтов."
        )
    report.figure(2)

    report.heading("4. Языки и фильтр релевантности")
    lang = t["languages"].head(12).copy()
    lang["share"] = [pct(v, assembled) for v in lang.rows]
    report.paragraph(
        f"В корпусе {summary['languages']} языков. Таблица показывает двенадцать крупнейших групп. Один язык может встречаться в нескольких коллекциях."
    )
    report.table(
        lang[["language", "rows", "share", "regions", "median_chars"]],
        ["Язык", "Статьи", "Доля", "Коллекции", "Медиана длины, символов"],
    )
    met = meta["metrics"]
    report.paragraph(
        f"Классификатор {meta['embedder']} обучен на {number(meta['trained_on'])} примерах. Пороги keep и drop равны {number(meta['thresholds']['keep'], 6)} и {number(meta['thresholds']['drop'], 6)}. "
        f"Пограничных решений в сохранённых оценках — {number(r.borderline.fillna(0).sum())}. "
        f"При разработке PR-AUC составила {number(met['pr_auc'], 3)}, в среднем по языкам — {number(met['pr_auc_macro_language'], 3)}, с весами страт — {number(met['pr_auc_weighted'], 3)}, на временной проверке — {number(met['pr_auc_temporal'], 3)}. Для оценки точности на текущем корпусе нужна независимая разметка."
    )
    report.figure(3)

    report.heading("5. Концентрация источников и дрейф")
    report.paragraph(
        f"В адресах статей обнаружено {number(summary['hosts'])} хостов. Эффективное число источников равно 1 / Σ доля². Пять наиболее концентрированных коллекций и коллекция США показывают диапазон этого показателя."
    )
    conc = (
        pd.concat([active.nlargest(5, "top1_share"), active[active.region.eq("us")]])
        .drop_duplicates("region")
        .copy()
    )
    for col in ["top1_share", "top5_share"]:
        conc[col] = [pct(v, 1) for v in conc[col]]
    report.table(
        conc[["region", "sources", "top1_share", "top5_share", "effective_sources"]],
        ["Коллекция", "Хосты", "Доля лидера", "Доля пяти лидеров", "Эффективное число"],
    )
    drift = (
        t["source_drift"]
        .query(
            "rows_before >= 100 and rows_after >= 100 and year_after == year_before + 1"
        )
        .nlargest(3, "total_variation")
    )
    report.paragraph(
        "Наибольшие изменения состава издателей между соседними годами приведены для пар с ≥100 статьями в каждом году. Полная вариация равна половине суммы абсолютных изменений долей хостов. При полном совпадении распределений она равна нулю, при отсутствии общих хостов — единице. 2026 год заканчивается октябрём."
    )
    report.table(
        drift[["region", "year_before", "year_after", "total_variation"]].assign(
            year_before=drift.year_before.astype(str),
            year_after=drift.year_after.astype(str),
        ),
        ["Коллекция", "Первый год", "Второй год", "Полная вариация"],
    )
    report.figure(4)

    report.heading("6. Качество текстов, дубли и даты")
    report.paragraph(
        "Проверка требует непустого нормализованного заголовка, хеша текста, длины 300–100 000 символов и даты в пределах периода с источником, отличным от unreliable. Предупреждение о содержимом срабатывает на известную заглушку в первых 600 символах или на текст с ≥10 копиями, ≥5 разными заголовками и размахом дат более 30 дней."
    )
    report.table(
        pd.DataFrame(
            [
                (
                    "Ненадёжная дата или дата за пределами периода",
                    summary["invalid_date_rows"],
                ),
                ("В том числе после конца периода", summary["future_date_rows"]),
                ("Длина свыше 100 000 символов", q.very_long.sum()),
                ("Предупреждение о содержимом", summary["content_warning_rows"]),
                (
                    "Итоговая и индексная даты расходятся более чем на неделю",
                    summary["date_disagreement_rows"],
                ),
                ("Ненулевое время суток в итоговой дате", summary["exact_time_rows"]),
                (
                    "Помечено локальных дублей в коллекциях",
                    summary["local_duplicate_rows"],
                ),
                (
                    "Избыточные строки по хешу во всём корпусе",
                    summary["all_duplicate_excess"],
                ),
                (
                    "Кластеры с текстом в нескольких коллекциях",
                    summary["cross_region_duplicate_clusters"],
                ),
                (
                    "Кластеры с допустимыми датами по обе стороны границы 2023 года",
                    summary["cross_split_text_clusters"],
                ),
            ]
        ),
        ["Проверка", "Строк или кластеров"],
    )
    report.paragraph(
        f"Если оставить один экземпляр каждого хеша до фильтра качества, число строк сократится на {pct(summary['all_duplicate_excess'], assembled)}. "
        "Для графа можно хранить один узел документа и связи со всеми URL, издателями и коллекциями. Это сохраняет историю перепечаток и ограничивает повторный учёт одного сообщения."
    )
    report.table(
        a.date_source.value_counts().rename_axis("источник").reset_index(name="строк"),
        ["Источник итоговой даты", "Строки"],
    )
    report.paragraph(
        "Расхождение с датой индекса может быть обоснованным исправлением. В таблице ниже приведены наибольшие абсолютные сдвиги среди дат, восстановленных из URL. Числовой идентификатор или часть времени в адресе могут выглядеть как год, месяц и день. Эти случаи нужно сверять с исходной страницей."
    )
    shifts = a.loc[
        a.date_source.eq("url") & a.date_disagreement_gt7d,
        ["region", "host", "date", "publish_date", "date_shift_days"],
    ].copy()
    shifts = shifts.loc[shifts.date_shift_days.abs().nlargest(3).index]
    for col in ["date", "publish_date"]:
        shifts[col] = shifts[col].dt.strftime("%Y-%m-%d")
    report.table(
        shifts, ["Коллекция", "Хост", "Дата из URL", "Дата индекса", "Сдвиг, дни"]
    )
    report.figure(5)

    report.heading("7. Темы, кандидаты событий и сущности")
    topic_cols = ["topic_" + k for k in TOPICS]
    counts = (
        a[topic_cols]
        .sum()
        .set_axis([TOPIC_NAMES[k] for k in TOPICS])
        .sort_values(ascending=False)
    )
    report.table(counts.reset_index(), ["Тема по заголовку", "Строки с совпадением"])
    report.paragraph(
        f"Тематические группы пересекаются. Среди уникальных текстов, прошедших проверки, у {number(events)} ({pct(events, unique)}) есть совпадение по добыче, переработке, перевозкам, трубопроводам, остановкам или погоде без совпадения по отчётности о запасах. "
        f"Из всех строк корпуса флаг отчётности получили {number(summary['inventory_title_flags'])}. Это кандидаты для проверки утечки уже опубликованных данных EIA или API."
    )
    report.paragraph(
        "Термины отражают язык словаря. Например, правила включают английские и арабские основы, а для китайского отсутствуют иероглифические варианты. Совпадения по странам нельзя напрямую переводить в частоту физических событий."
    )
    report.figure(6)
    entities = (
        t["entities"][["entity_" + k for k in ENTITIES]]
        .sum()
        .set_axis([ENTITY_NAMES[k] for k in ENTITIES])
        .sort_values(ascending=False)
    )
    report.table(
        entities.reset_index(),
        ["Группа компаний, мест или обозначений", "Заголовки с совпадением"],
    )
    report.paragraph(
        "Для связи упоминания с инфраструктурой нужны идентификатор объекта, место и время события, направление воздействия, мощность с единицами и длительность. Одно название компании может относиться к разным объектам."
    )
    report.figure(7)
    report.figure(9)

    report.heading("8. Привязка к релизам EIA")
    calendar = w.drop_duplicates("release_date").sort_values("release_date")
    report.paragraph(
        f"В календаре {len(calendar)} наблюдаемых релизов. Публикация относится к первому релизу строго после дня её публикации. Все статьи дня релиза переходят в следующее окно. "
        "При точности даты до дня время доступности статьи относительно публикации EIA внутри суток неизвестно. Календарь наследует пропуски 2022 и 2023 годов и задержку релиза в июне 2022 года."
    )
    cov = t["release_coverage"].copy()
    cov["cohort"] = cov.cohort.map(
        {"train_2015_2022": "2015–2022", "test_2023_2026": "2023–2026"}
    )
    example_countries = ["us", "de", "in", "fr", "sy", "cn"]
    sample_cov = pd.concat(
        [cov.loc[cov.region.eq(c)].sort_values("cohort") for c in example_countries]
    )
    report.table(
        sample_cov[
            [
                "region",
                "cohort",
                "releases",
                "weeks_with_text",
                "weeks_80pct",
                "articles",
            ]
        ],
        [
            "Коллекция",
            "Период",
            "Релизы",
            "Окна со статьями",
            "Окна с ≥80% попыток",
            "Статьи в окнах",
        ],
    )
    us_train = cov.loc[cov.region.eq("us") & cov.cohort.eq("2015–2022")]
    if not us_train.empty:
        row = us_train.iloc[0]
        report.paragraph(
            f"В первом отчёте в американской коллекции не было текстов для обучения до 2023 года. Теперь статьи есть в {int(row.weeks_with_text)} из {int(row.releases)} таких окон. "
            "Это позволяет проверить модель на более поздних релизах, сохраняя хронологию."
        )
    report.paragraph(
        "Для статистики оставляем только полные семидневные интервалы [предыдущий релиз, текущий релиз) внутри периода сбора. "
        f"Из {int(calendar.window_days.eq(7).sum())} семидневных окон осталось {summary['full_seven_day_windows']}. "
        f"Архив заканчивается {END:%d.%m.%Y}, а последний календарный релиз приходится на {calendar.release_date.max():%d.%m.%Y}. "
        "Первое окно не имеет известной левой границы. При отсутствии попыток наблюдаемое число текстов остаётся пропуском."
    )
    report.figure(8)

    report.heading("9. Количество новостей и сюрприз запасов")
    report.paragraph(
        "Сюрприз запасов равен первой публикации изменения коммерческих запасов нефти США минус консенсус, в млн баррелей. "
        "В каждой коллекции проверяем ранговую корреляцию Спирмена с числом собранных статей на полных семидневных окнах. Требуются ≥30 наблюдений. "
        "Сравниваем окна с любой попыткой и с попытками у ≥80% кандидатов. Совпадающие наборы дат учитываются один раз при поправке Бенджамини–Хохберга."
    )
    if len(cor):
        report.paragraph(
            f""
            f"Диапазон ρ от {number(cor.rho.min(), 3)} до {number(cor.rho.max(), 3)}, минимальное скорректированное p = {number(cor.p_bh.min(), 4)}. "
            "Таблица показывает пять сравнений с наименьшими исходными p."
        )
        results = cor.nsmallest(5, "p").copy()
        results["screen"] = results.screen.map(
            {
                "both": "оба порога",
                "any_attempt": "есть попытка",
                "80pct_attempted": "≥80% попыток",
            }
        )
        report.table(
            results[["region", "screen", "n", "rho", "p", "p_bh"]],
            ["Коллекция", "Окна", "N", "ρ", "p", "p после поправки"],
        )
    report.figure(10)
    report.paragraph(
        "Для отдельной проверки на будущих релизах обучаем методом наименьших квадратов две линейные модели со свободным членом. Первая использует разность API и консенсуса. Вторая добавляет log(1 + число уникальных текстов) и долю этих текстов с физической темой. "
        "Счётчики суммируются по коллекциям после глобальной дедупликации. В обеих моделях одинаковые недели со всеми попытками по отобранным кандидатам, хотя бы одним уникальным текстом и известными API, консенсусом и фактом."
    )
    if not bench.empty:
        models = bench.copy()
        models["model"] = models.model.map(
            {"api": "API − консенсус", "api_news": "API − консенсус и новости"}
        )
        report.table(
            models,
            [
                "Модель",
                "Недель до 2023",
                "Недель с 2023",
                "MAE, млн барр.",
                "RMSE, млн барр.",
            ],
        )
        diff = bench.set_index("model").mae_mmbbl
        delta = diff["api_news"] - diff["api"]
        report.paragraph(
            f"На отложенном периоде с 2023 года добавление двух новостных признаков изменило среднюю абсолютную ошибку на {number(delta, 4)} млн баррелей (новая модель минус базовая). "
            "Это результат одного временного разбиения для двух агрегатов. Шаблоны и перепечатки определены по всему корпусу, включая последующие публикации. Доступность восстановленных версий текста в момент прогноза документально не установлена."
        )

    report.heading("10. Выводы для событийного графа")
    report.paragraph(
        "Самая крупная потеря материала происходит при получении старых страниц. Вместе со счётчиком новостей стоит хранить долю восстановленных публикаций по издателю и месяцу. Это особенно важно для сравнения стран и для периодов, в которых менялся состав источников."
    )
    report.paragraph(
        f"Для ручной проверки сохранена выборка из {number(summary['review_rows'])} публикаций. В неё включены случайные строки, заголовки у порога классификатора, сомнительные даты, возможные заглушки и материалы об отчётах. "
        "Такая схема отбора помогает разбирать ошибки. Чтобы оценить точность по всему корпусу, нужна независимая выборка с известными вероятностями отбора."
    )
    report.paragraph(
        "При подготовке исторической оценки переносите копии одного документа в одну выборку, сохраняйте исходную дату и отдельно фиксируйте подтверждённое время доступности. "
        "Качество извлечения событий, связей с объектами и прогноза спрэда WTI требует отдельной проверки."
    )

    report.heading("11. Воспроизведение и проверки")
    report.paragraph(
        "Запустите из корня репозитория при наличии локальных новостных Parquet и распакованного каталога eda/data/processed из processed.zip."
    )
    report.code(
        ".venv/bin/python eda/news_eda.py\n.venv/bin/python eda/execute_notebook.py eda/notebooks/08_news_corpus.ipynb\n.venv/bin/python -m pytest -q tests/eda/test_news_eda.py"
    )
    report.paragraph(
        "Обычный запуск пересчитывает таблицы, графики, Markdown, HTML и JSON. Флаг --cached использует уже рассчитанные таблицы. Производные Parquet, манифест и review_sample.csv находятся в eda/data/derived/news/. "
        "Для формирования выборки используется seed 6107. Сводка JSON содержит контрольные суммы модуля анализа, метаданных модели и конфигурации регионов."
    )
    report.paragraph(
        f"В манифесте {number(summary['input_file_count'])} входных Parquet, включая три таблицы календаря и данных EIA, API и консенсуса. Размер и время изменения сверены после анализа. "
        f"SHA-256 отсортированного перечня путей, размеров, времён и числа строк — {summary['manifest_sha256']}. Выполнено проверок — {len(checks)}."
    )
    report.md.append("\n".join(f"- {v}." for v in checks))
    report.sections.append(
        "<ul>" + "".join(f"<li>{html.escape(v)}.</li>" for v in checks) + "</ul>"
    )
    style = """body{font:16px/1.6 system-ui,sans-serif;color:#202832;background:#fafbf9;margin:0}main{max-width:1240px;margin:auto;padding:48px 30px}h1{font-size:40px;line-height:1.15;margin:0 0 14px}h2{font-size:25px;margin-top:56px;border-top:1px solid #d9ded9;padding-top:24px}p,li{max-width:1080px}li{margin:12px 0}.eyebrow{font-size:13px;color:#527166;letter-spacing:2px;text-transform:uppercase}.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:30px 0}.metrics>div{background:#eaf0ec;border-radius:8px;padding:18px;min-width:165px}.metrics strong{display:block;font-size:29px;color:#175643}.metrics span{font-size:13px}img{width:100%;height:auto}figure{margin:35px 0}figcaption{color:#5a6661;font-size:13px}table{border-collapse:collapse;font-size:12px;width:100%}th{text-align:left;background:#e6ece8;position:sticky;top:0}td,th{padding:8px 10px;border-bottom:1px solid #dce3dd;max-width:420px;overflow-wrap:anywhere}tr:nth-child(even){background:#f0f3f0}.table-wrap{overflow:auto;max-height:700px;margin:25px 0}code{background:#e9edeb;padding:2px 5px;font-size:13px;overflow-wrap:anywhere}@media print{main{padding:0}.table-wrap{max-height:none;overflow:visible}h2{break-after:avoid}figure{break-inside:avoid}}"""
    (ROOT / "docs/news_eda.md").write_text(
        "\n\n".join(report.md) + "\n", encoding="utf-8"
    )
    (REPORT / "news_eda.html").write_text(
        f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{report.title}</title><style>{style}</style></head><body><main>'
        + "\n".join(report.sections)
        + "</main></body></html>\n",
        encoding="utf-8",
    )
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (REPORT / "summary.json").write_text(payload)
    (DERIVED / "summary.json").write_text(payload)
    t["review_sample"].to_csv(DERIVED / "review_sample.csv", index=False)
    return summary


def run_analysis(*, refresh: bool = True) -> dict:
    """Refresh the audit, or render previously measured cached tables."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if refresh:
        tables = summarize(collect())
        for name, data in tables.items():
            data.to_parquet(
                DERIVED / f"{name}.parquet", index=False, row_group_size=32_768
            )
    else:
        if not (DERIVED / "volume_benchmark.parquet").is_file():
            msg = "Пересчитайте таблицы по исходным файлам командой python eda/news_eda.py"
            raise FileNotFoundError(msg)
        article_columns = [
            "region",
            "story_id",
            "date",
            "date_ok",
            "host",
            "language",
            "chars",
            "date_shift_days",
            "text_hash",
            "duplicate_of",
            "screened",
            "candidate_event",
            "quality_ok",
            "exact_time_present",
            "content_warning",
            "date_disagreement_gt7d",
            "date_source",
            "publish_date",
            "next_release",
            *["topic_" + key for key in TOPICS],
        ]
        tables = {
            p.stem: pd.read_parquet(
                p, columns=article_columns if p.stem == "articles" else None
            )
            for p in DERIVED.glob("*.parquet")
        }
    checks = validate_tables(tables)
    figures = make_figures(tables)
    summary = make_report(tables, figures, checks)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)  # noqa: T201
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Render cached tables without rescanning the corpus",
    )
    args = parser.parse_args()
    run_analysis(refresh=not args.cached)
