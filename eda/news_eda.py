"""Offline audit of the interrupted multilingual news collection.

Run from the repository root: .venv/bin/python eda/news_eda.py
Only eda/data/derived/news and eda/reports/news are written. No network or model
inference is used. Region means the source collection, not the event location.
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


def collect() -> dict[str, pd.DataFrame]:  # noqa: C901 - sequential stage accounting
    """Scan each region separately so the full text corpus never fills RAM."""
    config = tomllib.loads((ROOT / "src/news/regions.toml").read_text())
    releases = pd.DatetimeIndex(
        u.release_frame()["release_date"].unique()
    ).sort_values()
    rows, monthly, articles, sources, failures, scores, weeks, files = (
        [] for _ in range(8)
    )
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
        for p in all_paths:
            st = p.stat()
            files.append(
                {
                    "path": str(p.relative_to(ROOT)),
                    "bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "rows": pq.read_metadata(p).num_rows,
                }
            )
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
            "extracted",
            "assembled",
        ]:
            row[key] = 0
        weekly = pd.DataFrame(index=releases)
        index = None
        if (interim / "stories.parquet").exists():
            index = pd.read_parquet(
                interim / "stories.parquet",
                columns=["story_id", "publish_date", "media_name", "language"],
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
                    "probability",
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
        if processed.exists():
            data = pd.read_parquet(processed)
            row["assembled"] = len(data)
            data["region"] = region
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
            data["exact_time_present"] = data.date.ne(data.date.dt.normalize())
            page_date = (
                pd.to_datetime(data.extracted_date, errors="coerce", utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
            data["page_date_shift_days"] = (
                page_date - data.date.dt.normalize()
            ).dt.days
            for topic, pattern in TOPICS.items():
                data["topic_" + topic] = data.title.str.contains(
                    pattern, case=False, regex=True, na=False
                )
            for entity, pattern in ENTITIES.items():
                data["entity_" + entity] = data.title.str.contains(
                    pattern, case=False, regex=True, na=False
                )
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
            data = data.drop(
                columns=[
                    "text",
                    "final_url",
                    "error",
                    "extracted_title",
                    "extracted_date",
                    "wayback_timestamp",
                    "http_status",
                ],
                errors="ignore",
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
    canonical = a[a.quality_ok].sort_values(
        ["date", "story_id", "region"], kind="stable"
    )
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
        a[a.date_ok]
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
        a[a.screened]
        .groupby(["region", "next_release"])
        .size()
        .rename("unique_screened")
    )
    weekly = weekly.merge(
        count, left_on=["region", "release_date"], right_index=True, how="left"
    )
    weekly["unique_screened"] = weekly.unique_screened.fillna(0).astype(int)
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
    # Descriptive association only, with multiple-testing control and coverage sensitivity.
    correlations = []
    for region, d in weekly.groupby("region"):
        for name, mask in [
            ("any_attempt", d.attempted.gt(0)),
            ("80pct_attempted", d.attempt_fraction.ge(0.8)),
        ]:
            use = d.loc[
                mask & d.window_days.eq(7), ["assembled", "surprise_mmbbl"]
            ].dropna()
            if len(use) >= 30 and use.assembled.nunique() > 1:
                r, p = stats.spearmanr(use.assembled, use.surprise_mmbbl)
                correlations.append(
                    {"region": region, "screen": name, "n": len(use), "rho": r, "p": p}
                )
    cor = pd.DataFrame(correlations)
    if not cor.empty:
        cor["p_bh"] = stats.false_discovery_control(cor.p.to_numpy(), method="bh")
    tables["count_correlations"] = cor
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
        a[a.title_normalized.ne("")]
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
    for _region, frame in a.groupby("region"):
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
            selected = part.copy()
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
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for col, label in [
        ("index_rows", "Indexed"),
        ("candidates", "Selected by title model"),
        ("attempted", "Fetch attempted"),
        ("assembled", "Assembled text"),
    ]:
        axes[0].plot(x, active[col], "o-", ms=3, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Rows (log scale)")
    axes[0].legend(ncol=2)
    for col, label, color in [
        ("attempted", "Attempted / selected", u.SERIES[0]),
        ("assembled", "Assembled / selected", u.SERIES[1]),
    ]:
        axes[1].plot(x, active[col] / active.candidates, "o-", label=label, color=color)
    axes[1].set_xticks(x, active.index.str.upper(), rotation=45)
    axes[1].set_ylabel("Fraction of selected stories")
    axes[1].legend()
    axes[0].set_title("The download stopped at different stages in different countries")
    fig.tight_layout()
    save(
        "01_pipeline",
        "Pipeline totals use unique attempted story IDs; a failed HTTP request is an attempt, not an article. Index queries differ between recall and full-stream regions.",
    )

    m = tables["monthly"].copy()
    ordered = active.index.tolist()
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True, sharey=True)
    for ax, col, title in zip(
        axes,
        ["indexed", "attempted", "assembled"],
        ["Indexed stories", "Attempted texts", "Assembled texts"],
        strict=True,
    ):
        mat = m.pivot(index="region", columns="month", values=col).reindex(ordered)
        image = ax.imshow(
            np.log10(mat.to_numpy() + 1),
            aspect="auto",
            interpolation="nearest",
            cmap=u.SEQUENTIAL,
        )
        ax.set_yticks(range(len(ordered)), [s.upper() for s in ordered], fontsize=7)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.018, label="log10(rows + 1)")
    ticks = [i for i, month in enumerate(MONTHS) if month.month == 1]
    axes[-1].set_xticks(ticks, [str(MONTHS[i].year) for i in ticks])
    fig.tight_layout()
    save(
        "02_coverage",
        "Publication-month counts. Pale cells in the text panels frequently mean work was never attempted. They must not be encoded as zero news flow in a model.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    failures = (
        tables["failures"].groupby("failure")["count"].sum().sort_values().tail(10)
    )
    failures.plot.barh(ax=axes[0], color=u.SERIES[1])
    axes[0].set_title("Why the latest fetch failed")
    axes[0].set_xlabel("Stories")
    ratio = active.assembled / active.attempted
    ratio.sort_values().plot.barh(ax=axes[1], color=u.SERIES[0])
    axes[1].set_title("Assembled / attempted by country")
    axes[1].set_xlabel("Fraction (includes assembly cleaning)")
    fig.tight_layout()
    save(
        "03_fetch_failures",
        "Failure categories describe the latest saved attempt per story. The assembled fraction additionally reflects template removal and the timing of the last snapshot.",
    )

    a = tables["articles"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hist = tables["score_hist"].groupby("bin_mid")["count"].sum()
    axes[0].bar(hist.index, hist.values, width=0.045, color=u.SERIES[0])
    meta = json.loads((ROOT / "data/models/news/title/meta.json").read_text())
    axes[0].axvline(
        meta["thresholds"]["keep"], color=u.SERIES[1], label="Saved keep/drop threshold"
    )
    axes[0].set_title("Scores in all downloaded classifier outputs")
    axes[0].set_xlabel("Title classifier score (not calibrated event probability)")
    axes[0].legend()
    lang = tables["languages"].set_index("language").rows
    lang.iloc[:20].sort_values().plot.barh(ax=axes[1], color=u.SERIES[2])
    axes[1].set_title("Languages of assembled articles")
    fig.tight_layout()
    save(
        "04_scores_languages",
        "No local gold labels are present, so current-corpus precision and recall cannot be estimated. Training metrics stored in meta.json are reported separately.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    active.sort_values("top1_share").top1_share.plot.barh(ax=axes[0], color=u.SERIES[0])
    axes[0].set_title("Largest host's share of each country corpus")
    axes[0].set_xlabel("Fraction of assembled rows")
    mat = (
        tables["source_year"].groupby(["year", "host"]).rows.sum().unstack(fill_value=0)
    )
    top = mat.sum().nlargest(8).index
    mix = mat[top].div(mat.sum(axis=1), axis=0)
    mix["other hosts"] = 1 - mix.sum(axis=1)
    mix.plot.area(ax=axes[1], linewidth=0, color=[*u.SERIES, u.NEUTRAL])
    axes[1].set_title("Observed host mix changes over time")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(
        "05_sources",
        "Collection-country and host mix shift with archive coverage and fetch success. Raw article-count growth is not evidence of rising oil-event frequency.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].hist(
        a.chars.clip(lower=1),
        bins=np.geomspace(250, max(a.chars.max(), 301), 65),
        color=u.SERIES[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Text length in characters")
    axes[0].axvline(300, color=u.SERIES[1])
    dup = active.local_duplicate_rows / active.assembled
    dup.sort_values().plot.barh(ax=axes[1], color=u.SERIES[1])
    axes[1].set_title("Local exact-duplicate fraction")
    shift = a.date_shift_days.clip(-365, 365)
    axes[2].hist(shift, bins=60, color=u.SERIES[2])
    axes[2].set_yscale("log")
    axes[2].set_title("Resolved date minus index date")
    axes[2].set_xlabel("Days (clipped to ±365)")
    fig.tight_layout()
    save(
        "06_quality",
        "Exact duplication uses the pipeline's normalized full-text hash. Repeated text may be syndication, multiple URLs, or extraction artifacts; manual examples distinguish these.",
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    topic = tables["topics"].set_index("region").reindex(ordered)
    image = ax.imshow(
        topic.to_numpy(), cmap=u.SEQUENTIAL, aspect="auto", vmin=0, vmax=0.5
    )
    ax.set_yticks(range(len(ordered)), [s.upper() for s in ordered])
    ax.set_xticks(range(len(TOPICS)), list(TOPICS), rotation=35, ha="right")
    ax.set_title("Title keyword screens: overlapping themes, not event labels")
    fig.colorbar(image, ax=ax, label="Share of assembled titles (saturated at 50%)")
    fig.tight_layout()
    save(
        "07_topics",
        "Multilingual title dictionaries are transparent but incomplete and unvalidated. Language differences affect match rates. Inventory-reporting matches require leakage review; price commentary is not a physical event label.",
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    entity = tables["entities"].set_index("region").reindex(ordered)
    image = ax.imshow(np.log10(entity.to_numpy() + 1), cmap=u.SEQUENTIAL, aspect="auto")
    ax.set_yticks(range(len(ordered)), [s.upper() for s in ordered])
    ax.set_xticks(range(len(ENTITIES)), list(ENTITIES), rotation=30, ha="right")
    ax.set_title("Anchors for a future infrastructure graph: title mentions")
    fig.colorbar(image, ax=ax, label="log10(title matches + 1)")
    fig.tight_layout()
    save(
        "08_entities",
        "A source collection is not an event location. Exact title matches are possible linking anchors; they do not identify a facility, event direction, duration, or lost capacity.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    cov = tables["release_coverage"]
    for ax, cohort in zip(axes, ["train_2015_2022", "test_2023_2026"], strict=True):
        d = cov[cov.cohort.eq(cohort)].set_index("region").reindex(ordered)
        yy = np.arange(len(d))
        ax.barh(
            yy,
            d.weeks_with_text / d.releases,
            color=u.SERIES[0],
            label="At least one text",
        )
        ax.barh(
            yy,
            d.weeks_80pct / d.releases,
            color=u.SERIES[2],
            height=0.35,
            label="≥80% of selected stories attempted",
        )
        ax.set_yticks(yy, [s.upper() for s in ordered])
        ax.set_xlim(0, 1)
        ax.set_title(cohort)
        ax.set_xlabel("Fraction of observed EIA release windows")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(
        "09_release_coverage",
        "Articles are assigned to the first observed EIA release strictly after their publication day. Missing release dates and non-seven-day windows are flagged; text retrieval dates do not establish historical availability.",
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    q = tables["quality"].set_index("region").reindex(ordered)
    for col, label in [
        ("assembled", "Assembled"),
        ("quality_pass", "Valid date, title and length"),
        ("unique_screened", "Globally unique screened text"),
        (
            "physical_candidates",
            "Physical-theme title screen, excluding inventory reports",
        ),
    ]:
        ax.plot(x, q[col], "o-", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [s.upper() for s in ordered], rotation=45)
    ax.set_ylabel("Rows (log scale)")
    ax.set_title("How much text remains after transparent EDA screens")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(
        "10_screening",
        "Global originals are attributed to the earliest valid dated row, with stable ID/country tie breaks. Retain all provenance links when building the graph. Screened means suitable for inspection, not certified model-ready.",
    )
    return figs


def validate_tables(t: dict[str, pd.DataFrame]) -> list[str]:
    """Check accounting and time boundaries against the complete local snapshot."""
    checks = []
    r, a, w = t["regions"], t["articles"], t["weekly"]
    conditions = {
        "Assembled row totals reconcile across countries and article table": len(a)
        == r.assembled.sum(),
        "No duplicate (country, story ID) in assembled snapshots": not a.duplicated(
            ["region", "story_id"]
        ).any(),
        "Every country has all 142 audit-month rows": len(t["monthly"])
        == len(r) * len(MONTHS),
        "Fetch attempts do not exceed selected stories": r.attempted.le(
            r.candidates
        ).all(),
        "Assembled rows do not exceed uniquely attempted stories": r.assembled.le(
            r.attempted
        ).all(),
        "Weekly attempted candidates never exceed selected candidates": w.attempted.le(
            w.candidates
        ).all(),
        "Unattempted weeks retain missing observed text counts": w.loc[
            w.attempted.eq(0), "observed_text_count"
        ]
        .isna()
        .all(),
        "Screened texts have valid dates, lengths and titles": a.loc[
            a.screened, "quality_ok"
        ].all(),
        "Screened full-text hashes are globally unique": a.loc[
            a.screened, "text_hash"
        ].is_unique,
        "Release windows strictly exclude the article's publication day": (
            a.loc[a.next_release.notna(), "date"].dt.normalize()
            < a.loc[a.next_release.notna(), "next_release"]
        ).all(),
        "Stored surprise equals actual minus consensus": np.allclose(
            u.release_frame().surprise_mmbbl,
            u.release_frame().actual_mmbbl - u.release_frame().consensus_mmbbl,
        ),
    }
    for name, passed in conditions.items():
        if not passed:
            raise ValueError(name)
        checks.append(name)
    return checks


def make_report(
    t: dict[str, pd.DataFrame], figs: list[tuple[str, str]], checks: list[str]
) -> dict:
    """Write a self-contained HTML report with measured claims and limitations."""
    a, r = t["articles"], t["regions"]
    active = r[r.assembled.gt(0)].copy()
    meta = json.loads((ROOT / "data/models/news/title/meta.json").read_text())
    cross = t["duplicate_clusters"].query("regions > 1")
    text_splits = (
        a[a.date_ok]
        .assign(cohort=lambda d: d.date.ge("2023-01-01"))
        .groupby("text_hash")
        .cohort.nunique()
        .gt(1)
    )
    top_hosts = (
        a.host.value_counts().head(12).rename_axis("host").reset_index(name="rows")
    )
    us = (
        active.set_index("region").loc["us"]
        if "us" in active.region.to_numpy()
        else None
    )
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
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "scope_start": str(START.date()),
        "scope_end": str(END.date()),
        "configured_regions": len(r),
        "regions_with_text": len(active),
        "indexed_rows": int(r.index_rows.sum()),
        "candidate_rows": int(r.candidates.sum()),
        "attempted_rows": int(r.attempted.sum()),
        "assembled_rows": len(a),
        "screened_unique_texts": int(a.screened.sum()),
        "physical_title_candidates": int(a.candidate_event.sum()),
        "languages": int(a.language.nunique()),
        "hosts": int(a.host.nunique()),
        "local_duplicate_rows": int(a.duplicate_of.notna().sum()),
        "all_duplicate_excess": int(len(a) - a.text_hash.nunique()),
        "cross_region_duplicate_clusters": len(cross),
        "cross_split_text_clusters": int(text_splits.sum()),
        "invalid_date_rows": int((~a.date_ok).sum()),
        "future_date_rows": int(a.date.gt(END).sum()),
        "exact_time_rows": int(a.exact_time_present.sum()),
        "content_warning_rows": int(a.content_warning.sum()),
        "date_disagreement_rows": int(a.date_disagreement_gt7d.sum()),
        "inventory_title_flags": int(a.topic_inventory_reporting.sum()),
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
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    t["review_sample"].to_csv(DERIVED / "review_sample.csv", index=False)
    sections = []

    def paragraph(s: str) -> None:
        sections.append(f"<p>{s}</p>")

    def heading(s: str) -> None:
        sections.append(f"<h2>{html.escape(s)}</h2>")

    def table(frame: pd.DataFrame, limit: int = 50) -> None:
        sections.append(
            '<div class="table-wrap">'
            + frame.head(limit).to_html(
                index=False,
                escape=True,
                border=0,
                na_rep="—",
                float_format=lambda x: f"{x:,.3f}",
            )
            + "</div>"
        )

    def figure(index: int) -> None:
        name, caption = figs[index]
        encoded = base64.b64encode((REPORT / f"{name}.png").read_bytes()).decode()
        sections.append(
            f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )

    paragraph(
        "Offline exploratory analysis of the interrupted multilingual collection. The original downloader, raw files, classifier outputs and assembled datasets were read without modification. All counts below come from this machine; the older Russian-only notebook and the non-news EDA remain separate."
    )
    sections.append(
        '<div class="metrics">'
        + "".join(
            f"<div><strong>{value:,}</strong><span>{label}</span></div>"
            for value, label in [
                (len(active), "countries with text"),
                (int(r.index_rows.sum()), "indexed rows"),
                (len(a), "assembled articles"),
                (int(a.screened.sum()), "unique screened texts"),
                (int(a.language.nunique()), "languages"),
            ]
        )
        + "</div>"
    )
    heading("1. Findings that change the research plan")
    findings = [
        f"The corpus is substantial but incomplete: {int(r.attempted.sum()):,} of {int(r.candidates.sum()):,} selected rows have a saved fetch attempt ({r.attempted.sum() / r.candidates.sum():.1%}). The missing portion is unobserved collection work, not evidence that no news occurred.",
        f"Exact repeated text is material: {int(a.duplicate_of.notna().sum()):,} rows are marked as local duplicates; full-corpus hashing finds {len(a) - a.text_hash.nunique():,} redundant rows and {len(cross):,} text clusters spanning multiple source countries. Syndication and overlapping collections can inflate counts and graph centrality.",
        f"Dates need repair: {int((~a.date_ok).sum()):,} rows have an out-of-range or explicitly unreliable resolved date; {int(a.date.gt(END).sum()):,} fall after the collection end. URL-derived dates are evidence to inspect, not ground truth.",
        f"All {len(a):,} saved rows contain only publication-day resolution for the EDA cutoff"
        if not a.exact_time_present.any()
        else f"Only {int(a.exact_time_present.sum()):,} rows contain non-midnight publication times",
        f"The saved model uses identical keep and drop thresholds ({meta['thresholds']['keep']:.6f}). This creates a binary filter, with {int(r.borderline.fillna(0).sum()):,} borderline decisions in the saved scores. Its current-corpus precision cannot be verified without local gold labels.",
        f"The broad title screens flag {int(a.topic_inventory_reporting.sum()):,} possible inventory-reporting stories. These require leakage review before they become predictors; title keyword matches are not validated event labels.",
    ]
    if us is not None:
        findings.insert(
            1,
            f"The US index contains {int(us.index_rows):,} rows but only {int(us.assembled):,} assembled articles, dated {us.first_text:%Y-%m-%d}–{us.last_text:%Y-%m-%d}. Its marked duplicate fraction is {us.local_duplicate_rows / us.assembled:.1%}. This snapshot cannot support a 2015–2022 US full-text training set.",
        )
    sections.append("<ol>" + "".join(f"<li>{s}</li>" for s in findings) + "</ol>")
    heading("2. Collection accounting and what remains unfinished")
    paragraph(
        "Each country is a Media Cloud source collection. A US-source story can describe Saudi Arabia; a Saudi-source story can discuss US inventories. Four regions use full-stream queries while most use recall terms, so retention rates have different denominators. Packed multi-month filenames are expanded to calendar months; file counts are not month counts. Metadata counts include existing snapshots, even when a stage was interrupted."
    )
    account = active[
        [
            "region",
            "mode",
            "months_downloaded",
            "index_rows",
            "candidates",
            "attempted",
            "extracted",
            "assembled",
        ]
    ].copy()
    account["attempted_selected_pct"] = account.attempted / account.candidates * 100
    account["assembled_attempted_pct"] = account.assembled / account.attempted * 100
    table(account.sort_values("index_rows", ascending=False))
    missing = r[r.index_rows.eq(0)].region.str.upper().tolist()
    paragraph(
        "Configured countries without a local merged index: "
        + html.escape(", ".join(missing))
        + ". Existing indexed windows are not proof of complete publisher coverage, classifier recall, or a complete Media Cloud archive."
    )
    figure(0)
    figure(1)
    heading("3. Fetch failures and selection bias")
    paragraph(
        "The failure audit deduplicates retries by story ID and takes the latest fetched_at, matching the downloader's read policy. The extraction count precedes assembly cleaning. Failed attempts and successfully extracted texts excluded as templates are different outcomes. Recent-first processing concentrates partial corpora near the collection end; a successful live fetch in 2026 is also not proof that the recovered text is the original historical version."
    )
    table(
        t["failures"]
        .groupby("failure")["count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    source = (
        t["sources"]
        .groupby(["region", "source", "stage"])["count"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["candidate", "attempted", "assembled"]:
        if col not in source:
            source[col] = 0
    source["attempt_fraction"] = source.attempted / source.candidate.replace(0, np.nan)
    source["assembled_fraction"] = source.assembled / source.attempted.replace(
        0, np.nan
    )
    paragraph(
        "Largest selected publisher strata, with both selection and recovery denominators:"
    )
    table(source.sort_values("candidate", ascending=False), 20)
    figure(2)
    heading("4. Classifier scores, languages and representativeness")
    paragraph(
        f"The saved multilingual-e5-base title model was trained on {meta['trained_on']:,} examples. Its metadata reports PR-AUC {meta['metrics']['pr_auc']:.3f}, language-macro PR-AUC {meta['metrics']['pr_auc_macro_language']:.3f}, stratum-weighted PR-AUC {meta['metrics']['pr_auc_weighted']:.3f}, and temporal PR-AUC {meta['metrics']['pr_auc_temporal']:.3f}. The gap between unweighted and weighted metrics matters. These are historical model-development metrics, not a new validation of this corpus. Score thresholds should not be interpreted as calibrated event probabilities. No encoder or paid labeler was run for this EDA."
    )
    table(t["languages"])
    figure(3)
    heading("5. Sources, concentration and temporal drift")
    paragraph(
        "Source balance matters twice: the index has changing archive coverage, and the attempted subset has a changing recovery mix. Effective source count is 1 / sum(host_share²). Adjacent-year total variation measures how much publisher weight changes; it can reflect collection artifacts, not changes in world events. The 2026 slice ends in October, so annual totals are not like-for-like."
    )
    table(
        active[
            ["region", "sources", "top1_share", "top5_share", "effective_sources"]
        ].sort_values("top1_share", ascending=False)
    )
    table(top_hosts)
    paragraph(
        "Largest adjacent-year source-mix shifts with at least 100 articles in both years:"
    )
    table(
        t["source_drift"]
        .query("rows_before >= 100 and rows_after >= 100")
        .sort_values("total_variation", ascending=False),
        12,
    )
    figure(4)
    heading("6. Text quality, exact repeats and date errors")
    paragraph(
        f"Character counts are language-neutral diagnostics; whitespace word counts are not comparable across all scripts. The basic screen requires a nonempty title, a normalized text hash, 300–100,000 characters, a resolved date within {START:%Y-%m-%d}–{END:%Y-%m-%d}, and date_source other than unreliable. It leaves {int(a.quality_ok.sum()):,} rows before global text deduplication. These thresholds are transparent EDA choices, not learned quality labels."
    )
    paragraph(
        f"Content screens flag {int(a.content_warning.sum()):,} rows: recognizable consent/access messages in the first 600 characters, or one text used at least ten times under at least five distinct titles over more than 30 days. These rows are excluded from the inspection-ready screen, pending manual review. A concrete case is the iDNES advertising-consent text surviving on many subdomains: per-host template removal misses a template spread across subdomains. The largest US clusters, in contrast, show wire-service syndication across dozens of stations."
    )
    table(t["template_clusters"], 10)
    paragraph(
        f"Resolved and index dates disagree by more than seven days in {int(a.date_disagreement_gt7d.sum()):,} rows. Some are legitimate corrections, so this disagreement is retained as a review flag rather than automatically rejected. Out-of-range checks alone miss plausible-looking wrong years. Numeric story IDs and timestamp suffixes can accidentally match the current URL date regex."
    )
    table(
        a.groupby("language")
        .chars.quantile([0.1, 0.5, 0.9, 0.99])
        .unstack()
        .rename(
            columns={
                0.1: "p10_chars",
                0.5: "p50_chars",
                0.9: "p90_chars",
                0.99: "p99_chars",
            }
        )
        .reset_index()
    )
    paragraph(
        f"{int(text_splits.sum()):,} exact-text clusters occur on both sides of the 2023 split. Group duplicates before train/test evaluation; the same content must not enter both folds. Identical titles are a separate signal and can describe recurring updates, so they are not automatically removed."
    )
    table(t["duplicate_clusters"].head(12))
    top_hashes = t["duplicate_clusters"].text_hash.head(4)
    paragraph(
        "Examples from the largest repeated-text clusters (first 180 characters of extracted text):"
    )
    examples = a[a.text_hash.isin(top_hashes)].groupby("text_hash", sort=False).head(2)
    examples = examples[["region", "date", "host", "title", "excerpt"]].copy()
    examples["excerpt"] = examples.excerpt.str.slice(0, 180)
    table(examples)
    paragraph(
        "Out-of-range date examples; compare the title, URL-derived date and original index date before repairing:"
    )
    table(
        a[~a.date_ok][
            ["region", "date", "publish_date", "date_source", "title", "url"]
        ].sort_values("date", ascending=False),
        12,
    )
    table(
        a.date_source.value_counts().rename_axis("date_source").reset_index(name="rows")
    )
    figure(5)
    heading("7. Physical themes, direct inventory reports and graph anchors")
    paragraph(
        "The title dictionaries cover several observed languages and allow overlap. They are useful for finding review examples, not comparing countries' true event prevalence. Matched production, refining, pipeline, outage, weather and shipping themes suggest event candidates. Generic price commentary and reporting of the EIA/API result should be reviewed separately. The event screen excludes inventory-reporting matches but has neither measured precision nor measured recall."
    )
    figure(6)
    paragraph(
        "Company and location title mentions provide initial linking candidates. Building the graph still requires entity disambiguation, facility IDs, event time, source publication time, capacity units, direction and duration. Missing a title keyword is not evidence that an article lacks a relevant entity in its body."
    )
    figure(7)
    table(t["quality"])
    figure(9)
    heading("8. EIA release alignment and historical feasibility")
    paragraph(
        "For exploratory alignment only, an article is assigned to the first observed Myfxbook EIA release strictly after its normalized publication day. This excludes all same-day stories because intraday availability cannot be established. The first window has no known start; missing calendar releases create longer windows. Windows other than seven days are flagged and excluded from the descriptive correlation screen. This is conservative by publication date but does not certify that the text, extraction version, or corrected timestamp was known at the historical cutoff."
    )
    paragraph(
        "The target calendar inherits the existing Myfxbook caveats: missing weeks in 2022 and 2023, and an ambiguous delayed publication in June 2022. No missing releases or consensus values were fabricated. The target and API estimates are used as outcomes/context, not features extracted from the article text."
    )
    figure(8)
    table(t["release_coverage"].query("weeks_with_text > 0"))
    paragraph(
        "The weekly panel retains indexed, selected, attempted and assembled counts. observed_text_count is missing when no texts were attempted. An attempted fraction of at least 80% is a coverage sensitivity screen, not a claim of completeness. Even fully attempted selections remain conditioned on the query, classifier and fetch success."
    )
    paragraph(
        "Pipeline throughput is grouped by the original index publication date; assembled content is grouped by the resolved publication date. Corrections can move stories between windows, so stage ratios are diagnostics rather than exact same-window recovery probabilities. Large date disagreements should be reviewed before modeling."
    )
    heading("9. News counts versus inventory surprise: descriptive screen")
    paragraph(
        "Spearman correlations below use assembled counts in seven-day windows with at least 30 observations, once for all attempted windows and once for windows with at least 80% of selected stories attempted. Benjamini–Hochberg p-values adjust across the complete displayed family. Counts still reflect archive and recovery bias, duplicates and source mix. This is an exploratory association, not a predictive test or evidence of incremental value over API. The reviewed non-news baseline already shows that API minus consensus is a strong comparator."
    )
    cor = t["count_correlations"]
    if not cor.empty:
        paragraph(
            f"Measured result: {int(cor.p_bh.lt(0.05).sum())} of {len(cor)} country/coverage comparisons pass a 5% false-discovery threshold. This screen provides no convincing evidence that raw publication counts predict the inventory surprise."
        )
    table(cor.sort_values("p_bh") if not cor.empty else cor, 100)
    heading("10. Concrete next steps for the graph dataset")
    sections.append(
        "<ol><li>Repair the date resolver against the anomalies above, retaining original and corrected fields plus the reason. Reassemble only after testing those cases.</li><li>Use the stage and monthly tables to backfill missing historical texts, prioritizing US training years and publisher strata with large remaining selected counts. Do not treat partial months as zero events.</li><li>Annotate the stratified review sample for oil relevance, physical event type, text quality and date validity. It is enriched for difficult cases and is not a probability sample for estimating corpus-wide precision.</li><li>Collapse exact text into one document node, keep publisher/URL/country provenance as edges, and split duplicate groups together during evaluation. Review repeated-title variants before adding near-duplicate clustering.</li><li>Extract event location independently of source country. Link facilities and companies to the infrastructure tables with explicit confidence and unresolved cases.</li><li>Fix a prediction timestamp and collect genuine publication/availability evidence. Validate incremental performance over consensus, API and pre-release fundamentals with temporal evaluation after coverage and leakage audits pass.</li></ol>"
    )
    paragraph(
        f"Manual review file: <code>eda/data/derived/news/review_sample.csv</code>, {len(t['review_sample']):,} rows. Human label fields are deliberately blank. The compact article audit includes excerpts and flags; the original full texts remain in the source datasets."
    )
    heading("11. Reproduction, provenance and checks")
    paragraph(
        "Run <code>MPLCONFIGDIR=/tmp/oil-news-matplotlib .venv/bin/python eda/news_eda.py</code> from the repository root. Open <code>eda/notebooks/08_news_corpus.ipynb</code> for the executed walkthrough. The analyzer reads one country at a time, uses no network, does not retrain the classifier, and writes only derived EDA tables and this report. Seed: 6107. Tables are Parquet plus a CSV review sample; the HTML embeds its figures and can be shared as a single file."
    )
    paragraph(
        f"Generated UTC: {html.escape(summary['generated_at_utc'])}. Input manifest: {len(manifest):,} files, SHA-256 of sorted path/size/mtime/row metadata <code>{digest}</code>. This fingerprint is metadata-based, not a cryptographic hash of file contents. Inputs changed during analysis: {len(changed)}."
    )
    sections.append(
        "<ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in checks) + "</ul>"
    )
    style = """body{font:16px/1.6 system-ui,sans-serif;color:#202832;background:#fafbf9;margin:0}main{max-width:1240px;margin:auto;padding:48px 30px}h1{font-size:40px;line-height:1.15;margin:0 0 14px}h2{font-size:25px;margin-top:56px;border-top:1px solid #d9ded9;padding-top:24px}p,li{max-width:1080px}li{margin:12px 0}.eyebrow{font-size:13px;color:#527166;letter-spacing:2px;text-transform:uppercase}.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:30px 0}.metrics>div{background:#eaf0ec;border-radius:8px;padding:18px;min-width:165px}.metrics strong{display:block;font-size:29px;color:#175643}.metrics span{font-size:13px}img{width:100%;height:auto}figure{margin:35px 0}figcaption{color:#5a6661;font-size:13px}table{border-collapse:collapse;font-size:12px;width:100%}th{text-align:left;background:#e6ece8;position:sticky;top:0}td,th{padding:8px 10px;border-bottom:1px solid #dce3dd;max-width:420px;overflow-wrap:anywhere}tr:nth-child(even){background:#f0f3f0}.table-wrap{overflow:auto;max-height:700px;margin:25px 0}code{background:#e9edeb;padding:2px 5px;font-size:13px;overflow-wrap:anywhere}@media print{main{padding:0}.table-wrap{max-height:none;overflow:visible}h2{break-after:avoid}figure{break-inside:avoid}}"""
    output = (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Oil news corpus — EDA</title><style>{style}</style><main><p class="eyebrow">Oil news graph / local collection audit</p><h1>What the interrupted news collection can support</h1>'
        + "\n".join(sections)
        + "</main></html>"
    )
    (REPORT / "news_eda.html").write_text(output, encoding="utf-8")
    (REPORT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def run_analysis(*, refresh: bool = True) -> dict:
    """Refresh the audit, or render previously measured cached tables."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if refresh:
        tables = summarize(collect())
        for name, data in tables.items():
            data.to_parquet(DERIVED / f"{name}.parquet", index=False)
    else:
        tables = {p.stem: pd.read_parquet(p) for p in DERIVED.glob("*.parquet")}
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
