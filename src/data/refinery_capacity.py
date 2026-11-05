from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.eia.gov/petroleum/refinerycapacity"

logger = logging.getLogger(__name__)


class RefineryCapacityDownloader:
    """Download annual EIA refinery-capacity workbooks."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    @staticmethod
    def candidate_urls(year: int) -> list[tuple[str, str]]:
        """Return candidate official URLs and extensions for one year."""

        short_year = str(year)[-2:]

        if year >= 2026:
            return [
                (
                    f"{BASE_URL}/refcap{short_year}.xlsx",
                    ".xlsx",
                )
            ]

        base = f"{BASE_URL}/archive/{year}/refcap{short_year}"
        return [
            (f"{base}.xlsx", ".xlsx"),
            (f"{base}.xls", ".xls"),
        ]

    def download_year(
        self,
        year: int,
        target_dir: Path,
    ) -> Path:
        """Download one annual refinery-capacity workbook."""

        target_dir.mkdir(parents=True, exist_ok=True)
        last_response: requests.Response | None = None

        for url, extension in self.candidate_urls(year):
            response = self.session.get(
                url,
                timeout=self.timeout,
            )
            last_response = response

            if response.ok:
                target_path = target_dir / f"refcap{year}{extension}"
                target_path.write_bytes(response.content)

                logger.info(
                    "Downloaded EIA refinery capacity %s: %s bytes",
                    year,
                    f"{target_path.stat().st_size:,}",
                )
                return target_path

        status_code = (
            last_response.status_code if last_response is not None else "unknown"
        )
        msg = (
            f"Could not download refinery capacity workbook for {year}. "
            f"Last HTTP status: {status_code}."
        )
        raise requests.HTTPError(msg)

    def download_range(
        self,
        start_year: int,
        end_year: int,
        target_dir: Path,
        *,
        strict: bool = True,
    ) -> list[Path]:
        """Download annual snapshots for an inclusive year range."""

        paths: list[Path] = []

        for year in range(start_year, end_year + 1):
            try:
                path = self.download_year(
                    year,
                    target_dir,
                )
            except requests.RequestException:
                if strict:
                    raise

                logger.warning(
                    "Skipping unavailable EIA refinery workbook for %s.",
                    year,
                )
                continue

            paths.append(path)

        return paths


class RefineryCapacityParser:
    """Normalize annual EIA individual-refinery capacity workbooks."""

    @staticmethod
    def _snake_case(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return value.strip("_")

    @staticmethod
    def _report_year(path: Path) -> int:
        match = re.search(
            r"refcap(20\d{2}|\d{2})",
            path.stem.lower(),
        )

        if match is None:
            msg = f"Could not determine report year from {path.name}."
            raise ValueError(msg)

        value = match.group(1)
        return int(value) if len(value) == 4 else 2000 + int(value)

    @staticmethod
    def _excel_engine(path: Path) -> str:
        return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"

    @classmethod
    def discover_files(
        cls,
        directory: Path,
    ) -> list[Path]:
        """Find annual refinery capacity files and sort by report year."""

        files = [
            path
            for path in directory.glob("refcap*")
            if path.suffix.lower() in {".xls", ".xlsx"}
        ]
        return sorted(
            files,
            key=cls._report_year,
        )

    @classmethod
    def _find_header_row(
        cls,
        path: Path,
    ) -> int:
        preview = pd.read_excel(
            path,
            sheet_name=0,
            header=None,
            nrows=25,
            engine=cls._excel_engine(path),
        )

        required = {
            "CORPORATION",
            "COMPANY_NAME",
            "PRODUCT",
            "QUANTITY",
        }

        for row_index, row in preview.iterrows():
            values = {str(value).strip().upper() for value in row.dropna()}

            if required.issubset(values):
                return int(row_index)

        msg = f"Could not find header row in {path.name}."
        raise ValueError(msg)

    @classmethod
    def read_file(
        cls,
        path: Path,
    ) -> pd.DataFrame:
        """Read one annual individual-refinery workbook."""

        header_row = cls._find_header_row(path)
        frame = pd.read_excel(
            path,
            sheet_name=0,
            header=header_row,
            engine=cls._excel_engine(path),
        )
        frame.columns = [cls._snake_case(str(column)) for column in frame.columns]
        frame = frame.dropna(how="all").reset_index(drop=True)

        required_columns = {
            "corporation",
            "company_name",
            "state_name",
            "site",
            "padd",
            "product",
            "supply",
            "quantity",
        }
        missing = required_columns - set(frame.columns)

        if missing:
            msg = f"{path.name} is missing columns: {sorted(missing)}"
            raise ValueError(msg)

        frame["report_year"] = cls._report_year(path)
        frame["source_file"] = path.name
        frame["quantity"] = pd.to_numeric(
            frame["quantity"],
            errors="coerce",
        )
        frame["padd"] = pd.to_numeric(
            frame["padd"],
            errors="coerce",
        ).astype("Int64")

        return frame

    @classmethod
    def build_long(
        cls,
        files: list[Path],
    ) -> pd.DataFrame:
        """Combine all annual raw refinery rows into one long table."""

        if not files:
            msg = "No EIA refinery-capacity workbooks were found."
            raise FileNotFoundError(msg)

        return pd.concat(
            [cls.read_file(path) for path in files],
            ignore_index=True,
        )

    @classmethod
    def build_refinery_year(
        cls,
        long_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create one row per refinery-year with crude distillation capacity."""

        capacity_rows = long_frame[
            long_frame["product"].isin(
                [
                    "OPERATING CAPACITY",
                    "TOTAL OPERABLE CAPACITY",
                ]
            )
            & long_frame["supply"].str.contains(
                "Atmospheric Crude Distillation Capacity",
                case=False,
                na=False,
            )
        ].copy()

        capacity_rows["capacity_basis"] = pd.NA
        capacity_rows.loc[
            capacity_rows["supply"].str.contains(
                "calendar day",
                case=False,
                na=False,
            ),
            "capacity_basis",
        ] = "calendar_day"
        capacity_rows.loc[
            capacity_rows["supply"].str.contains(
                "stream day",
                case=False,
                na=False,
            ),
            "capacity_basis",
        ] = "stream_day"

        refinery_keys = [
            "report_year",
            "corporation",
            "company_name",
            "rdist_label",
            "state_name",
            "site",
            "padd",
        ]
        available_keys = [
            column for column in refinery_keys if column in capacity_rows.columns
        ]

        pivot = capacity_rows.pivot_table(
            index=available_keys,
            columns=["product", "capacity_basis"],
            values="quantity",
            aggfunc="max",
        ).reset_index()
        pivot.columns = [
            "_".join(str(part) for part in column if str(part) not in {"", "nan"})
            if isinstance(column, tuple)
            else str(column)
            for column in pivot.columns
        ]
        pivot.columns = [cls._snake_case(column) for column in pivot.columns]

        rename_map = {
            "operating_capacity_calendar_day": "operating_capacity_bpcd",
            "operating_capacity_stream_day": "operating_capacity_bpsd",
            ("total_operable_capacity_calendar_day"): "total_operable_capacity_bpcd",
            ("total_operable_capacity_stream_day"): "total_operable_capacity_bpsd",
        }
        result = pivot.rename(columns=rename_map)

        if "operating_capacity_bpcd" not in result.columns:
            result["operating_capacity_bpcd"] = pd.NA

        result["operating_capacity_bpcd_reported"] = result["operating_capacity_bpcd"]
        result["operating_capacity_imputed_zero"] = (
            result["operating_capacity_bpcd"].isna()
            & result["total_operable_capacity_bpcd"].notna()
        )
        result["operating_capacity_bpcd"] = result["operating_capacity_bpcd"].fillna(
            0.0
        )
        result["idle_capacity_bpcd"] = (
            result["total_operable_capacity_bpcd"] - result["operating_capacity_bpcd"]
        ).clip(lower=0)
        result["is_idle"] = result["total_operable_capacity_bpcd"].gt(0) & result[
            "operating_capacity_bpcd"
        ].eq(0)
        result["operating_capacity_share"] = (
            result["operating_capacity_bpcd"] / result["total_operable_capacity_bpcd"]
        )

        result["effective_date"] = pd.to_datetime(
            result["report_year"].astype("string") + "-01-01"
        )
        result["safe_available_from"] = pd.to_datetime(
            (result["report_year"] + 1).astype("string") + "-01-01"
        )
        result["availability_rule"] = "conservative_one_year_lag"

        state = (
            result["state_name"]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("<missing>")
        )
        site = (
            result["site"].astype("string").str.strip().str.lower().fillna("<missing>")
        )
        company = (
            result["company_name"]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("<missing>")
        )
        corporation = (
            result["corporation"]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("<missing>")
        )
        rdist = (
            result["rdist_label"]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("<missing>")
        )
        padd = result["padd"].astype("string").fillna("<missing>")

        # A city/site name is not a unique refinery identifier:
        # Houston, Corpus Christi, Port Arthur, etc. contain multiple
        # distinct refinery records. Keep a separate location key for
        # geographic grouping and use the full source identity as the
        # report-year record key.
        result["refinery_location_key"] = state + "|" + site
        result["refinery_key"] = (
            state
            + "|"
            + site
            + "|"
            + company
            + "|"
            + corporation
            + "|padd="
            + padd
            + "|rdist="
            + rdist
        )
        result["refinery_key_scope"] = "eia_source_record_not_stable_physical_entity"

        return result.reset_index(drop=True)

    @staticmethod
    def split_main_padds(
        refinery_year: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate PADD 1-5 from territories/other report rows."""

        main_mask = refinery_year["padd"].isin([1, 2, 3, 4, 5])
        main = refinery_year.loc[main_mask].reset_index(drop=True)
        other = refinery_year.loc[~main_mask].reset_index(drop=True)
        return main, other
