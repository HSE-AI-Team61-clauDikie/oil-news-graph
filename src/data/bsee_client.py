from __future__ import annotations

import csv
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

PRODUCTION_BY_PLATFORM_URL = (
    "https://www.data.bsee.gov/Production/Files/ProdByPlatformRawData.zip"
)
PLATFORM_STRUCTURES_URL = "https://www.data.bsee.gov/Platform/Files/platstrufixed.zip"
PLATFORM_LOCATIONS_URL = "https://www.data.bsee.gov/Platform/Files/platlocfixed.zip"
PIPELINE_MASTERS_URL = "https://www.data.bsee.gov/Pipeline/Files/pplmastdelimit.zip"
PIPELINE_HISTORIES_URL = "https://www.data.bsee.gov/Pipeline/Files/2100delimit.zip"

PLATFORM_STRUCTURE_WIDTHS = (
    2,
    6,
    8,
    2,
    1,
    11,
    11,
    1,
    1,
    11,
    3,
    3,
    3,
    3,
    15,
    3,
    5,
    6,
    6,
    3,
    16,
    8,
    20,
)
PLATFORM_STRUCTURE_COLUMNS = (
    "area_code",
    "block_number",
    "complex_id_num",
    "deck_count",
    "east_west_departure_code",
    "install_date",
    "last_revision_date",
    "major_structure_flag",
    "north_south_departure_code",
    "removal_date",
    "slant_slot_count",
    "slot_count",
    "slot_drill_count",
    "satellite_completion_count",
    "structure_name",
    "structure_number",
    "structure_type_code",
    "surface_east_west_distance",
    "surface_north_south_distance",
    "underwater_completion_count",
    "authority_type",
    "authority_number",
    "authority_status",
)

PLATFORM_LOCATION_WIDTHS = (
    3,
    8,
    3,
    2,
    6,
    15,
    6,
    1,
    6,
    1,
    17,
    17,
    14,
    13,
)
PLATFORM_LOCATION_COLUMNS = (
    "district_code",
    "complex_id_num",
    "structure_number",
    "area_code",
    "block_number",
    "structure_name",
    "surface_north_south_distance",
    "north_south_departure_code",
    "surface_east_west_distance",
    "east_west_departure_code",
    "x_location",
    "y_location",
    "longitude",
    "latitude",
)

PIPELINE_MASTER_COLUMNS = (
    "segment_num",
    "seg_length",
    "orig_id_name",
    "orig_ar_code",
    "orig_blk_num",
    "orig_lse_num",
    "dest_id_name",
    "dest_ar_code",
    "dest_blk_num",
    "dest_lse_num",
    "aban_aprv_dt",
    "aban_date",
    "approved_date",
    "auth_code",
    "bd_ppl_sdv_fl",
    "bur_dsgn_fl",
    "cat_life_tm",
    "dep_flag",
    "ppl_const_date",
    "lk_detec_fl",
    "last_rev_date",
    "init_hs_dt",
    "fed_st_lgth",
    "status_code",
    "ppl_size_code",
    "row_number",
    "recv_maop_prss",
    "recv_seg_num",
    "prpsl_con_dt",
    "prod_code",
    "sys_desig_code",
    "ppl_row_prmte",
    "facil_operator",
    "min_wtr_dpth",
    "max_wtr_dpth",
    "prot_number",
    "maop_prss",
    "cathodic_code",
    "bidir_flag",
    "bd_ppl_fsv_fl",
    "aprv_code",
    "aban_type",
)

PIPELINE_HISTORY_COLUMNS = (
    "segment_num",
    "history_date",
    "history_code",
    "last_rev_date",
    "remark",
)

PIPELINE_STATUS = {
    "A/C": "abandoned_and_combined",
    "ABN": "abandoned",
    "ACT": "active",
    "CNCL": "cancelled",
    "COMB": "combined",
    "O/C": "out_and_combined",
    "OUT": "out_of_service",
    "PABN": "proposed_abandonment",
    "PREM": "proposed_removal",
    "PROP": "proposed",
    "R/A": "relinquished_and_abandoned",
    "R/C": "relinquished_and_combined",
    "R/R": "relinquished_and_removed",
    "RELQ": "relinquished",
    "REM": "removed",
}

PIPELINE_HISTORY_EVENT = {
    "A": "as_built_approved",
    "B": "abandoned_or_proposed_abandonment",
    "C": "combined_into_new_segment",
    "D": "departure_approved",
    "E": "miscellaneous",
    "F": "flow_direction_change",
    "H": "hydrostatic_test",
    "I": "first_oil_flow_alaska",
    "L": "flush_fill",
    "M": "maop_change",
    "O": "out_of_service",
    "P": "pipe_added_or_segment_created",
    "R": "repair",
    "S": "service_change",
    "T": "temporary_cessation",
    "U": "placed_or_returned_to_service",
}

CRUDE_OIL_PRODUCT_CODES = {
    "BLKO",
    "BLOH",
    "G/O",
    "G/OH",
    "O/W",
    "OIL",
    "OILH",
}

logger = logging.getLogger(__name__)


class BseeClient:
    """Downloader and parser for BSEE bulk infrastructure data."""

    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    def _download_archive(
        self,
        url: str,
        target_path: Path,
    ) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        response = self.session.get(
            url,
            timeout=self.timeout,
        )
        response.raise_for_status()
        target_path.write_bytes(response.content)

        logger.info(
            "Downloaded BSEE archive %s: %s bytes",
            target_path.name,
            f"{target_path.stat().st_size:,}",
        )
        return target_path

    def download_production_by_platform(
        self,
        target_path: Path,
    ) -> Path:
        """Download production-by-platform raw data."""

        return self._download_archive(
            PRODUCTION_BY_PLATFORM_URL,
            target_path,
        )

    def download_platform_structures(
        self,
        target_path: Path,
    ) -> Path:
        """Download historical platform structure records."""

        return self._download_archive(
            PLATFORM_STRUCTURES_URL,
            target_path,
        )

    def download_platform_locations(
        self,
        target_path: Path,
    ) -> Path:
        """Download platform coordinates."""

        return self._download_archive(
            PLATFORM_LOCATIONS_URL,
            target_path,
        )

    def download_pipeline_masters(
        self,
        target_path: Path,
    ) -> Path:
        """Download BSEE pipeline master records."""

        return self._download_archive(
            PIPELINE_MASTERS_URL,
            target_path,
        )

    def download_pipeline_histories(
        self,
        target_path: Path,
    ) -> Path:
        """Download BSEE pipeline event history."""

        return self._download_archive(
            PIPELINE_HISTORIES_URL,
            target_path,
        )

    @staticmethod
    def extract_archive(
        archive_path: Path,
        target_dir: Path,
    ) -> list[Path]:
        """Extract ZIP archive and return extracted files."""

        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)

        return [
            path
            for path in target_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        ]

    @staticmethod
    def _decode_sample(raw: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue

        return raw.decode("latin1", errors="replace")

    @classmethod
    def _detect_separator(cls, path: Path) -> str:
        with path.open("rb") as file:
            raw_sample = file.read(65_536)
        sample = cls._decode_sample(raw_sample)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters="|,\t;^",
            )
            return dialect.delimiter
        except csv.Error:
            return "|"

    @staticmethod
    def _snake_case(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return value.strip("_")

    @staticmethod
    def _clean_strings(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()

        for column in result.select_dtypes(include=["object", "string"]):
            result[column] = (
                result[column].astype("string").str.strip().replace({"": pd.NA})
            )

        return result

    @staticmethod
    def _first_data_file(
        extracted_files: list[Path],
    ) -> Path:
        candidates = [
            path
            for path in extracted_files
            if path.suffix.lower() in {".txt", ".csv", ".dat"}
        ]

        if not candidates:
            msg = "No text data file found in the BSEE archive."
            raise FileNotFoundError(msg)

        return max(
            candidates,
            key=lambda path: path.stat().st_size,
        )

    @classmethod
    def read_delimited_file(
        cls,
        path: Path,
    ) -> pd.DataFrame:
        """Read an extracted BSEE delimited text file."""

        separator = cls._detect_separator(path)
        frame = pd.read_csv(
            path,
            sep=separator,
            engine="python",
            encoding="latin1",
        )
        frame.columns = [cls._snake_case(str(column)) for column in frame.columns]
        return cls._clean_strings(frame)

    @classmethod
    def _read_headerless_delimited(
        cls,
        path: Path,
        names: tuple[str, ...],
    ) -> pd.DataFrame:
        separator = cls._detect_separator(path)
        frame = pd.read_csv(
            path,
            sep=separator,
            engine="python",
            encoding="latin1",
            header=None,
            names=list(names),
            dtype="string",
        )

        if frame.shape[1] != len(names):
            msg = (
                f"Unexpected BSEE column count in {path.name}: "
                f"expected {len(names)}, got {frame.shape[1]}"
            )
            raise ValueError(msg)

        return cls._clean_strings(frame)

    @classmethod
    def _read_fixed_width(
        cls,
        path: Path,
        widths: tuple[int, ...],
        names: tuple[str, ...],
    ) -> pd.DataFrame:
        frame = pd.read_fwf(
            path,
            widths=list(widths),
            names=list(names),
            dtype="string",
            encoding="latin1",
        )
        return cls._clean_strings(frame)

    @classmethod
    def load_production_by_platform(
        cls,
        extracted_files: list[Path],
    ) -> pd.DataFrame:
        """Load production-by-platform table."""

        path = cls._first_data_file(extracted_files)
        frame = cls.read_delimited_file(path)

        for column in (
            "production_month",
            "production_date",
            "install_date",
            "removal_date",
        ):
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column],
                    errors="coerce",
                )

        for column in (
            "complex_id_num",
            "structure_number",
            "pf_operator_num",
            "producing_wells",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).astype("Int64")

        for column in ("bopd", "mcfpd", "boepd", "bwpd"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        return frame

    @classmethod
    def prepare_production_since(
        cls,
        frame: pd.DataFrame,
        start_date: str = "2015-01-01",
    ) -> pd.DataFrame:
        """Keep the model period and remove exact duplicate rows."""

        if "production_date" not in frame.columns:
            msg = "production_date missing from BSEE production table."
            raise KeyError(msg)

        start = pd.Timestamp(start_date)
        result = frame.loc[frame["production_date"] >= start].copy()

        return result.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def aggregate_production_by_complex(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate structure rows to complex-month physical production."""

        required = {
            "complex_id_num",
            "production_date",
            "structure_number",
            "bopd",
            "mcfpd",
            "boepd",
            "bwpd",
            "producing_wells",
        }
        missing = required - set(frame.columns)

        if missing:
            msg = f"Missing columns for complex aggregation: {sorted(missing)}"
            raise KeyError(msg)

        return (
            frame.groupby(
                ["complex_id_num", "production_date"],
                dropna=False,
            )
            .agg(
                structure_count=("structure_number", "nunique"),
                producing_wells=("producing_wells", "sum"),
                bopd=("bopd", "sum"),
                mcfpd=("mcfpd", "sum"),
                boepd=("boepd", "sum"),
                bwpd=("bwpd", "sum"),
                operator_count=("pf_operator", "nunique"),
                primary_operator=("pf_operator", "first"),
            )
            .reset_index()
        )

    @classmethod
    def load_platform_structures(
        cls,
        extracted_files: list[Path],
    ) -> pd.DataFrame:
        """Parse historical BSEE Platform Structures fixed-width dump."""

        path = cls._first_data_file(extracted_files)
        frame = cls._read_fixed_width(
            path,
            PLATFORM_STRUCTURE_WIDTHS,
            PLATFORM_STRUCTURE_COLUMNS,
        )

        for column in (
            "install_date",
            "last_revision_date",
            "removal_date",
        ):
            frame[column] = pd.to_datetime(
                frame[column],
                format="%d-%b-%Y",
                errors="coerce",
            )

        for column in (
            "complex_id_num",
            "structure_number",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).astype("Int64")

        return frame

    @classmethod
    def load_platform_locations(
        cls,
        extracted_files: list[Path],
    ) -> pd.DataFrame:
        """Parse BSEE Platform Locations fixed-width dump."""

        path = cls._first_data_file(extracted_files)
        frame = cls._read_fixed_width(
            path,
            PLATFORM_LOCATION_WIDTHS,
            PLATFORM_LOCATION_COLUMNS,
        )

        for column in (
            "complex_id_num",
            "structure_number",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).astype("Int64")

        for column in (
            "longitude",
            "latitude",
            "x_location",
            "y_location",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        return frame

    @staticmethod
    def build_platform_history(
        structures: pd.DataFrame,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine historic structure dates with stable coordinates."""

        keys = ["complex_id_num", "structure_number"]

        structure_rows = (
            structures.sort_values(
                [*keys, "last_revision_date"],
                na_position="first",
            )
            .drop_duplicates(
                subset=keys,
                keep="last",
            )
            .copy()
        )
        location_rows = locations.drop_duplicates(
            subset=keys,
            keep="last",
        ).copy()

        location_columns = [
            *keys,
            "district_code",
            "longitude",
            "latitude",
            "x_location",
            "y_location",
        ]

        result = structure_rows.merge(
            location_rows[location_columns],
            on=keys,
            how="left",
            validate="one_to_one",
        )

        result["structure_id"] = (
            result["complex_id_num"].astype("string")
            + "-"
            + result["structure_number"].astype("string")
        )
        result["active_from"] = result["install_date"]
        result["active_to"] = result["removal_date"]
        result["is_current"] = result["removal_date"].isna()
        result["has_coordinates"] = (
            result["latitude"].notna() & result["longitude"].notna()
        )

        return result.reset_index(drop=True)

    @classmethod
    def load_pipeline_masters(
        cls,
        extracted_files: list[Path],
    ) -> pd.DataFrame:
        """Load headerless BSEE pipeline master attributes."""

        path = cls._first_data_file(extracted_files)
        frame = cls._read_headerless_delimited(
            path,
            PIPELINE_MASTER_COLUMNS,
        )

        for column in (
            "aban_aprv_dt",
            "aban_date",
            "approved_date",
            "ppl_const_date",
            "last_rev_date",
            "init_hs_dt",
            "prpsl_con_dt",
        ):
            frame[column] = pd.to_datetime(
                frame[column],
                format="%Y%m%d",
                errors="coerce",
            )

        frame["segment_num"] = pd.to_numeric(
            frame["segment_num"],
            errors="coerce",
        ).astype("Int64")

        frame["recv_seg_num"] = pd.to_numeric(
            frame["recv_seg_num"],
            errors="coerce",
        ).astype("Int64")

        frame["status_label"] = (
            frame["status_code"].astype("string").str.upper().map(PIPELINE_STATUS)
        )
        frame["is_currently_active"] = (
            frame["status_code"].astype("string").str.upper().eq("ACT")
        )

        frame["prod_code"] = frame["prod_code"].astype("string").str.upper()
        frame["is_crude_oil_related"] = frame["prod_code"].isin(CRUDE_OIL_PRODUCT_CODES)
        frame["is_ngl"] = frame["prod_code"].eq("NGL")

        return frame

    @classmethod
    def load_pipeline_histories(
        cls,
        extracted_files: list[Path],
    ) -> pd.DataFrame:
        """Load headerless dated BSEE pipeline history events."""

        path = cls._first_data_file(extracted_files)
        frame = cls._read_headerless_delimited(
            path,
            PIPELINE_HISTORY_COLUMNS,
        )

        for column in ("history_date", "last_rev_date"):
            frame[column] = pd.to_datetime(
                frame[column],
                format="%m/%d/%Y",
                errors="coerce",
            )

        frame["segment_num"] = pd.to_numeric(
            frame["segment_num"],
            errors="coerce",
        ).astype("Int64")

        frame["history_code"] = frame["history_code"].astype("string").str.upper()
        frame["history_event"] = frame["history_code"].map(PIPELINE_HISTORY_EVENT)
        frame["history_code_documented"] = frame["history_code"].isin(
            PIPELINE_HISTORY_EVENT
        )

        frame["service_state_signal"] = pd.NA
        frame.loc[
            frame["history_code"].eq("U"),
            "service_state_signal",
        ] = "active"
        frame.loc[
            frame["history_code"].isin(["O", "T"]),
            "service_state_signal",
        ] = "inactive"
        frame.loc[
            frame["history_code"].eq("B"),
            "service_state_signal",
        ] = "abandonment_related"

        return frame
