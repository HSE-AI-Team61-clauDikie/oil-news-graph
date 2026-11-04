from __future__ import annotations

import io
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PIPELINE_DATA_SHEET = "Data"
PIPELINE_OWNERSHIP_SHEET = "Oil & NGL Pipeline Ownership"
ENTITIES_SHEET = "All Entities"
ENTITY_OWNERSHIP_SHEET = "Entity Ownership"

PIPELINE_WORKBOOK_PATTERN = "GEM-GOIT-Oil-NGL-Pipelines-*.xlsx"
OWNERSHIP_WORKBOOK_PATTERN = "Global-Energy-Ownership-Tracker-*.xlsx"
GEOJSON_PATTERN = "GEM-GOIT-Oil-NGL-Pipelines-*.geojson"

NUMERIC_PIPELINE_COLUMNS = {
    "capacity",
    "capacity_boed",
    "length_known",
    "length_known_km",
    "length_estimate_km",
    "length_merged_km",
    "cost_usd",
    "cost_euro",
    "cost",
}

YEAR_COLUMNS = {
    "proposal_year",
    "construction_year",
    "start_year1",
    "start_year2",
    "start_year3",
    "cancelled_year",
    "stop_year",
    "shelved_year",
}

US_PATTERN = r"(?:United States|USA|U\.S\.)"

type JsonValue = (
    str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
)


@dataclass(frozen=True)
class GemFiles:
    """Resolved files for one GEM GOIT download bundle."""

    pipeline_workbook: Path
    pipeline_geojson: Path
    ownership_workbook: Path


def _snake_case(value: str) -> str:
    value = value.strip()
    value = value.replace("%", " percent ")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.lower().strip("_")


def _clean_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in result.select_dtypes(include=["object"]).columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "--": pd.NA})
        )

    return result


def _coordinate_pairs(
    value: JsonValue,
) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []

    if (
        len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        return [(float(value[0]), float(value[1]))]

    points: list[tuple[float, float]] = []

    for item in value:
        points.extend(_coordinate_pairs(item))

    return points


def _geometry_record(
    geometry: dict[str, JsonValue] | None,
) -> dict[str, object]:
    if not geometry:
        return {
            "geometry_json": None,
            "geometry_type": None,
            "route_has_geometry": False,
            "route_point_count": 0,
            "route_start_longitude": pd.NA,
            "route_start_latitude": pd.NA,
            "route_end_longitude": pd.NA,
            "route_end_latitude": pd.NA,
            "route_min_longitude": pd.NA,
            "route_min_latitude": pd.NA,
            "route_max_longitude": pd.NA,
            "route_max_latitude": pd.NA,
        }

    points = _coordinate_pairs(geometry.get("coordinates"))

    if not points:
        start_longitude = pd.NA
        start_latitude = pd.NA
        end_longitude = pd.NA
        end_latitude = pd.NA
        min_longitude = pd.NA
        min_latitude = pd.NA
        max_longitude = pd.NA
        max_latitude = pd.NA
    else:
        longitudes = [point[0] for point in points]
        latitudes = [point[1] for point in points]

        start_longitude, start_latitude = points[0]
        end_longitude, end_latitude = points[-1]
        min_longitude = min(longitudes)
        min_latitude = min(latitudes)
        max_longitude = max(longitudes)
        max_latitude = max(latitudes)

    return {
        "geometry_json": json.dumps(
            geometry,
            ensure_ascii=False,
        ),
        "geometry_type": geometry.get("type"),
        "route_has_geometry": bool(points),
        "route_point_count": len(points),
        "route_start_longitude": start_longitude,
        "route_start_latitude": start_latitude,
        "route_end_longitude": end_longitude,
        "route_end_latitude": end_latitude,
        "route_min_longitude": min_longitude,
        "route_min_latitude": min_latitude,
        "route_max_longitude": max_longitude,
        "route_max_latitude": max_latitude,
    }


def _unique_json(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return json.dumps(
        cleaned,
        ensure_ascii=False,
    )


class GemGoitParser:
    """Parser for GEM Global Oil Infrastructure Tracker downloads."""

    def __init__(self, source_dir: Path) -> None:
        self.source_dir = source_dir
        self.source_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _copy_zip_member(
        archive: zipfile.ZipFile,
        member: str,
        target_path: Path,
    ) -> Path:
        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            archive.open(member) as source,
            target_path.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)

        return target_path

    def prepare_download_zip(
        self,
        download_zip: Path,
    ) -> GemFiles:
        """Extract only the required GEM files from the downloaded ZIP."""

        if not download_zip.exists():
            msg = f"GEM download ZIP not found: {download_zip}"
            raise FileNotFoundError(msg)

        with zipfile.ZipFile(download_zip) as outer:
            names = outer.namelist()

            pipeline_member = next(
                (
                    name
                    for name in names
                    if re.search(
                        r"GEM-GOIT-Oil-NGL-Pipelines-.*\.xlsx$",
                        name,
                    )
                ),
                None,
            )
            ownership_member = next(
                (
                    name
                    for name in names
                    if re.search(
                        r"Global-Energy-Ownership-Tracker-.*\.xlsx$",
                        name,
                    )
                ),
                None,
            )
            spatial_zip_member = next(
                (
                    name
                    for name in names
                    if re.search(
                        r"GEM-GOIT-Oil-NGL-Pipelines-.*corrected\.zip$",
                        name,
                    )
                ),
                None,
            )

            required = {
                "pipeline workbook": pipeline_member,
                "ownership workbook": ownership_member,
                "spatial ZIP": spatial_zip_member,
            }
            missing = [label for label, member in required.items() if member is None]

            if missing:
                msg = "GEM ZIP is missing required files: " + ", ".join(missing)
                raise FileNotFoundError(msg)

            pipeline_path = self.source_dir / Path(str(pipeline_member)).name
            ownership_path = self.source_dir / Path(str(ownership_member)).name

            self._copy_zip_member(
                outer,
                str(pipeline_member),
                pipeline_path,
            )
            self._copy_zip_member(
                outer,
                str(ownership_member),
                ownership_path,
            )

            spatial_bytes = outer.read(str(spatial_zip_member))

        with zipfile.ZipFile(io.BytesIO(spatial_bytes)) as spatial:
            geojson_member = next(
                (
                    name
                    for name in spatial.namelist()
                    if name.endswith(".geojson") and not name.startswith("__MACOSX/")
                ),
                None,
            )

            if geojson_member is None:
                msg = "GeoJSON not found inside GEM spatial ZIP."
                raise FileNotFoundError(msg)

            geojson_path = self.source_dir / Path(geojson_member).name

            self._copy_zip_member(
                spatial,
                geojson_member,
                geojson_path,
            )

        return GemFiles(
            pipeline_workbook=pipeline_path,
            pipeline_geojson=geojson_path,
            ownership_workbook=ownership_path,
        )

    def discover_files(self) -> GemFiles:
        """Find an already extracted GEM bundle."""

        pipeline_files = sorted(self.source_dir.glob(PIPELINE_WORKBOOK_PATTERN))
        ownership_files = sorted(self.source_dir.glob(OWNERSHIP_WORKBOOK_PATTERN))
        geojson_files = sorted(self.source_dir.glob(GEOJSON_PATTERN))

        if not pipeline_files:
            msg = "GEM pipeline workbook not found."
            raise FileNotFoundError(msg)

        if not ownership_files:
            msg = "GEM ownership workbook not found."
            raise FileNotFoundError(msg)

        if not geojson_files:
            msg = "GEM pipeline GeoJSON not found."
            raise FileNotFoundError(msg)

        return GemFiles(
            pipeline_workbook=pipeline_files[-1],
            pipeline_geojson=geojson_files[-1],
            ownership_workbook=ownership_files[-1],
        )

    @staticmethod
    def load_pipeline_attributes(
        path: Path,
    ) -> pd.DataFrame:
        """Load the GOIT pipeline Data worksheet."""

        frame = pd.read_excel(
            path,
            sheet_name=PIPELINE_DATA_SHEET,
        )
        frame.columns = [_snake_case(str(column)) for column in frame.columns]
        frame = _clean_string_columns(frame)

        if "project_id" not in frame.columns:
            msg = "ProjectID column missing from GEM pipeline workbook."
            raise KeyError(msg)

        if frame["project_id"].duplicated().any():
            msg = "ProjectID is not unique in GEM pipeline workbook."
            raise ValueError(msg)

        for column in NUMERIC_PIPELINE_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        for column in YEAR_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).astype("Int64")

        if "last_updated" in frame.columns:
            frame["last_updated"] = pd.to_datetime(
                frame["last_updated"],
                errors="coerce",
            )

        return frame

    @staticmethod
    def load_pipeline_geometry(
        path: Path,
    ) -> pd.DataFrame:
        """Load ProjectID and route geometry from GOIT GeoJSON."""

        payload = json.loads(path.read_text(encoding="utf-8"))

        rows = []

        for feature in payload.get("features", []):
            properties = feature.get("properties") or {}
            project_id = properties.get("ProjectID")

            if project_id is None:
                continue

            row = {
                "project_id": str(project_id),
                **_geometry_record(feature.get("geometry")),
            }
            rows.append(row)

        frame = pd.DataFrame(rows)

        if frame.empty:
            msg = "No ProjectID features found in GEM GeoJSON."
            raise ValueError(msg)

        if frame["project_id"].duplicated().any():
            msg = "ProjectID is not unique in GEM GeoJSON."
            raise ValueError(msg)

        return frame

    @staticmethod
    def load_pipeline_ownership(
        path: Path,
    ) -> pd.DataFrame:
        """Load pipeline-level ownership paths."""

        frame = pd.read_excel(
            path,
            sheet_name=PIPELINE_OWNERSHIP_SHEET,
        )
        frame.columns = [_snake_case(str(column)) for column in frame.columns]
        frame = _clean_string_columns(frame)

        numeric_columns = {
            "share",
            "capacity_boed",
        }

        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        return frame

    @staticmethod
    def load_entities(
        path: Path,
    ) -> pd.DataFrame:
        """Load GEM company/entity master table."""

        frame = pd.read_excel(
            path,
            sheet_name=ENTITIES_SHEET,
        )
        frame.columns = [_snake_case(str(column)) for column in frame.columns]
        return _clean_string_columns(frame)

    @staticmethod
    def load_entity_ownership(
        path: Path,
    ) -> pd.DataFrame:
        """Load entity-to-entity ownership edges."""

        frame = pd.read_excel(
            path,
            sheet_name=ENTITY_OWNERSHIP_SHEET,
        )
        frame.columns = [_snake_case(str(column)) for column in frame.columns]
        frame = _clean_string_columns(frame)

        share_column = "percent_share_of_ownership"

        if share_column in frame.columns:
            frame[share_column] = pd.to_numeric(
                frame[share_column],
                errors="coerce",
            )

        return frame

    @staticmethod
    def _add_us_flag(
        pipelines: pd.DataFrame,
    ) -> pd.DataFrame:
        result = pipelines.copy()

        country_columns = [
            column
            for column in (
                "countries_or_areas",
                "start_country_or_area",
                "end_country_or_area",
            )
            if column in result.columns
        ]

        is_us = pd.Series(
            data=False,
            index=result.index,
            dtype=bool,
        )

        for column in country_columns:
            is_us |= (
                result[column]
                .astype("string")
                .str.contains(
                    US_PATTERN,
                    case=False,
                    regex=True,
                    na=False,
                )
            )

        result["is_us"] = is_us
        return result

    @staticmethod
    def build_ownership_summary(
        ownership: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate one-to-many pipeline ownership to ProjectID."""

        if ownership.empty:
            return pd.DataFrame(columns=["project_id"])

        grouped = ownership.groupby(
            "project_id",
            dropna=False,
        )

        summary = grouped.agg(
            ownership_rows=("project_id", "size"),
            parent_count=("parent", "nunique"),
            immediate_owner_count=(
                "immediate_project_owner",
                "nunique",
            ),
            max_parent_share_pct=("share", "max"),
        ).reset_index()

        parent_names = (
            grouped["parent"]
            .apply(_unique_json)
            .rename("parent_names_json")
            .reset_index()
        )
        immediate_owners = (
            grouped["immediate_project_owner"]
            .apply(_unique_json)
            .rename("immediate_owner_names_json")
            .reset_index()
        )

        return summary.merge(
            parent_names,
            on="project_id",
            how="left",
            validate="one_to_one",
        ).merge(
            immediate_owners,
            on="project_id",
            how="left",
            validate="one_to_one",
        )

    def build_tables(
        self,
        files: GemFiles | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build normalized pipeline, ownership, and entity tables."""

        resolved = files or self.discover_files()

        attributes = self.load_pipeline_attributes(resolved.pipeline_workbook)
        geometry = self.load_pipeline_geometry(resolved.pipeline_geojson)
        ownership = self.load_pipeline_ownership(resolved.ownership_workbook)
        entities = self.load_entities(resolved.ownership_workbook)
        entity_ownership = self.load_entity_ownership(resolved.ownership_workbook)

        attribute_ids = set(attributes["project_id"].astype(str))
        geometry_ids = set(geometry["project_id"].astype(str))

        missing_geometry = attribute_ids - geometry_ids
        geometry_only = geometry_ids - attribute_ids

        if missing_geometry:
            logger.warning(
                "%s GEM projects have no GeoJSON record.",
                len(missing_geometry),
            )

        if geometry_only:
            logger.warning(
                "%s GeoJSON projects are missing from Excel.",
                len(geometry_only),
            )

        pipelines = attributes.merge(
            geometry,
            on="project_id",
            how="left",
            validate="one_to_one",
        )
        pipelines = self._add_us_flag(pipelines)

        ownership_summary = self.build_ownership_summary(ownership)

        pipelines = pipelines.merge(
            ownership_summary,
            on="project_id",
            how="left",
            validate="one_to_one",
        )

        us_project_ids = set(
            pipelines.loc[
                pipelines["is_us"],
                "project_id",
            ].astype(str)
        )

        ownership_us = ownership[
            ownership["project_id"].astype(str).isin(us_project_ids)
        ].reset_index(drop=True)

        return {
            "pipelines_all": pipelines,
            "pipelines_us": pipelines[pipelines["is_us"]].reset_index(drop=True),
            "pipeline_ownership_all": (ownership.reset_index(drop=True)),
            "pipeline_ownership_us": ownership_us,
            "entities": entities.reset_index(drop=True),
            "entity_ownership": (entity_ownership.reset_index(drop=True)),
        }

    @staticmethod
    def save_tables(
        tables: dict[str, pd.DataFrame],
        target_dir: Path,
    ) -> None:
        """Save normalized GEM tables as Parquet."""

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, frame in tables.items():
            frame.to_parquet(
                target_dir / f"gem_{name}.parquet",
                index=False,
            )
