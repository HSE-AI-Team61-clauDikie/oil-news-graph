from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = (
    "https://gis.boem.gov/server/rest/services/BOEM_BSEE/MMC_Layers/FeatureServer"
)

PLATFORMS_LAYER = 0
PIPELINES_LAYER = 2
PLANNING_AREAS_LAYER = 20

logger = logging.getLogger(__name__)


class BoemClient:
    """Client for BOEM/BSEE ArcGIS FeatureServer layers."""

    def __init__(
        self,
        timeout: int = 120,
        page_size: int = 5_000,
    ) -> None:
        self.timeout = timeout
        self.page_size = page_size
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    def _request_json(
        self,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        if not response.ok:
            body = response.text[:1500]
            msg = f"BOEM ArcGIS returned HTTP {response.status_code}. Response: {body}"
            raise requests.HTTPError(msg, response=response)

        payload = response.json()

        if "error" in payload:
            msg = f"BOEM ArcGIS error: {payload['error']}"
            raise RuntimeError(msg)

        return payload

    def fetch_geojson(
        self,
        layer_id: int,
        *,
        where: str = "1=1",
        out_fields: str = "*",
    ) -> dict[str, Any]:
        """Download all features from one ArcGIS layer as GeoJSON."""

        url = f"{BASE_URL}/{layer_id}/query"
        offset = 0
        features: list[dict[str, Any]] = []

        while True:
            payload = self._request_json(
                url,
                params={
                    "where": where,
                    "outFields": out_fields,
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "resultOffset": offset,
                    "resultRecordCount": self.page_size,
                    "f": "geojson",
                },
            )

            page = payload.get("features", [])
            if not page:
                break

            features.extend(page)
            offset += len(page)

            logger.info(
                "BOEM layer %s: downloaded %s features",
                layer_id,
                f"{len(features):,}",
            )

            if len(page) < self.page_size:
                break

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    @staticmethod
    def points_to_frame(
        geojson: dict[str, Any],
    ) -> pd.DataFrame:
        """Flatten a point FeatureCollection to a DataFrame."""

        rows = []

        for feature in geojson.get("features", []):
            properties = dict(feature.get("properties") or {})
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates")

            if (
                geometry.get("type") == "Point"
                and isinstance(coordinates, list)
                and len(coordinates) >= 2
            ):
                properties["longitude"] = coordinates[0]
                properties["latitude"] = coordinates[1]

            rows.append(properties)

        return pd.DataFrame(rows)

    @staticmethod
    def save_geojson(
        geojson: dict[str, Any],
        path: Path,
    ) -> None:
        """Save GeoJSON to disk."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(geojson, ensure_ascii=False),
            encoding="utf-8",
        )

    def fetch_platforms(
        self,
        *,
        active_only: bool = False,
    ) -> dict[str, Any]:
        """Download OCS drilling platforms."""

        where = "REMOVAL_DATE IS NULL" if active_only else "1=1"
        return self.fetch_geojson(
            PLATFORMS_LAYER,
            where=where,
        )

    def fetch_pipelines(self) -> dict[str, Any]:
        """Download offshore oil and gas pipelines."""

        return self.fetch_geojson(PIPELINES_LAYER)

    def fetch_planning_areas(self) -> dict[str, Any]:
        """Download BOEM oil and gas planning areas."""

        return self.fetch_geojson(PLANNING_AREAS_LAYER)
