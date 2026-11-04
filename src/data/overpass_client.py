from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
import requests

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

logger = logging.getLogger(__name__)

REGION_BBOXES = {
    "gulf_coast": (24.0, -99.0, 31.5, -79.0),
    "midcontinent": (31.0, -104.0, 40.5, -90.0),
    "west_coast": (31.5, -125.0, 49.5, -116.0),
    "east_coast": (25.0, -82.0, 47.5, -66.0),
}


class OverpassClient:
    """Overpass API client for infrastructure gap filling."""

    def __init__(
        self,
        timeout: int = 180,
        pause: float = 3.0,
        max_attempts_per_endpoint: int = 5,
    ) -> None:
        self.timeout = timeout
        self.pause = pause
        self.max_attempts_per_endpoint = max_attempts_per_endpoint

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    def _post_query(
        self,
        endpoint: str,
        query: str,
    ) -> requests.Response:
        return self.session.post(
            endpoint,
            data={"data": query},
            timeout=self.timeout,
        )

    def _query_endpoint(
        self,
        endpoint: str,
        query: str,
    ) -> dict[str, Any] | None:
        for attempt in range(1, self.max_attempts_per_endpoint + 1):
            try:
                response = self._post_query(endpoint, query)
            except requests.RequestException as exc:
                logger.warning(
                    "Overpass request failed on %s (attempt %s/%s): %s",
                    endpoint,
                    attempt,
                    self.max_attempts_per_endpoint,
                    exc,
                )
                time.sleep(self.pause * attempt)
                continue

            if response.ok:
                logger.info("Overpass query succeeded on %s", endpoint)
                return response.json()

            if response.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Overpass %s returned HTTP %s (attempt %s/%s)",
                    endpoint,
                    response.status_code,
                    attempt,
                    self.max_attempts_per_endpoint,
                )
                time.sleep(self.pause * attempt)
                continue

            body = response.text[:1500]
            msg = (
                f"Overpass returned HTTP {response.status_code} "
                f"from {endpoint}. Response: {body}"
            )
            raise requests.HTTPError(msg, response=response)

        return None

    def _query(self, query: str) -> dict[str, Any]:
        errors: list[str] = []

        for endpoint in OVERPASS_ENDPOINTS:
            logger.info("Trying Overpass endpoint: %s", endpoint)

            try:
                payload = self._query_endpoint(endpoint, query)
            except requests.RequestException as exc:
                errors.append(f"{endpoint}: {exc}")
                continue

            if payload is not None:
                time.sleep(self.pause)
                return payload

            errors.append(f"{endpoint}: exhausted retries")

        details = "; ".join(errors)
        msg = f"All Overpass endpoints failed. Details: {details}"
        raise RuntimeError(msg)

    @staticmethod
    def _bbox_text(
        bbox: tuple[float, float, float, float],
    ) -> str:
        south, west, north, east = bbox
        return f"{south},{west},{north},{east}"

    @staticmethod
    def _to_frame(
        payload: dict[str, Any],
        *,
        category: str,
        region: str,
    ) -> pd.DataFrame:
        rows = []

        for element in payload.get("elements", []):
            center = element.get("center") or {}
            tags = element.get("tags") or {}

            latitude = element.get("lat", center.get("lat"))
            longitude = element.get("lon", center.get("lon"))

            rows.append(
                {
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                    "name": tags.get("name"),
                    "operator": tags.get("operator"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "category": category,
                    "region": region,
                    "tags": json.dumps(tags, ensure_ascii=False),
                }
            )

        return pd.DataFrame(rows)

    def fetch_refineries(
        self,
        region: str,
    ) -> pd.DataFrame:
        """Fetch OSM objects tagged as refineries."""

        bbox = self._bbox_text(REGION_BBOXES[region])

        query = f"""
        [out:json][timeout:180];
        (
          nwr["industrial"="refinery"]({bbox});
          nwr["man_made"="works"]["product"~"oil|petroleum"]({bbox});
        );
        out center tags;
        """

        return self._to_frame(
            self._query(query),
            category="refinery",
            region=region,
        )

    def fetch_oil_storage(
        self,
        region: str,
    ) -> pd.DataFrame:
        """Fetch explicitly oil-tagged storage and terminal objects."""

        bbox = self._bbox_text(REGION_BBOXES[region])

        query = f"""
        [out:json][timeout:180];
        (
          nwr["man_made"="storage_tank"]["content"~"oil|crude"]({bbox});
          nwr["man_made"="storage_tank"]["substance"~"oil|crude"]({bbox});
          nwr["industrial"="oil"]({bbox});
        );
        out center tags;
        """

        return self._to_frame(
            self._query(query),
            category="oil_storage_or_terminal",
            region=region,
        )
