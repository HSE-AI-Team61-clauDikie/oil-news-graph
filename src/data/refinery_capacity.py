from __future__ import annotations

import logging
from pathlib import Path

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
    ) -> list[Path]:
        """Download annual snapshots for an inclusive year range."""

        return [
            self.download_year(
                year,
                target_dir,
            )
            for year in range(start_year, end_year + 1)
        ]
