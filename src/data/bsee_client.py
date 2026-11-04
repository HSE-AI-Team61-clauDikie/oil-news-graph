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

logger = logging.getLogger(__name__)


class BseeClient:
    """Downloader and parser for BSEE bulk production data."""

    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "oil-news-graph-coursework/1.0",
            }
        )

    def download_production_by_platform(
        self,
        target_path: Path,
    ) -> Path:
        """Download the BSEE production-by-platform ZIP archive."""

        target_path.parent.mkdir(parents=True, exist_ok=True)

        response = self.session.get(
            PRODUCTION_BY_PLATFORM_URL,
            timeout=self.timeout,
        )
        response.raise_for_status()
        target_path.write_bytes(response.content)

        logger.info(
            "Downloaded BSEE archive: %s bytes",
            f"{target_path.stat().st_size:,}",
        )
        return target_path

    @staticmethod
    def extract_archive(
        archive_path: Path,
        target_dir: Path,
    ) -> list[Path]:
        """Extract ZIP archive and return extracted files."""

        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)

        return [path for path in target_dir.rglob("*") if path.is_file()]

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
                delimiters="|,	;^",
            )
            return dialect.delimiter
        except csv.Error:
            return "|"

    @staticmethod
    def _snake_case(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return value.strip("_")

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
        return frame

    @classmethod
    def load_production_by_platform(
        cls,
        extracted_files: list[Path],
    ) -> pd.DataFrame:
        """Load the most likely production-by-platform table."""

        candidates = [
            path
            for path in extracted_files
            if path.suffix.lower() in {".txt", ".csv", ".dat"}
        ]

        if not candidates:
            msg = "В BSEE ZIP не найден текстовый delimited-файл."
            raise FileNotFoundError(msg)

        candidates.sort(
            key=lambda path: path.stat().st_size,
            reverse=True,
        )

        frame = cls.read_delimited_file(candidates[0])

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

        return frame
