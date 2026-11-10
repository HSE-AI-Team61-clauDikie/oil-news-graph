from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Записать parquet атомарно, с отдельным временным файлом для каждого writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
