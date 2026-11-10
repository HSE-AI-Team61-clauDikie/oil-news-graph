from pathlib import Path

import pandas as pd
import pytest

from news.storage import write_atomic


def test_failed_write_preserves_previous_file_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data.parquet"
    write_atomic(pd.DataFrame({"value": [1]}), path)

    def broken_write(_, tmp, **__) -> None:
        Path(tmp).write_bytes(b"interrupted parquet write")
        msg = "disk write interrupted"
        raise OSError(msg)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken_write)
    with pytest.raises(OSError, match="interrupted"):
        write_atomic(pd.DataFrame({"value": [2]}), path)

    assert pd.read_parquet(path)["value"].tolist() == [1]
    assert not list(tmp_path.glob("*.tmp"))


def test_overlapping_writers_use_separate_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data.parquet"
    first = pd.DataFrame({"value": [1]})
    second = pd.DataFrame({"value": [2]})
    original = pd.DataFrame.to_parquet

    def overlapping_write(frame, tmp, **kwargs) -> None:
        original(frame, tmp, **kwargs)
        if frame is first:
            # A second writer publishes after the first has written its temp,
            # but before the first publishes it.
            write_atomic(second, path)
            assert pd.read_parquet(path)["value"].tolist() == [2]

    monkeypatch.setattr(pd.DataFrame, "to_parquet", overlapping_write)
    write_atomic(first, path)

    assert pd.read_parquet(path)["value"].tolist() == [1]
    assert not list(tmp_path.glob("*.tmp"))
