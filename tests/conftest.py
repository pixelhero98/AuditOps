from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from auditops.pipeline import connect_db, process_zip

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def build_fixture_zip(tmp_path: Path, fixture_name: str) -> Path:
    source_dir = FIXTURE_ROOT / fixture_name
    zip_path = tmp_path / f"{fixture_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(source_dir.iterdir()):
            zf.write(path, arcname=path.name)
    return zip_path


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "auditops.sqlite"
    for index, fixture_name in enumerate(["filing_10k", "filing_q1", "filing_q2"]):
        process_zip(
            zip_path=str(build_fixture_zip(tmp_path, fixture_name)),
            out_db=str(db_path),
            extract_narrative=True,
            reset_db=(index == 0),
        )
    return db_path


@pytest.fixture
def db_conn(populated_db: Path):
    conn = connect_db(str(populated_db))
    try:
        yield conn
    finally:
        conn.close()
