"""Replay synthetic SEC tasks without network access or model weights."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from auditops.pipeline import connect_db, process_zip
from auditops.runtime import execute_task_plan
from auditops.tasks import build_task_plan_target, build_task_specs


def main() -> None:
    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    with TemporaryDirectory(prefix="auditops-smoke-") as directory:
        root = Path(directory)
        database = root / "filings.sqlite"
        for index, name in enumerate(("filing_10k", "filing_q1", "filing_q2")):
            archive = root / f"{name}.zip"
            with ZipFile(archive, "w") as output:
                for source in sorted((fixtures / name).iterdir()):
                    output.write(source, source.name)
            process_zip(
                zip_path=str(archive),
                out_db=str(database),
                extract_narrative=True,
                reset_db=index == 0,
            )
        connection = connect_db(str(database))
        try:
            tasks = build_task_specs(connection)
            if not tasks:
                raise RuntimeError("Synthetic fixture generated no tasks")
            for task in tasks:
                answer = execute_task_plan(connection, build_task_plan_target(task))
                if answer != task["target_answer"]:
                    raise AssertionError(f"Replay mismatch: {task['task_id']}")
            print(f"Synthetic smoke passed: {len(tasks)} deterministic task replays")
        finally:
            connection.close()


if __name__ == "__main__":
    main()
