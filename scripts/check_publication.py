"""Check publishable files, secrets, schemas and local documentation links.

Run from a source checkout. Optional forbidden terms are checked case-insensitively
without printing matched content. Ignored runtime artifacts are never publication inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import jsonschema

from auditops.provenance import scan_artifacts_for_secrets


def publication_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    return [root / name for name in names if (root / name).is_file()]


def check(root: Path, forbidden_terms: list[str]) -> list[str]:
    files = publication_files(root)
    failures = []
    for hit in scan_artifacts_for_secrets(files):
        failures.append(
            f"{Path(hit['path']).relative_to(root)}: {hit['code']} ({hit['source']})"
        )
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append(f"{relative}: symlink in publication")
            continue
        text = path.read_text(encoding="utf-8-sig")
        for term in forbidden_terms:
            if term.casefold() in (relative + "\n" + text).casefold():
                failures.append(f"{relative}: forbidden publication term")
        if path.suffix in {".sh", ".sbatch"} and b"\r" in path.read_bytes():
            failures.append(f"{relative}: shell file is not LF-only")
        if path.name.endswith(".schema.json"):
            try:
                jsonschema.Draft202012Validator.check_schema(json.loads(text))
            except (ValueError, jsonschema.SchemaError):
                failures.append(f"{relative}: invalid JSON schema")
        if path.suffix not in {".md", ".html"}:
            continue
        links = (
            re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
            if path.suffix == ".md"
            else re.findall(r'(?:href|src)="([^"]+)"', text)
        )
        for link in links:
            link = link.strip("<>")
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                failures.append(f"{relative}: broken or nonportable documentation link")
    return sorted(set(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbidden-term", action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = check(root, args.forbidden_term)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Publication checks passed: {len(publication_files(root))} files")


if __name__ == "__main__":
    main()
