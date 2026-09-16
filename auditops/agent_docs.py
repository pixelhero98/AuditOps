"""Deterministically render registry- and schema-backed agent documentation.

The checked-in HTML remains easy to read and review, while generated fragments keep
operation routing and public contract vocabularies aligned with executable sources.
Run ``python -m auditops.agent_docs --check`` in CI or ``--write`` after an intentional
registry/schema change.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from html import escape
from pathlib import Path

from .agent_operations import TASK_OPERATION_SPECS

BEGIN_MARKER = "<!-- BEGIN GENERATED: {name} -->"
END_MARKER = "<!-- END GENERATED: {name} -->"


def _load_schema(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Schema must be a JSON object: {path}")
    return value


def _required_string(value: object, *, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value


def _required_string_list(value: object, *, field: str, source: Path) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{source}: {field} must be a non-empty string array")
    return list(value)


def _schema_enum(schema: Mapping[str, object], field: str, source: Path) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise TypeError(f"{source}: properties must be an object")
    definition = properties.get(field)
    if not isinstance(definition, dict):
        raise TypeError(f"{source}: properties.{field} must be an object")
    return _required_string_list(
        definition.get("enum"), field=f"properties.{field}.enum", source=source
    )


def _argument_names(
    schema: Mapping[str, object], schema_id: str, source: Path
) -> list[str]:
    prefix = "tool_arguments.v2.2#/$defs/"
    if not schema_id.startswith(prefix):
        raise ValueError(f"Unsupported tool argument schema ID: {schema_id}")
    definition_name = schema_id.removeprefix(prefix)
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise TypeError(f"{source}: $defs must be an object")
    definition = definitions.get(definition_name)
    if definition is None:
        raise ValueError(f"{source}: missing $defs.{definition_name}")
    if not isinstance(definition, dict):
        raise TypeError(f"{source}: $defs.{definition_name} must be an object")
    return _required_string_list(
        definition.get("required"),
        field=f"$defs.{definition_name}.required",
        source=source,
    )


def _code(value: str) -> str:
    return f"<code>{escape(value)}</code>"


def render_operation_registry_table(repo_root: Path) -> str:
    """Render the exact task-operation table from the registry and argument schema."""

    argument_path = repo_root / "auditops" / "specs" / "tool_arguments.v2.2.schema.json"
    argument_schema = _load_schema(argument_path)
    lines = [
        '        <div class="table-scroll">',
        "          <table>",
        "            <thead>",
        "              <tr><th>Task type</th><th>Runtime ID</th><th>Operation</th><th>Argument schema and required fields</th><th>Observation schema</th><th>Terminal output schema</th></tr>",
        "            </thead>",
        "            <tbody>",
    ]
    for spec in TASK_OPERATION_SPECS.values():
        arguments = ", ".join(
            _code(name)
            for name in _argument_names(
                argument_schema, spec.tool_argument_schema_id, argument_path
            )
        )
        lines.extend(
            [
                "              <tr>",
                f"                <td>{_code(spec.task_type)}</td>",
                f"                <td>{_code(spec.runtime_id)}</td>",
                f"                <td>{_code(spec.operation)}</td>",
                f'                <td>{_code(spec.tool_argument_schema_id)}<br><span class="small">{arguments}</span></td>',
                f"                <td>{_code(spec.observation_schema_id)}</td>",
                f"                <td>{_code(spec.terminal_output_schema_id)}</td>",
                "              </tr>",
            ]
        )
    lines.extend(["            </tbody>", "          </table>", "        </div>"])
    return "\n".join(lines)


def render_schema_catalog_table(repo_root: Path) -> str:
    """Render the version-suffixed public schema catalog from committed snapshots."""

    specs_dir = repo_root / "auditops" / "specs"
    rows: list[tuple[str, str, str]] = []
    for path in sorted(specs_dir.glob("*.v*.schema.json")):
        # v2.3 has a separate, versioned schema-boundary document. Keeping it
        # out of the immutable v2.2 catalog prevents a later profile from
        # silently changing the meaning of the older documentation snapshot.
        if path.name.endswith(".v2.3.schema.json"):
            continue
        schema = _load_schema(path)
        schema_id = _required_string(schema.get("$id"), field="$id", source=path)
        title = _required_string(schema.get("title"), field="title", source=path)
        rows.append((schema_id, title, path.name))
    if not rows:
        raise ValueError(f"No version-suffixed public schemas found in {specs_dir}")
    rows.sort(key=lambda row: (row[0], row[2]))
    lines = [
        '        <div class="table-scroll">',
        "          <table>",
        "            <thead><tr><th>Public interface</th><th>Schema ID</th><th>Committed snapshot</th></tr></thead>",
        "            <tbody>",
    ]
    for schema_id, title, filename in rows:
        href = f"../auditops/specs/{filename}"
        lines.append(
            "              <tr>"
            f"<td>{escape(title)}</td>"
            f"<td>{_code(schema_id)}</td>"
            f'<td><a href="{escape(href, quote=True)}">{_code(filename)}</a></td>'
            "</tr>"
        )
    lines.extend(["            </tbody>", "          </table>", "        </div>"])
    return "\n".join(lines)


_OUTCOME_MEANINGS = {
    "RELEASED": ("A final answer passed every deterministic release check.", "Yes"),
    "SAFE_REFUSAL": (
        "The refusal is allowed and safe to expose. Necessity is scored offline.",
        "Yes",
    ),
    "MODEL_FAILURE": (
        "Unusable or unsupported model output, a non-repairable proposal, or exhausted repair.",
        "No",
    ),
    "INFRASTRUCTURE_FAILURE": (
        "Context overflow, timeout, OOM, backend, or model-adapter failure.",
        "No",
    ),
    "INTEGRITY_FAILURE": (
        "Artifact/hash/task binding, overwrite, deterministic replay, or tool-observation mismatch.",
        "No; hard gate",
    ),
}


def render_terminal_outcome_table(repo_root: Path) -> str:
    """Render terminal outcomes after checking case/run schema agreement."""

    specs_dir = repo_root / "auditops" / "specs"
    case_path = specs_dir / "agent_case_result.v2.2.schema.json"
    run_path = specs_dir / "agent_run_record.v2.2.schema.json"
    outcomes = _schema_enum(_load_schema(case_path), "outcome", case_path)
    run_outcomes = _schema_enum(_load_schema(run_path), "outcome", run_path)
    if outcomes != run_outcomes:
        raise ValueError(
            "AgentCaseResultV2.2 and AgentRunRecordV2.2 outcome enums differ"
        )
    if set(outcomes) != set(_OUTCOME_MEANINGS):
        raise ValueError("Terminal outcome descriptions do not cover the schema enum")
    lines = [
        '        <div class="table-scroll">',
        "          <table>",
        "            <thead><tr><th>Outcome</th><th>Use</th><th>Releaseable?</th></tr></thead>",
        "            <tbody>",
    ]
    for outcome in outcomes:
        meaning, releaseable = _OUTCOME_MEANINGS[outcome]
        lines.append(
            "              <tr>"
            f"<td>{_code(outcome)}</td>"
            f"<td>{escape(meaning)}</td>"
            f"<td>{escape(releaseable)}</td>"
            "</tr>"
        )
    lines.extend(["            </tbody>", "          </table>", "        </div>"])
    return "\n".join(lines)


FragmentRenderer = Callable[[Path], str]
DOCUMENT_FRAGMENTS: Mapping[str, Mapping[str, FragmentRenderer]] = {
    "docs/agent-tool-schema.html": {
        "operation-registry": render_operation_registry_table,
        "schema-catalog": render_schema_catalog_table,
    },
    "docs/agent-runtime-protocol.html": {
        "terminal-outcomes": render_terminal_outcome_table,
    },
}


def replace_generated_fragment(document: str, name: str, fragment: str) -> str:
    """Replace exactly one named generated region without touching surrounding prose."""

    begin = BEGIN_MARKER.format(name=name)
    end = END_MARKER.format(name=name)
    if document.count(begin) != 1 or document.count(end) != 1:
        raise ValueError(f"Document must contain exactly one {name!r} marker pair")
    begin_offset = document.index(begin)
    line_start = document.rfind("\n", 0, begin_offset) + 1
    indentation = document[line_start:begin_offset]
    if indentation.strip():
        raise ValueError(f"Generated marker {name!r} must start on its own line")
    prefix, remainder = document.split(begin, 1)
    _old_fragment, suffix = remainder.split(end, 1)
    return f"{prefix}{begin}\n{fragment.rstrip()}\n{indentation}{end}{suffix}"


def expected_document(path: Path, repo_root: Path) -> str:
    """Return one document with every registered generated region refreshed."""

    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    renderers = DOCUMENT_FRAGMENTS.get(relative)
    if renderers is None:
        raise ValueError(f"Document is not registered for generation: {relative}")
    document = path.read_text(encoding="utf-8")
    for name, renderer in renderers.items():
        document = replace_generated_fragment(document, name, renderer(repo_root))
    return document


def generated_docs_are_current(repo_root: Path) -> list[Path]:
    """Return registered documents whose checked-in generated regions have drifted."""

    stale: list[Path] = []
    for relative in DOCUMENT_FRAGMENTS:
        path = repo_root / relative
        if path.read_text(encoding="utf-8") != expected_document(path, repo_root):
            stale.append(path)
    return stale


def write_generated_docs(repo_root: Path) -> None:
    """Refresh registered documents using LF newlines and UTF-8."""

    for relative in DOCUMENT_FRAGMENTS:
        path = repo_root / relative
        path.write_text(
            expected_document(path, repo_root), encoding="utf-8", newline="\n"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or refresh registry/schema-backed AuditOps HTML fragments."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="fail when HTML is stale")
    action.add_argument("--write", action="store_true", help="refresh checked-in HTML")
    parser.add_argument("--repo-root", type=Path, help="AuditOps repository root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    if args.write:
        write_generated_docs(repo_root)
        return 0
    stale = generated_docs_are_current(repo_root)
    if stale:
        for path in stale:
            print(f"stale generated documentation: {path.relative_to(repo_root)}")
        print("run: python -m auditops.agent_docs --write")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
