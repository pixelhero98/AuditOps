from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from auditops.agent_contracts import validate_agent_task_input
from auditops.agent_docs import (
    BEGIN_MARKER,
    DOCUMENT_FRAGMENTS,
    END_MARKER,
    generated_docs_are_current,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    REPO_ROOT / "docs" / "agent-tool-schema.html",
    REPO_ROOT / "docs" / "agent-runtime-protocol.html",
)
V23_DOCS = (
    REPO_ROOT / "docs" / "agent-tool-schema-v2.3.html",
    REPO_ROOT / "docs" / "agent-runtime-protocol-v2.3.html",
)
ALLOWED_STATUSES = {"IMPLEMENTED", "PROPOSED", "DEFERRED"}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.resource_urls: list[str] = []
        self.section_statuses: list[str | None] = []
        self.json_examples: list[str] = []
        self._json_parts: list[str] | None = None
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag not in VOID_ELEMENTS:
            self.stack.append(tag)
        element_id = values.get("id")
        if element_id:
            if element_id in self.ids:
                self.errors.append(f"duplicate id: {element_id}")
            self.ids.add(element_id)
        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"] or "")
        if tag in {"img", "source", "video", "audio", "iframe"} and values.get("src"):
            self.resource_urls.append(values["src"] or "")
        if tag == "link" and values.get("href"):
            self.resource_urls.append(values["href"] or "")
        if tag == "script":
            self.scripts += 1
            if values.get("src"):
                self.resource_urls.append(values["src"] or "")
        if tag == "section":
            self.section_statuses.append(values.get("data-status"))
        if tag == "code" and "language-json" in (values.get("class") or "").split():
            if self._json_parts is not None:
                self.errors.append("nested JSON code block")
            self._json_parts = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_ELEMENTS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            self.errors.append(f"unexpected closing tag: {tag}")
        elif self.stack[-1] != tag:
            self.errors.append(f"closing {tag} while {self.stack[-1]} is open")
        else:
            self.stack.pop()
        if tag == "code" and self._json_parts is not None:
            self.json_examples.append("".join(self._json_parts))
            self._json_parts = None

    def handle_data(self, data: str) -> None:
        if self._json_parts is not None:
            self._json_parts.append(data)

    def close(self) -> None:
        super().close()
        if self.stack:
            self.errors.append(f"unclosed tags: {', '.join(self.stack)}")
        if self._json_parts is not None:
            self.errors.append("unclosed JSON code block")


def _parse(path: Path) -> _DocumentParser:
    parser = _DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


@pytest.fixture(scope="module")
def parsed_docs() -> dict[Path, _DocumentParser]:
    return {path: _parse(path) for path in DOCS}


def test_agent_docs_are_balanced_local_and_status_labelled(
    parsed_docs: dict[Path, _DocumentParser],
) -> None:
    observed_statuses: set[str] = set()
    for path, parsed in parsed_docs.items():
        assert not parsed.errors, f"{path.name}: {parsed.errors}"
        assert parsed.scripts == 0, f"{path.name} must not contain scripts"
        assert not parsed.resource_urls, f"{path.name} has external/resource URLs"
        assert parsed.section_statuses, f"{path.name} has no status-labelled sections"
        assert all(status in ALLOWED_STATUSES for status in parsed.section_statuses)
        observed_statuses.update(status for status in parsed.section_statuses if status)
    assert observed_statuses == ALLOWED_STATUSES


def test_agent_docs_have_no_broken_local_links(
    parsed_docs: dict[Path, _DocumentParser],
) -> None:
    for source, parsed in parsed_docs.items():
        for href in parsed.hrefs:
            split = urlsplit(href)
            assert not split.scheme and not split.netloc, (
                f"{source.name} must use repository-local links, found {href!r}"
            )
            target = (
                source
                if not split.path
                else (source.parent / unquote(split.path)).resolve()
            )
            assert target.exists(), f"{source.name} has broken link {href!r}"
            if split.fragment:
                target_parser = parsed_docs.get(target) or _parse(target)
                assert unquote(split.fragment) in target_parser.ids, (
                    f"{source.name} links to missing fragment {href!r}"
                )


def test_agent_docs_json_examples_parse(
    parsed_docs: dict[Path, _DocumentParser],
) -> None:
    examples = 0
    for path, parsed in parsed_docs.items():
        for index, example in enumerate(parsed.json_examples, start=1):
            try:
                json.loads(example)
            except json.JSONDecodeError as exc:
                pytest.fail(f"{path.name} JSON example {index} is invalid: {exc}")
            examples += 1
    assert examples >= 2
    validate_agent_task_input(json.loads(parsed_docs[DOCS[0]].json_examples[0]))


def test_agent_docs_keep_correct_names_and_cross_links(
    parsed_docs: dict[Path, _DocumentParser],
) -> None:
    tool_doc, runtime_doc = DOCS
    assert (
        "<title>AuditOps v2.2 — Tool and Contract Schema</title>"
        in tool_doc.read_text(encoding="utf-8")
    )
    assert "<title>AuditOps v2.2 — Runtime Protocol</title>" in runtime_doc.read_text(
        encoding="utf-8"
    )
    assert "agent-runtime-protocol.html" in parsed_docs[tool_doc].hrefs
    assert "agent-tool-schema.html" in parsed_docs[runtime_doc].hrefs
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8-sig")
    assert "](docs/agent-runtime-protocol.html)" in readme
    assert "](docs/agent-tool-schema.html)" in readme


def test_tool_doc_links_every_versioned_public_schema(
    parsed_docs: dict[Path, _DocumentParser],
) -> None:
    linked_names = {
        Path(urlsplit(href).path).name for href in parsed_docs[DOCS[0]].hrefs
    }
    expected = {
        "agent_task_input.v2.schema.json",
        "agent_proposal.v2.schema.json",
        "verifier_result.v2.schema.json",
        "agent_case_result.v2.schema.json",
        "agent_run_record.v2.schema.json",
        "tool_arguments.v2.schema.json",
        "tool_observation.v2.schema.json",
        "failure.v2.schema.json",
        "context_pack.v2.schema.json",
        "model_config.v2.schema.json",
        "quantitative_answer_or_refusal.v2.schema.json",
        "narrative_answer_or_refusal.v2.schema.json",
        "vlm_diagnostic_task.v1.schema.json",
        "vlm_diagnostic_result.v1.schema.json",
    }
    assert expected <= linked_names


def test_generated_registry_and_schema_tables_match_checked_in_html() -> None:
    assert generated_docs_are_current(REPO_ROOT) == []
    for relative, fragments in DOCUMENT_FRAGMENTS.items():
        document = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for name, renderer in fragments.items():
            begin = BEGIN_MARKER.format(name=name)
            end = END_MARKER.format(name=name)
            actual = document.split(begin, 1)[1].split(end, 1)[0].strip()
            assert actual == renderer(REPO_ROOT).strip(), f"stale fragment: {name}"

    operation_fragment = DOCUMENT_FRAGMENTS["docs/agent-tool-schema.html"][
        "operation-registry"
    ](REPO_ROOT)
    assert "<th>Runtime ID</th>" in operation_fragment
    assert "<code>auditops.agent_runtime.v2.2</code>" in operation_fragment


def test_v23_docs_are_self_contained_status_labelled_and_link_clean() -> None:
    parsed_docs = {path.resolve(): _parse(path) for path in V23_DOCS}
    for source, parsed in parsed_docs.items():
        assert not parsed.errors, f"{source.name}: {parsed.errors}"
        assert parsed.scripts == 0, f"{source.name} must not contain scripts"
        assert not parsed.resource_urls, f"{source.name} has external/resource URLs"
        assert parsed.section_statuses
        assert all(
            status in {"IMPLEMENTED", "DEFERRED"} for status in parsed.section_statuses
        )
        for href in parsed.hrefs:
            split = urlsplit(href)
            assert not split.scheme and not split.netloc
            target = (
                source
                if not split.path
                else (source.parent / unquote(split.path)).resolve()
            )
            assert target.exists(), f"{source.name} has broken link {href!r}"
            if split.fragment:
                target_parser = parsed_docs.get(target) or _parse(target)
                assert unquote(split.fragment) in target_parser.ids
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8-sig")
    assert "](docs/agent-runtime-protocol-v2.3.html)" in readme
    assert "](docs/agent-tool-schema-v2.3.html)" in readme
