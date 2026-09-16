"""Dependency-light entry point for synthetic v2.2 structured-output probes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .grammar_probe import run_structured_output_probes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AuditOps v2.2 synthetic structured-output probes"
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run_structured_output_probes(args.model_config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
