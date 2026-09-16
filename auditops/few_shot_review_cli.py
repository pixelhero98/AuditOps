"""Dependency-light entry point for the v2.2 few-shot review packet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .agent_batch import write_few_shot_review_packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render and tokenize the AuditOps v2.2 few-shot review packet"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument(
        "--model-config", action="append", required=True, dest="model_configs"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = write_few_shot_review_packet(
        args.benchmark_dir, args.model_configs, args.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
