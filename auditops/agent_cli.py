"""Dependency-light entry point for offline GPU inference inside the vLLM image."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .agent_batch import run_agent_baseline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AuditOps offline text-agent inference"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--runtime-mode",
        default="safety_hybrid",
        choices=("direct", "capability_agent", "safety_hybrid"),
    )
    parser.add_argument(
        "--prompt-condition",
        required=True,
        choices=("zero_shot", "few_shot"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--few-shot-approval", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=("full", "narrative_gate"),
        default=None,
    )
    parser.add_argument("--full-run-authorization", default=None)
    parser.add_argument("--expected-authorizer", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_agent_baseline(
        args.benchmark_dir,
        args.model_config,
        args.output_dir,
        runtime_mode=args.runtime_mode,
        prompt_condition=args.prompt_condition,
        few_shot_approval=args.few_shot_approval,
        limit=args.limit,
        selection_mode=args.selection_mode,
        full_run_authorization=args.full_run_authorization,
        expected_authorizer=args.expected_authorizer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
