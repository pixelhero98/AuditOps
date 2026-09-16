"""Minimal container entrypoint for exact AuditOps v2.4 request preflight."""

from __future__ import annotations

import argparse
import json

from .agent_preflight_v24 import preflight_agent_requests_v24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = preflight_agent_requests_v24(
        args.benchmark_dir,
        args.model_config,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
