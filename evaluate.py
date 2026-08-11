"""
Evaluation harness — command-line front end for the dashboard's evaluator.

Thin on purpose: every score comes from ``backend.evaluation``, the same module
the ``/api/eval`` endpoints call, so a number printed here and a number on the
Evaluation tab can never disagree.

This used to score answers by word overlap against a reference, which its own
docstring called "a crude proxy for what an LLM judge would check". It is now a
real LLM judge — RAGAS — run with whichever models the workspace is configured
to use.

Usage::

    python evaluate.py                          # both bundled golden sets
    python evaluate.py --testset my-questions   # a saved workspace test set
    python evaluate.py --file eval/golden_v2_fixed.json
    python evaluate.py --user <id> --generate 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from backend.evaluation import (  # noqa: E402
    METRIC_LABELS,
    TestSet,
    generate_testset,
    list_testsets,
    load_testset,
    parse_testset,
    run_evaluation,
    save_testset,
)
from backend.main import setup_pipeline  # noqa: E402
from backend.user_config import load_user_config  # noqa: E402
from backend.workspace import Workspace  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GOLDEN_V1 = os.path.join(PROJECT_ROOT, "eval", "golden_v1_flawed.json")
GOLDEN_V2 = os.path.join(PROJECT_ROOT, "eval", "golden_v2_fixed.json")


def _load_file(path: str) -> TestSet:
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as handle:
        return parse_testset(name, handle.read(), source="sample")


def _print_report(report: dict) -> None:
    print()
    for row in report["rows"]:
        scores = row.get("scores") or {}
        faith = scores.get("faithfulness")
        flag = "  ?  " if faith is None else (" OK  " if faith >= 0.6 else " LOW ")
        detail = " ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        print(f"  [{flag}] {row['route'] or 'failed':<13} {row['question'][:60]}")
        if detail:
            print(f"          {detail}")
        if row.get("error"):
            print(f"          ERROR: {row['error']}")

    print("\n  " + "-" * 50)
    for metric in report["metrics"]:
        value = report["scores"].get(metric)
        print(f"  {METRIC_LABELS[metric]:<24} {'—' if value is None else f'{value:.3f}'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--user", default=None, help="Workspace to evaluate (default: local).")
    parser.add_argument("--testset", help="Name of a saved test set in the workspace.")
    parser.add_argument("--file", help="Path to a JSON/CSV test set.")
    parser.add_argument(
        "--generate",
        type=int,
        metavar="N",
        help="Generate an N-question test set from the workspace's documents and save it.",
    )
    parser.add_argument("--list", action="store_true", help="List saved test sets and exit.")
    args = parser.parse_args()

    workspace = Workspace.for_user(args.user).ensure()

    if args.list:
        sets = list_testsets(workspace)
        if not sets:
            print("No saved test sets.")
        for entry in sets:
            print(
                f"  {entry['name']:<28} {entry['size']:>3} questions  "
                f"{'refs' if entry['has_references'] else 'no refs'}  ({entry['source']})"
            )
        return 0

    print(f"Loading pipeline for workspace '{workspace.user_id}'...")
    try:
        config = load_user_config(workspace)
        pipeline = setup_pipeline(workspace, config)
    except Exception as exc:
        print(f"Failed to load pipeline: {exc}", file=sys.stderr)
        print(
            "Upload documents and build the index first: python -m backend.ingest",
            file=sys.stderr,
        )
        return 1

    if args.generate:
        name = f"generated-{uuid.uuid4().hex[:8]}"
        items = generate_testset(pipeline, config, size=args.generate, log=print)
        save_testset(workspace, TestSet(name=name, items=items, source="generated"))
        print(f"\nSaved test set '{name}' with {len(items)} question(s).")
        print(json.dumps(items, indent=2))
        return 0

    if args.testset:
        testsets = [load_testset(workspace, args.testset)]
    elif args.file:
        testsets = [_load_file(args.file)]
    else:
        # The default run is still the v1-vs-v2 comparison the golden sets were
        # built for: the point is that fixing the *reference answers* moves the
        # score, without touching retrieval.
        testsets = [_load_file(GOLDEN_V1), _load_file(GOLDEN_V2)]

    reports = []
    for testset in testsets:
        print("\n" + "=" * 70)
        print(f"EVAL: {testset.name}  ({len(testset.items)} questions)")
        print("=" * 70)
        report = run_evaluation(
            pipeline, config, testset, run_id=str(uuid.uuid4()), log=print
        )
        _print_report(report)
        reports.append((testset.name, report))

    if len(reports) == 2:
        print("\n" + "=" * 70)
        for metric in reports[0][1]["metrics"]:
            before = reports[0][1]["scores"].get(metric)
            after = reports[1][1]["scores"].get(metric)
            if before is None or after is None:
                continue
            print(f"  {METRIC_LABELS[metric]:<24} {before:.3f} -> {after:.3f}")
        print("  (difference comes from the reference answers, not the retriever)")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
