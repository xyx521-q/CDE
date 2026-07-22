"""Read run metadata from a Weights & Biases project without modifying it."""

import argparse
import json
from itertools import islice

import wandb

DEFAULT_PROJECT = "blessingbrandro/gde-vmas"


def json_default(value):
    return str(value)


def summary_record(summary):
    return summary._json_dict


def run_record(run):
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
        "config": dict(run.config),
        "summary": summary_record(run.summary),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B entity/project path (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument("--run-id", help="Show one run instead of listing runs.")
    parser.add_argument(
        "--limit", type=int, default=20, help="Maximum runs to list (default: 20)."
    )
    parser.add_argument(
        "--order",
        default="-created_at",
        help="W&B run ordering expression (default: -created_at).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    api = wandb.Api()
    if args.run_id:
        run = api.run(f"{args.project}/{args.run_id}")
        payload = run_record(run)
    else:
        runs = api.runs(args.project, order=args.order, per_page=args.limit)
        payload = [
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,
                "summary": summary_record(run.summary),
            }
            for run in islice(runs, args.limit)
        ]

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
