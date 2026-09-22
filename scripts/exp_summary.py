"""Aggregate all experiment results into one table + experiments/summary.json.

The summary.json is the single input a future dashboard reads:
every run of every experiment, flattened, with per-game details preserved.

Usage:
    .venv/bin/python scripts/exp_summary.py            # table + write summary.json
    .venv/bin/python scripts/exp_summary.py --json     # print summary.json to stdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def collect() -> dict:
    experiments = []
    for exp_dir in sorted(EXPERIMENTS.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        runs = []
        for f in sorted((exp_dir / "results").glob("run-*.json")):
            try:
                runs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                print(f"warning: skipping unreadable {f}")
        config = {}
        cfg_file = exp_dir / "config.json"
        if cfg_file.exists():
            config = json.loads(cfg_file.read_text())
        experiments.append({
            "name": exp_dir.name,
            "description": config.get("description", ""),
            "config": config,
            "n_runs": len(runs),
            "best_score": max((r["aggregate"]["score"] or 0 for r in runs), default=None),
            "last_run": runs[-1]["run_id"] if runs else None,
            "last_score": runs[-1]["aggregate"]["score"] if runs else None,
            "runs": runs,
        })
    return {"schema": 1, "experiments": experiments}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="print JSON to stdout")
    args = p.parse_args()

    summary = collect()
    out = EXPERIMENTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print(f"{'experiment':<28} {'runs':>4} {'best':>7} {'last':>7}  last run")
    print("-" * 70)
    for e in summary["experiments"]:
        best = f"{e['best_score']:.3f}" if e["best_score"] is not None else "-"
        last = f"{e['last_score']:.3f}" if e["last_score"] is not None else "-"
        print(f"{e['name']:<28} {e['n_runs']:>4} {best:>7} {last:>7}  {e['last_run'] or '-'}")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
