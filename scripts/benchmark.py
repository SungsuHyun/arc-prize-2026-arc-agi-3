"""Run ALL experiment versions under identical conditions and record results.

Each experiment runs in its own subprocess (isolated imports), sharing one
benchmark tag (bench-<timestamp>) so runs are comparable as a group. After
all runs finish, experiments/summary.json and experiments/dashboard.html
are regenerated.

Usage:
    .venv/bin/python scripts/benchmark.py                     # all experiments, all games
    .venv/bin/python scripts/benchmark.py --game ls20,vc33 --max-steps 400
    .venv/bin/python scripts/benchmark.py --only v001,v003    # subset of versions
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
PY = ROOT / ".venv" / "bin" / "python"


def experiment_dirs(only: str | None) -> list[Path]:
    dirs = sorted(d for d in EXPERIMENTS.iterdir()
                  if d.is_dir() and not d.name.startswith("_")
                  and (d / "agent.py").exists())
    if only:
        prefixes = [p.strip() for p in only.split(",")]
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in prefixes)]
    return dirs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="comma-separated short game ids (default: all games)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--only", default=None,
                   help="comma-separated experiment name prefixes")
    args = p.parse_args()

    dirs = experiment_dirs(args.only)
    if not dirs:
        raise SystemExit("No experiments found.")

    bench_id = "bench-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print(f"=== {bench_id}: {len(dirs)} experiment(s), "
          f"games={args.game or 'ALL'}, max_steps={args.max_steps} ===\n")

    failed = []
    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] {d.name} ...", flush=True)
        cmd = [str(PY), str(ROOT / "scripts" / "run_experiment.py"), d.name,
               "--max-steps", str(args.max_steps), "--tag", bench_id]
        if args.game:
            cmd += ["--game", args.game]
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-4:]
        for line in tail:
            print(f"    {line}")
        if r.returncode != 0:
            failed.append(d.name)
            print(f"    FAILED (exit {r.returncode}):")
            for line in r.stderr.splitlines()[-5:]:
                print(f"    {line}")
        print()

    subprocess.run([str(PY), str(ROOT / "scripts" / "exp_summary.py")], check=True)
    subprocess.run([str(PY), str(ROOT / "scripts" / "build_dashboard.py")], check=True)

    print(f"\nBenchmark tag: {bench_id}")
    if failed:
        print(f"FAILED experiments: {failed}")
        sys.exit(1)
    print(f"Dashboard: {EXPERIMENTS / 'dashboard.html'}")


if __name__ == "__main__":
    main()
