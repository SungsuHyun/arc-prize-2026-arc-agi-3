"""Run one or more experiment versions under identical conditions and record results.

Each experiment runs in its own subprocess (isolated imports), sharing one
benchmark tag (bench-<timestamp>) so runs are comparable as a group. After
all runs finish, experiments/summary.json and the site are regenerated.

Versions must be named explicitly (LLM runs are expensive); `--all` opts in
to re-running every version.

Usage:
    .venv/bin/python scripts/benchmark.py v006                        # one version
    .venv/bin/python scripts/benchmark.py v005,v006 --game ls20,vc33 --max-steps 400
    .venv/bin/python scripts/benchmark.py --all                       # every version
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
    p.add_argument("versions", nargs="?", default=None,
                   help="comma-separated experiment name prefixes, e.g. v006 or v005,v006")
    p.add_argument("--game", default=None,
                   help="comma-separated short game ids (default: all games)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--only", default=None, help="alias of the positional versions argument")
    p.add_argument("--all", action="store_true",
                   help="run EVERY experiment version (expensive; off by default)")
    p.add_argument("--no-publish", action="store_true",
                   help="skip publishing the site to GitHub Pages")
    args = p.parse_args()

    only = args.versions or args.only
    if not only and not args.all:
        raise SystemExit("Name the version(s) to benchmark (e.g. `make bench NAME=v006`) "
                         "or pass --all / ALL=1 to run every version.")
    dirs = experiment_dirs(None if args.all else only)
    if not dirs:
        raise SystemExit(f"No experiments match {only!r}.")

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
    if not args.no_publish:
        pub = subprocess.run(["bash", str(ROOT / "scripts" / "publish_site.sh")],
                             capture_output=True, text=True)
        if pub.returncode != 0:
            print(f"WARNING: site publish failed:\n{pub.stderr[-500:]}")

    print(f"\nBenchmark tag: {bench_id}")
    if failed:
        print(f"FAILED experiments: {failed}")
        sys.exit(1)
    base = "https://sungsuhyun.github.io/arc-prize-2026-arc-agi-3"
    print(f"Report: {base}/{bench_id}.html")
    print(f"Index:  {base}/")


if __name__ == "__main__":
    main()
