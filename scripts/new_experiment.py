"""Scaffold a new experiment version.

Copies the current agent/my_agent.py (or another experiment's agent.py with
--from) into a new numbered experiment folder with a fresh config + notes.

Usage:
    .venv/bin/python scripts/new_experiment.py greedy-search
    .venv/bin/python scripts/new_experiment.py tweak-lr --from v002
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def next_version() -> int:
    versions = [int(m.group(1)) for d in EXPERIMENTS.iterdir()
                if d.is_dir() and (m := re.match(r"v(\d+)-", d.name))]
    return max(versions, default=0) + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("title", help="short kebab-case title, e.g. greedy-search")
    p.add_argument("--from", dest="base", default=None,
                   help="copy agent.py from this experiment instead of agent/my_agent.py")
    args = p.parse_args()

    name = f"v{next_version():03d}-{args.title}"
    exp_dir = EXPERIMENTS / name
    exp_dir.mkdir()
    (exp_dir / "results").mkdir()

    if args.base:
        matches = [d for d in EXPERIMENTS.iterdir()
                   if d.is_dir() and d.name.startswith(args.base)]
        if len(matches) != 1:
            raise SystemExit(f"--from {args.base!r} matched {len(matches)} experiments")
        src = matches[0] / "agent.py"
        base_note = f"based on {matches[0].name}"
    else:
        src = ROOT / "agent" / "my_agent.py"
        base_note = "based on agent/my_agent.py"
    shutil.copy(src, exp_dir / "agent.py")

    template = EXPERIMENTS / "_template"
    config = json.loads((template / "config.json").read_text())
    config["description"] = args.title.replace("-", " ")
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    notes = (template / "notes.md").read_text().replace("{NAME}", name)
    (exp_dir / "notes.md").write_text(notes)

    print(f"Created experiments/{name}/  ({base_note})")
    print(f"  1. edit experiments/{name}/agent.py")
    print(f"  2. edit experiments/{name}/config.json (games, max_steps, params)")
    print(f"  3. run:  make exp-run NAME={name}")


if __name__ == "__main__":
    main()
