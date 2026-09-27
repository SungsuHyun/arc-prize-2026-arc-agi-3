#!/usr/bin/env python
"""Table of rulebook runs: python scripts/rulebook_summary.py [--tag substring] [--last N]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--last", type=int, default=12)
a = ap.parse_args()
runs = sorted((ROOT / "experiments" / "rulebook" / "results").glob("run-*.json"))
rows = []
for p in runs:
    d = json.loads(p.read_text())
    if a.tag and a.tag not in d.get("tag", ""):
        continue
    games = d.get("games", [])
    cell = " ".join(f"{g['game_id'][:4]}:L{g.get('levels_completed', '?')}/{g.get('score', 0) or 0:.2f}" for g in games)
    rows.append((d["run_id"][:15], d.get("tag", "")[:22], "model" if not d["config"]["params"].get("no_model") else "no-model",
                 f"{d['aggregate']['score']:.2f}", f"{d['aggregate']['games_level2plus']}/{len(games)}", cell))
for r in rows[-a.last:]:
    print(f"{r[0]:15} {r[1]:22} {r[2]:8} agg={r[3]:>6} L2+={r[4]:5} {r[5]}")
