#!/usr/bin/env python
"""Per-game comparison: v012d 25-game passes (archived harness summaries) vs an arcnav 25-game run JSON.
usage: python scripts/compare_lines.py experiments/arcnav/results/run-XXXX.json"""
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
v012 = defaultdict(list)
for summ in glob.glob(str(ROOT / "vendor/duck-runs/*v012d-all25-*/summary.txt")):
    for line in open(summ):
        m = re.match(r"\s+(\w{4})-\w+: score=([\d.]+), levels=([\d.]+)/(\d+)", line)
        if m:
            v012[m.group(1)].append((float(m.group(2)), float(m.group(3))))
arc = json.load(open(sys.argv[1]))
rows = []
for g in arc["games"]:
    gid = g["game_id"].split("-")[0]
    v = v012.get(gid, [])
    v_lv = sum(1 for s, l in v if l >= 1); v_mean = sum(s for s, _ in v) / len(v) if v else 0.0
    rows.append((gid, g.get("levels_completed", 0), g.get("score", 0) or 0, v_lv, len(v), v_mean))
print(f"{'game':6} {'arcnav lv':>9} {'arcnav score':>12} | {'v012d lv1 passes':>16} {'v012d mean':>10}")
for gid, lv, sc, vl, n, vm in sorted(rows, key=lambda r: (-(r[3] / max(1, r[4])), r[0])):
    flag = "  <- v012d only" if vl and not lv else ("  <- arcnav only" if lv and not vl else "")
    print(f"{gid:6} {lv:>9} {sc:>12.2f} | {f'{vl}/{n}':>16} {vm:>10.2f}{flag}")
print(f"\narcnav mean {arc['aggregate']['score']:.2f}, level1+ games {sum(1 for r in rows if r[1] >= 1)}, level2+ {arc['aggregate'].get('games_level2plus', 0)} | "
      f"v012d mean over passes {sum(sum(s for s, _ in v) for v in v012.values()) / max(1, sum(len(v) for v in v012.values())) * 25 / 25:.2f}")
