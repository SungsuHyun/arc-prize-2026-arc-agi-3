#!/usr/bin/env python
"""Layer-0 measurement: run the model-free Explorer on public games and report levels completed and actions.
Deterministic (no LLM) so every run is reproducible. Usage: explore_all.py [--games a,b,c] [--levels 3] [--actions 300] [--seconds 300] [--jobs 4]"""
import argparse, json, logging, os, sys, time
if os.environ.get("PYTHONHASHSEED") != "0":   # str hashes are per-process salted; pin them so runs are reproducible
    os.environ["PYTHONHASHSEED"] = "0"; os.execv(sys.executable, [sys.executable] + sys.argv)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
from arcnav.runner import make_arcade
from arcnav.agent import GameSession
from arcnav.explorer import Explorer

ap = argparse.ArgumentParser()
ap.add_argument("--games", default="")
ap.add_argument("--levels", type=int, default=3)
ap.add_argument("--actions", type=int, default=300)
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--out", default="/home/hss/code/kaggle/2026/arc-prize-2026-arc-agi-3/experiments/arcnav/explorer")
a = ap.parse_args()
arc = make_arcade()
games = a.games.split(",") if a.games else sorted(e.game_id.split("-")[0] for e in arc.get_environments())
out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path("/home/hss/.claude/jobs/21ca0929/tmp/explorer-logs"); log_dir.mkdir(parents=True, exist_ok=True)

def one(g):
    t0 = time.time()
    try:
        s = GameSession(arc.make(g), g, None, log_dir=log_dir, max_minutes=max(5, a.seconds * a.levels / 60 + 2), max_actions=a.actions * a.levels + 50, verbose=False)
        s.reset()
        ex = Explorer(s, log=lambda m: None, max_actions_per_level=a.actions, max_seconds=a.seconds)
        levels = ex.play(max_levels=a.levels)
        row = {"game": g, "levels_completed": sum(1 for r in levels if r["completed"]), "actions": s.actions_used,
               "per_level": [{k: v for k, v in r.items() if k != "path"} | ({"path": r.get("path", [])[-6:]} if r.get("path") else {}) for r in levels],
               "nodes": len(ex.nodes), "seconds": round(time.time() - t0, 1), "valid": list(s.valid_actions)}
        s.sandbox.close()
    except Exception as e:
        row = {"game": g, "error": f"{type(e).__name__}: {e}"[:200], "seconds": round(time.time() - t0, 1)}
    print(json.dumps(row)[:400], flush=True)
    return row

with ThreadPoolExecutor(a.jobs) as pool:
    rows = list(pool.map(one, games))
summary = {"games": len(rows), "levels_completed": sum(r.get("levels_completed", 0) for r in rows),
           "games_level1": sum(1 for r in rows if r.get("levels_completed", 0) >= 1),
           "games_level2plus": sum(1 for r in rows if r.get("levels_completed", 0) >= 2),
           "actions": sum(r.get("actions", 0) for r in rows), "errors": [r["game"] for r in rows if r.get("error")],
           "params": vars(a), "rows": rows}
p = out_dir / f"explore-{time.strftime('%Y%m%d-%H%M%S')}.json"
p.write_text(json.dumps(summary, indent=1))
print(f"\nlevels completed: {summary['levels_completed']} | games with level 1: {summary['games_level1']}/{len(rows)} | level>=2: {summary['games_level2plus']} | errors: {summary['errors']}\nsaved {p}")
