#!/usr/bin/env python
"""tools/import_run_logs.py — turn recorded agent runs (experiments/rulebook/results/logs/<run>/<game>.actions.jsonl)
into raw.jsonl play logs by re-executing the recorded actions on the offline engine (deterministic), so the replay
harness has logs before human logs exist.

    .venv/bin/python pbg/tools/import_run_logs.py --run 20260928-030547-16108 [--games ls20,tn36] [--out pbg/data/human_logs/agent]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pbg.core.types import Action  # noqa: E402
from pbg.env.wrapper import ArcadeEnv  # noqa: E402
from pbg.harness.online_runner import make_arcade  # noqa: E402

MODEL_TO_ID = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "ACTION7": 7}


def to_action(rec_action) -> Action:
    if isinstance(rec_action, dict):
        return Action.click(int(rec_action["row"]), int(rec_action["col"]))
    return Action.button(MODEL_TO_ID[rec_action])


def import_one(arc, game: str, actions_path: Path, out_dir: Path, *, max_steps: int = 100000) -> dict:
    recs = [json.loads(l) for l in actions_path.read_text().splitlines() if l.strip()]
    out = out_dir / game; out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.jsonl").exists():
        (out / "raw.jsonl").unlink()
    env = ArcadeEnv(arc.make(game), game, log_dir=out)
    env.reset(); n = 0; mismatches = 0
    for r in recs[:max_steps]:
        if r["kind"] == "reset":
            if n > 0:
                env.reset()
            continue
        rt = env.step(to_action(r["action"]))
        n += 1
        if rt.error:
            break
        if r.get("hash") and rt.after is not None:
            import hashlib
            from pbg.core.types import grid_to_rows
            h = hashlib.sha1("\n".join(grid_to_rows(rt.after.grid)).encode()).hexdigest()[:12]
            if h != r["hash"]:
                mismatches += 1
        if rt.status.state in ("WIN", "GAME_OVER"):
            break
    return {"game": game, "steps": n, "hash_mismatches": mismatches, "levels": env.status().level - 1, "out": str(out / "raw.jsonl")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id under experiments/rulebook/results/logs/")
    ap.add_argument("--games"); ap.add_argument("--out", default=str(ROOT / "pbg" / "data" / "human_logs" / "agent"))
    a = ap.parse_args(argv)
    log_dir = ROOT / "experiments" / "rulebook" / "results" / "logs" / a.run
    arc = make_arcade()
    games = [g.strip() for g in a.games.split(",")] if a.games else sorted(p.name.split(".")[0] for p in log_dir.glob("*.actions.jsonl"))
    for g in games:
        p = log_dir / f"{g}.actions.jsonl"
        if p.exists():
            print(import_one(arc, g, p, Path(a.out)))


if __name__ == "__main__":
    main()
