"""harness/attribution.py — bottleneck attribution from events.jsonl (spec §16): tag each failed episode with the
module responsible.

  last state PROBE repeating                    -> Perception / Probe
  budget exhausted in HYPOTHESIZE with score<1  -> World Model Lab
  model verified but no plan repeatedly         -> Goal Inference
  budget exhausted during execution, no mismatch -> Planner (heuristic) or Goal Inference (progress)"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def attribute(events: list[dict], result: dict | None = None) -> str:
    if not events:
        return "unknown"
    if result and result.get("state") == "WIN":
        return "none"
    states = [e["to"] for e in events]
    reasons = [e.get("reason", "") for e in events]
    last = events[-1]
    tail = states[-12:]
    if tail.count("PROBE") >= 2:
        return "perception/probe"
    if last["from"] == "HYPOTHESIZE" or any("no hypothesis" in r for r in reasons[-5:]):
        top_scores = [r for r in reasons if r.startswith("top h=")]
        if top_scores and "score=1.00" not in top_scores[-1]:
            return "world_model_lab"
        if not top_scores:
            return "world_model_lab"
    if sum(1 for r in reasons if "verified model -> goal demoted" in r) >= 2:
        return "goal_inference"
    if any(r.startswith("plan of") for r in reasons) and not any("mismatch" in r for r in reasons[-10:]):
        return "planner/goal_progress"
    if any("mismatch" in r for r in reasons[-10:]):
        return "world_model_lab"
    return "unknown"


def attribute_run(run_json: Path) -> dict:
    run = json.loads(Path(run_json).read_text())
    log_dir = Path(run_json).parent / "logs" / run["run_id"]
    tags = {}
    for g in run["games"]:
        p = log_dir / f"{g['game_id']}.events.jsonl"
        events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        tags[g["game_id"]] = attribute(events, g)
    dist = Counter(v for v in tags.values())
    return {"run_id": run["run_id"], "tags": tags, "distribution": dict(dist)}


if __name__ == "__main__":
    import sys
    print(json.dumps(attribute_run(Path(sys.argv[1])), indent=1))
