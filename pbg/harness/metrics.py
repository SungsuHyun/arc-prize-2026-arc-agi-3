"""harness/metrics.py — integrated metrics (spec §16): level clear rate, action efficiency vs human, level transition
cost, LLM call ratio, bottleneck distribution; holdout games reported separately."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .attribution import attribute_run

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "pbg" / "data" / "holdout.txt"


def holdout_games() -> set[str]:
    if HOLDOUT.exists():
        return {l.strip() for l in HOLDOUT.read_text().splitlines() if l.strip() and not l.startswith("#")}
    return set()


def summarize_run(run_json: Path, human_actions: Optional[dict] = None) -> dict:
    run = json.loads(Path(run_json).read_text())
    hold = holdout_games()
    attr = attribute_run(run_json)
    out = {"run_id": run["run_id"], "tag": run.get("tag"), "groups": {}}
    for group, games in (("dev", [g for g in run["games"] if g["game_id"] not in hold]), ("holdout", [g for g in run["games"] if g["game_id"] in hold])):
        if not games:
            continue
        cleared = sum(g.get("levels_completed") or 0 for g in games); total = sum(g.get("levels_total") or 0 for g in games)
        actions = sum(g.get("actions") or 0 for g in games); llm = sum(g.get("llm_calls") or 0 for g in games)
        eff = []
        for g in games:
            la = g.get("level_actions") or {}; base = g.get("level_baseline_actions") or []
            for i in range(g.get("levels_completed") or 0):
                if i < len(base) and base[i] and str(i + 1) in la:
                    eff.append(la[str(i + 1)] / base[i])
        out["groups"][group] = {"games": len(games), "level_clear_rate": round(cleared / total, 3) if total else None, "levels_cleared": cleared, "levels_total": total,
                                "action_efficiency_vs_baseline": round(sum(eff) / len(eff), 2) if eff else None, "llm_call_ratio": round(llm / actions, 4) if actions else None,
                                "mean_score": round(sum(g.get("score") or 0 for g in games) / len(games), 4),
                                "bottlenecks": {k: v for k, v in attr["distribution"].items()} if group == "dev" else {g["game_id"]: attr["tags"].get(g["game_id"]) for g in games}}
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize_run(Path(sys.argv[1])), indent=1))
