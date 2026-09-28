"""harness/postmortem.py — per-level post-mortem of one game in a pbg run: where the actions and the time went, how the
hypotheses/goals evolved, what every plan did, and a verdict on why the level was not cleared (spec §16 attribution,
finer grained than attribution.py).

    .venv/bin/python -m pbg.harness.postmortem <run_id> <game> [--json]"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "pbg" / "results"


def load(run_id: str, game: str):
    run = json.loads((RESULTS / f"run-{run_id}.json").read_text())
    g = next(x for x in run["games"] if x["game_id"] == game)
    ev = [json.loads(l) for l in (RESULTS / "logs" / run_id / f"{game}.events.jsonl").read_text().splitlines() if l.strip()]
    log_lines = (RESULTS / "logs" / run_id / f"{game}.log").read_text().splitlines()
    return run, g, ev, log_lines


def per_level(ev: list[dict], g: dict) -> list[dict]:
    """Split the event stream at level-up events; per segment: dwell by state, actions by transition, plans, mismatches."""
    segments = []; cur = {"level": 1, "events": []}
    for e in ev:
        m = re.match(r"level (\d+) -> (\d+)", e["reason"])
        cur["events"].append(e)
        if m:
            segments.append(cur); cur = {"level": int(m.group(2)), "events": []}
    segments.append(cur)
    out = []
    for seg in segments:
        es = seg["events"]
        if not es:
            continue
        t0, t1 = es[0]["ts"], es[-1]["ts"]
        used0, used1 = es[0]["budget_used"], es[-1]["budget_used"]
        actions_by = Counter(); prev = used0
        for e in es:
            actions_by[(e["from"], e["to"])] += e["budget_used"] - prev; prev = e["budget_used"]
        plans = [e for e in es if e["reason"].startswith("plan of") or e["reason"].startswith("explore plan")]
        outcomes = Counter(" ".join(e["reason"].split(" ")[:2]) for e in es if e["from"] == "EXECUTE")
        tops = [e["reason"] for e in es if e["reason"].startswith("top h=")]
        goals = Counter(re.search(r"top=([^;|]+)", t).group(1) if re.search(r"top=([^;|]+)", t) else "?" for t in tops)
        scores = [float(m.group(1)) for t in tops for m in [re.search(r"score=([\d.]+)", t)] if m]
        out.append({"level": seg["level"], "seconds": round(t1 - t0), "actions": used1 - used0, "actions_by": {f"{a}->{b}": n for (a, b), n in actions_by.most_common() if n > 0},
                    "hyp_rounds": len(tops), "score_first": scores[0] if scores else None, "score_last": scores[-1] if scores else None, "score_max": max(scores) if scores else None,
                    "top_goals": goals.most_common(4), "plans": len(plans), "plan_lengths": [int(re.search(r"of (\d+)", p["reason"]).group(1)) for p in plans if re.search(r"of (\d+)", p["reason"])][:30],
                    "exec_outcomes": dict(outcomes), "mismatches": sum(1 for e in es if "mismatch" in e["reason"]), "game_overs": sum(1 for e in es if "game over" in e["reason"]),
                    "llm_jobs": sum(1 for e in es if "llm job started" in e["reason"]), "stuck": sum(1 for e in es if e["reason"].startswith("stuck")),
                    "cleared": any(re.match(r"level (\d+) -> (\d+)", e["reason"]) for e in es)})
    return out


def verdict(lv: dict) -> str:
    if lv["cleared"]:
        return "cleared"
    if lv["hyp_rounds"] == 0:
        return "no hypothesize round (probe/LLM never returned)"
    if (lv["score_max"] or 0) < 0.8:
        return f"world model never explained the log (best score {lv['score_max']})"
    if lv["plans"] == 0:
        return "model ok but no plan for any goal (goal templates do not express the win condition or search failed)"
    if lv["mismatches"] >= max(3, lv["plans"] // 2):
        return f"plans executed but predictions failed ({lv['mismatches']} mismatches: model wrong outside the observed states)"
    if lv["exec_outcomes"].get("plan exhausted", 0) >= 3 and not lv["cleared"]:
        return "plans completed without a level-up: the goal reached is not the win condition"
    if lv["game_overs"] >= 2:
        return f"repeated game overs ({lv['game_overs']}): action limit / hazard not modelled"
    return "budget or time ran out while still exploring"


def report(run_id: str, game: str) -> dict:
    run, g, ev, log_lines = load(run_id, game)
    levels = per_level(ev, g)
    for lv in levels:
        lv["verdict"] = verdict(lv)
    llm = [l for l in log_lines if "llm candidate" in l or "llm job" in l]
    return {"run_id": run_id, "game": game, "levels_completed": g.get("levels_completed"), "levels_total": g.get("levels_total"), "actions": g.get("actions"),
            "score": g.get("score"), "baseline": g.get("level_baseline_actions"), "level_actions": g.get("level_actions"), "stop": g.get("stop_reason"),
            "llm_calls": g.get("llm_calls"), "llm_lines": llm[-12:], "final_hypotheses": g.get("hypotheses"), "final_goals": g.get("goals"), "levels": levels}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("run_id"); ap.add_argument("game"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = report(a.run_id, a.game)
    if a.json:
        print(json.dumps(r, indent=1, default=str)); return
    print(f"== {r['game']} run {r['run_id']}: levels {r['levels_completed']}/{r['levels_total']} actions {r['actions']} score {r['score']} stop {r['stop']} llm {r['llm_calls']}")
    print(f"   baseline per level {r['baseline']} | ours {r['level_actions']}")
    for lv in r["levels"]:
        print(f"-- level {lv['level']}: {lv['seconds']}s {lv['actions']} actions | hyp rounds {lv['hyp_rounds']} score {lv['score_first']}->{lv['score_last']} (max {lv['score_max']}) | plans {lv['plans']} lens {lv['plan_lengths'][:12]}")
        print(f"   actions by {lv['actions_by']}")
        print(f"   goals {lv['top_goals']} | exec {lv['exec_outcomes']} | mismatches {lv['mismatches']} game_overs {lv['game_overs']} stuck {lv['stuck']} llm_jobs {lv['llm_jobs']}")
        print(f"   VERDICT: {lv['verdict']}")
    print("   final hypotheses:", [(h['name'], h['score']) for h in (r['final_hypotheses'] or [])][:4])
    print("   final goals:", [(x['name'], x['confidence']) for x in (r['final_goals'] or [])][:5])


if __name__ == "__main__":
    main()
