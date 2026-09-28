"""harness/replay_runner.py — module-level evaluation over recorded logs without the live environment (spec §16).

Given a raw.jsonl (human or agent play), it: replays every frame through Perception (timing, tracking stability),
classifies action semantics, induces prior-driven hypotheses (+ optional LLM refinement), evaluates them on the whole
log, runs goal inference with the level-up filter, and (given a verified model) plans from the level-start scene.

    .venv/bin/python -m pbg.harness.replay_runner path/to/raw.jsonl [--llm] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from ..core.contracts import RuleModel
from ..core.types import Action, Transition
from ..env.replay import ReplayEnv
from ..goal.infer import GoalInference
from ..perception import Perception
from ..planner.execute import Planner
from ..probe.semantics import classify_actions
from ..wml.refine import WorldModelLab


def replay_transitions(records, game_id: str = "") -> tuple[list[Transition], Perception, dict]:
    env = ReplayEnv(records, game_id, lenient=True)
    p = Perception()
    frame = env.reset(); scene = p.parse(frame, None)
    ts: list[Transition] = []; step = 0
    id_changes = 0; tracked = 0
    while True:
        a = env.next_action()
        if a is None:
            break
        rt = env.step(a)
        if rt.error:
            break
        step += 1
        if a.type == "RESET":
            scene = p.parse(rt.after, None); continue
        after = p.parse(rt.after, scene)
        t = p.transition(scene, a, after, before_frame=rt.before, after_frame=rt.after, status_change=rt.status_change, tid=f"{env.game_id}:{rt.level}:{step}",
                         level=rt.level, step_idx=step)
        if p.merge_confirmed:
            p.refit(ts); after = t.after
        ts.append(t)
        # tracking stability: objects that stayed (same colour+shape at the same place) must keep their id
        before_ids = {o.identity(): o.id for o in scene.objects}
        for o in after.objects:
            if o.identity() in before_ids:
                tracked += 1
                if before_ids[o.identity()] != o.id:
                    id_changes += 1
        scene = after
        if rt.status_change == "LEVEL_UP":
            scene = p.parse(rt.after, None)
    stats = {"transitions": len(ts), "parse_ms_mean": round(1000 * sum(p.timings) / max(1, len(p.timings)), 2), "parse_ms_max": round(1000 * max(p.timings or [0]), 2),
             "tracking_keep_rate": round(1 - id_changes / tracked, 4) if tracked else None, "levels": sorted({t.level for t in ts}),
             "level_ups": sum(1 for t in ts if t.status_change in ("LEVEL_UP", "WIN"))}
    return ts, p, stats


def run_pipeline(raw_path: Path, *, llm=None, plan: bool = True, log=print) -> dict:
    records = [json.loads(l) for l in Path(raw_path).read_text().splitlines() if l.strip()]
    game_id = records[0].get("game_id", Path(raw_path).parent.name) if records else "?"
    ts, perception, stats = replay_transitions(records, game_id)
    out = {"game_id": game_id, "perception": stats}
    if not ts:
        return out
    sem = classify_actions(ts)
    out["semantics"] = {k: v for k, v in sem.items()}
    avail = sorted({t.action.key for t in ts})
    available = [Action.button(int(k[6:])) for k in avail if k.startswith("ACTION")] + ([Action("CLICK")] if "CLICK" in avail else [])
    lab = WorldModelLab(llm=llm, log=log)
    t0 = time.time()
    H = lab.refine(ts, [], None, scene=ts[0].before, semantics=sem, available=available, use_llm=llm is not None)
    out["hypotheses"] = [{"name": h.name, "score": round(h.score, 3), "coverage": round(h.coverage, 3), "violations": len(h.violations), "verified": h.verified,
                          "rules": {r.name: round(r.confidence, 2) for r in h.model.rules()} if isinstance(h.model, RuleModel) else {}} for h in H]
    out["wml_seconds"] = round(time.time() - t0, 2); out["llm_calls"] = lab.llm_calls
    gi = GoalInference(log=log)
    roles_fn = (lambda sc, m=H[0].model: m.with_roles(sc)) if H and isinstance(H[0].model, RuleModel) else None
    by_level: dict[int, list[Transition]] = {}
    for t in ts:
        by_level.setdefault(t.level, []).append(t)
    goals_out = {}
    for lv, lts in by_level.items():
        G = gi.refine(ts, [], None, lv, lts[0].before, roles_fn=roles_fn)
        goals_out[lv] = [g.to_json() for g in G[:5]]
    out["goals"] = goals_out
    if plan and H and H[0].score >= 1.0:
        planner = Planner(log=log)
        plans = {}
        for lv, lts in by_level.items():
            G = gi.refine(ts, [], None, lv, lts[0].before, roles_fn=roles_fn)
            p = planner.search(lts[0].before, H[:1], G, available=available, semantics=sem)
            human = sum(1 for t in lts)
            plans[lv] = {"plan_len": len(p) if p else None, "human_actions": human, "reason": planner.last_info.reason if planner.last_info else ""}
        out["plans"] = plans
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+"); ap.add_argument("--llm", action="store_true"); ap.add_argument("--json"); ap.add_argument("--no-plan", action="store_true")
    a = ap.parse_args(argv)
    llm = None
    if a.llm:
        from ..llm.gateway import LLMGateway
        llm = LLMGateway()
    results = [run_pipeline(Path(p), llm=llm, plan=not a.no_plan) for p in a.raw]
    for r in results:
        print(json.dumps(r, indent=1, default=str)[:4000])
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
