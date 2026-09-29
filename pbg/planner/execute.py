"""planner/execute.py — Planner facade (spec §10): algorithm selection, multi-hypothesis planning (common prefix of
the top plans), execution monitoring with immediate stop on mismatch."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..core.contracts import Hypothesis, scene_equal, scene_mismatch
from ..core.types import Action, Scene, Transition
from .astar import astar
from .beam import beam_search
from .common import action_set, make_successor, state_key
from .mcts import mcts
from .stuck import StuckDetector


@dataclass
class ExecResult:
    kind: str                     # PROGRESS | FAILED | MISMATCH | STUCK | EXHAUSTED | ERROR
    scene: Scene
    t: Optional[Transition] = None
    executed: int = 0
    detail: dict = field(default_factory=dict)


@dataclass
class PlanInfo:
    plan: list[Action]
    hypothesis: Optional[Hypothesis] = None
    goal=None
    scored: list = field(default_factory=list)
    algorithm: str = ""
    reason: str = ""


class Planner:
    def __init__(self, *, depth: int = 60, weight: float = 8.0, beam_width: int = 32, time_limit: float = 3.0, stochastic: bool = False,
                 log=None):
        self.depth, self.weight, self.beam_width, self.time_limit = depth, weight, beam_width, time_limit
        self.stochastic = stochastic
        self.log = log or (lambda *a, **k: None)
        self.stuck = StuckDetector()
        self.responsive: list[tuple[int, int]] = []
        self.last_info: Optional[PlanInfo] = None
        self.click_map = None             # click response map (orchestrator knowledge): restricts click candidates
        self.observed: dict = {}          # (state_key, action label) -> observed after scene (ground truth for planning)
        self.observed_seen: list = []

    def _actions_fn(self, available: list[Action], semantics):
        def fn(scene: Scene) -> list[Action]:
            return action_set(scene, available, semantics=semantics, responsive=self.responsive, click_map=self.click_map)
        return fn

    def _abstract(self, scene: Scene, model, goal) -> Optional[list[Action]]:
        """Goal-provided abstract plan (over the goal-relevant objects only), verified by full simulation with the model."""
        ap = getattr(goal, "abstract_plan", None)
        if ap is None:
            return None
        try:
            rs = model.with_roles(scene) if hasattr(model, "with_roles") else scene
            trig_keys = ap(rs, model)
        except Exception:
            return None
        if not trig_keys:
            return None
        plan = []; s_ = scene
        for tk in trig_keys:
            r0, c0, r1, c1 = tk
            a = Action.click((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)
            nxt = model.predict(s_, a)
            if nxt is None:
                return None
            plan.append(a); s_ = nxt
        return plan if goal.is_goal(model.with_roles(s_) if hasattr(model, "with_roles") else s_) else None

    def _search_one(self, scene: Scene, model, goal, available, semantics, depth: int, visited: set) -> tuple[Optional[list[Action]], str]:
        actions_fn = self._actions_fn(available, semantics)
        succ = make_successor(model, actions_fn, self.observed)
        setattr(goal, "model_hint", model)
        ab = self._abstract(scene, model, goal)
        if ab:
            return ab, "abstract"

        # a goal's progress heuristic, damped when it proved non-monotone (spec §9)
        w = self.weight * getattr(goal, "progress_weight", 1.0)
        if self.stochastic:
            p = mcts(scene, lambda s, a: model.predict(s, a), actions_fn, goal.is_goal, goal.progress)
            return p, "mcts"
        est = getattr(goal, "estimate", None)
        p = astar(scene, succ, goal.is_goal, goal.progress, depth=depth, weight=w, visited_penalty=visited, time_limit=self.time_limit,
                  heuristic=(lambda s_, e=est: e(s_)) if est else None)
        if p is not None:
            return p, "astar"
        p = beam_search(scene, succ, goal.is_goal, goal.progress, depth=depth, width=self.beam_width, time_limit=self.time_limit)
        return p, "beam" if p is not None else "none"

    def search(self, scene: Scene, H: list[Hypothesis], G: list, *, available: list[Action], semantics=None, depth: Optional[int] = None,
               visited: Optional[set] = None) -> Optional[list[Action]]:
        """Multi-hypothesis planning (spec §10): top-3 hypotheses x top-2 goals, execute only the common prefix."""
        depth = depth or self.depth
        plans = []
        info = PlanInfo([], algorithm="", reason="")
        t0 = time.perf_counter()
        ranked = sorted(H, key=lambda h: -h.plan_weight())
        for h in ranked[:3]:
            for g in G[:3]:
                p, alg = self._search_one(scene, h.model, g, available, semantics, depth, visited or set())
                if p:
                    plans.append((h.plan_weight() * max(g.confidence, 0.05), p, h, g, alg))
        if not plans:
            # a verified model can still be useless for planning (UNKNOWN outside the observed states): try the rest
            for h in ranked[3:]:
                for g in G[:2]:
                    p, alg = self._search_one(scene, h.model, g, available, semantics, depth, visited or set())
                    if p:
                        plans.append((h.plan_weight() * max(g.confidence, 0.05) * 0.5, p, h, g, alg))
        info.scored = [(round(s, 3), len(p), h.name, g.name, alg) for s, p, h, g, alg in plans]
        if not plans:
            info.reason = "no plan under any (hypothesis, goal) pair"
            self.last_info = info
            self.log(f"planner: no plan ({time.perf_counter() - t0:.2f}s) H={len(H)} G={len(G)}")
            return None
        plans.sort(key=lambda x: (-x[0], len(x[1])))
        best = plans[0]
        # the common prefix is taken across HYPOTHESES for the best goal (where the plans diverge, the next action is
        # the experiment that separates them); different goals are not averaged
        same_goal = [p for _, p, _, g, _ in plans if g is best[3]][:3]
        prefix = _common_prefix(same_goal)
        info.plan = prefix if prefix else best[1][:1]
        info.hypothesis, info.goal, info.algorithm = best[2], best[3], best[4]
        info.reason = f"{len(plans)} plans; prefix {len(prefix)} of best {len(best[1])}"
        self.last_info = info
        self.log(f"planner: {info.reason} via {info.algorithm} goal={best[3].name} h={best[2].name} ({time.perf_counter() - t0:.2f}s)")
        return info.plan

    def execute(self, plan: list[Action], H: list[Hypothesis], session, G=None, *, kind: str = "plan") -> ExecResult:
        """Run a plan step by step; stop immediately on level change, game over, prediction mismatch or stuck (spec §10)."""
        current = session.scene
        chosen = self.last_info.hypothesis if self.last_info and self.last_info.hypothesis in H else (H[0] if H else None)
        model = chosen.model if chosen else None
        top_goal = G[0] if G else None
        for i, a in enumerate(plan):
            pred = None
            if model is not None:
                try:
                    pred = model.predict(current, a)
                except Exception:
                    pred = None
            t = session.act(a, kind)
            if t is None:
                return ExecResult("ERROR", current, None, i, {"reason": "env error or budget"})
            if t.status_change is None:
                self.observed[(state_key(t.before), a.label())] = t.after
            if a.type == "CLICK" and not t.diff.is_noop:
                self.responsive.append((a.row, a.col)); self.responsive = self.responsive[-20:]
            if t.status_change in ("LEVEL_UP", "WIN"):
                return ExecResult("PROGRESS", t.after, t, i + 1)
            if t.status_change == "GAME_OVER":
                return ExecResult("FAILED", t.after, t, i + 1)
            if pred is not None and not _equal_ignoring_ui(pred, t.after, model):
                if chosen is not None:
                    chosen.recent_mismatches += 1
                return ExecResult("MISMATCH", t.after, t, i + 1, {"mismatch": scene_mismatch(pred, t.after)})
            if pred is None and model is not None:
                return ExecResult("MISMATCH", t.after, t, i + 1, {"mismatch": "unknown prediction"})
            prog = top_goal.progress(t.after) if top_goal is not None else None
            why = self.stuck.check(t.after.frame_hash, prog)
            if why:
                return ExecResult("STUCK", t.after, t, i + 1, {"reason": why})
            current = t.after
        return ExecResult("EXHAUSTED", current, None, len(plan))


def _equal_ignoring_ui(pred: Scene, obs: Scene, model=None) -> bool:
    from ..wml.evaluate import _equal
    return _equal(pred, obs, True, None, model)


def _common_prefix(plans: list[list[Action]]) -> list[Action]:
    if not plans:
        return []
    out = []
    for i in range(min(len(p) for p in plans)):
        a = plans[0][i]
        if all(p[i] == a for p in plans):
            out.append(a)
        else:
            break
    return out
