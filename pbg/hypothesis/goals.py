"""hypothesis/goals.py — the win condition, inferred from the moment a level was completed (docs/029, task 8).

On level 2+ the models are often exact (lp85, vc33, m0r0 verified at 1.00) and still no template goal yields a plan:
the win condition is not in the template library. What we do have is evidence: the board right before and right after
the previous level ended. The reasoning model is asked for ONE thing — a goal predicate as code — and the predicate is
checked against that evidence (true after the win, false before it and on earlier boards of that level, false on the
current board). Only predicates that pass become goals; they are tried first by the planner and carried to the next
level, where the new win evidence re-checks them."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.types import Scene
from ..wml.prompts import API_SUMMARY
from .contract import frame_text

MAX_ROUNDS_PER_LEVEL = 3


@dataclass
class WinEvidence:
    level: int
    action: str
    before: Scene
    after: Scene
    before_grid: np.ndarray
    after_grid: np.ndarray
    earlier: list = field(default_factory=list)     # scenes of that level where the goal was NOT met (negatives)


GOAL_CONTRACT = '''You infer the WIN CONDITION of an unknown pixel board game. A level was just completed: you see the board right
BEFORE the winning action and right AFTER it, earlier boards of that level (where the level was not yet won), and the
current board of the NEXT level. Write the condition as code and nothing else:

def build_goal():
    class Goal:
        name = "<short name>"
        def is_goal(self, scene) -> bool: ...      # TRUE on the after-board, FALSE on the before-board and on every earlier board
        def progress(self, scene) -> float: ...   # 0..1, how close the board is to the condition (higher = closer)
    return Goal()

Reply with exactly ONE ```python block. Rules: describe the condition through object RELATIONS (roles, colours, counts,
alignment, containment, adjacency, equality of positions/shapes), never through coordinates of this level: the same
predicate must apply to the next level, where objects sit elsewhere and there may be more of them. Scene, Object and
numpy (as np) are already defined; do not import. Objects carry .role (from the model), .color, .colors, .bbox
(r0, c0, r1, c1), .center, .height, .width, .area, .shape_sig, .region; scene.objects, scene.by_role(r), scene.by_color(c),
scene.regions (.id, .bg_color, .bbox, .kind_hint). Objects in a ui strip are counters, not part of the condition.
'''


def _objects_text(scene: Scene, limit: int = 60) -> str:
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    objs = sorted([o for o in scene.objects if o.region not in strips], key=lambda o: (-o.area, o.id))
    lines = []
    for o in objs[:limit]:
        col = f"colours={list(o.colors)}" if len(o.colors) > 1 else f"colour={o.color}"
        lines.append(f"  obj#{o.id} role={o.role or '?'} {col} bbox=({o.bbox[0]},{o.bbox[1]})-({o.bbox[2]},{o.bbox[3]}) size={o.height}x{o.width}")
    if len(objs) > limit:
        lines.append(f"  ... {len(objs) - limit} more")
    return "\n".join(lines) or "  (no objects)"


def _window(a: np.ndarray, b: np.ndarray, pad: int = 2) -> Optional[tuple]:
    ys, xs = np.nonzero(np.asarray(a) != np.asarray(b))
    if len(ys) == 0:
        return None
    return (max(0, int(ys.min()) - pad), max(0, int(xs.min()) - pad), min(a.shape[0], int(ys.max()) + 1 + pad), min(a.shape[1], int(xs.max()) + 1 + pad))


def goal_prompt(ev: WinEvidence, roled: Callable[[Scene], Scene], current: Scene, current_grid: np.ndarray, *, rejected: list = (), feedback: list = (),
                alternative: int = 0) -> list[dict]:
    before, after = roled(ev.before), roled(ev.after)
    user = [f"# Level {ev.level} was completed by the action {ev.action}.", "",
            "# Board right BEFORE the winning action (objects with roles)", _objects_text(before), "",
            "# Board right AFTER it (the level ended here)", _objects_text(after), ""]
    w = _window(ev.before_grid, ev.after_grid)
    if w is not None and (w[2] - w[0]) <= 30 and (w[3] - w[1]) <= 48:
        user += [f"# The cells that changed with the winning action, rows {w[0]}-{w[2] - 1}, cols {w[1]}-{w[3] - 1} (hex colours)",
                 "before:", frame_text(ev.before_grid, (w[0], w[2]), (w[1], w[3])), "after:", frame_text(ev.after_grid, (w[0], w[2]), (w[1], w[3])), ""]
    for i, sc in enumerate(ev.earlier[:4]):
        user += [f"# An earlier board of level {ev.level} where the level was NOT yet won ({i + 1})", _objects_text(roled(sc), limit=40), ""]
    user += [f"# Current board (level {ev.level + 1}, not won): the predicate must be FALSE here and reachable by play", _objects_text(roled(current)), ""]
    if rejected:
        user += ["# Rejected earlier (do not repeat): " + "; ".join(str(r)[:60] for r in rejected), ""]
    if feedback:
        user += ["# Why the previous proposals failed verification"] + [f"- {f}" for f in feedback[-4:]] + [""]
    if alternative:
        user += [f"# Alternative {alternative}: consider a different reading of what made the level end (a count reached, a shape completed, an alignment, a containment, a colour match)."]
    user.append("Write build_goal() now.")
    return [{"role": "system", "content": GOAL_CONTRACT + "\n" + API_SUMMARY}, {"role": "user", "content": "\n".join(user)}]


def verify_goal(goal, roled: Callable[[Scene], Scene], ev: WinEvidence, current: Optional[Scene]) -> tuple[bool, str]:
    """True on the after-board, false before it and on the earlier boards, false on the current board, progress in [0,1)."""
    try:
        if not goal.is_goal(roled(ev.after)):
            return False, f"'{goal.name}' is FALSE on the board right after level {ev.level} was completed"
        if goal.is_goal(roled(ev.before)):
            return False, f"'{goal.name}' is already TRUE on the board right before the winning action (it must become true BY that action)"
        for i, sc in enumerate(ev.earlier):
            if goal.is_goal(roled(sc)):
                return False, f"'{goal.name}' is TRUE on an earlier board of level {ev.level} where the level was not won"
        if current is not None:
            if goal.is_goal(roled(current)):
                return False, f"'{goal.name}' is already TRUE on the current board of level {ev.level + 1}, which is not won"
            p = float(goal.progress(roled(current)))
            if not (0.0 <= p <= 1.0):
                return False, f"'{goal.name}'.progress returned {p!r} on the current board (must be within 0..1)"
        p_after = float(goal.progress(roled(ev.after))); p_before = float(goal.progress(roled(ev.before)))
        if p_after < p_before:
            return False, f"'{goal.name}'.progress is lower after the win ({p_after:.2f}) than before ({p_before:.2f}): progress must rise toward the condition"
    except Exception as e:
        return False, f"'{getattr(goal, 'name', '?')}' raised {type(e).__name__}: {str(e)[:100]}"
    return True, "ok"


def propose_goals(llm, sandbox, ev: WinEvidence, roled: Callable[[Scene], Scene], current: Scene, current_grid: np.ndarray, *, K: int = 2,
                  rejected: list = (), feedback: list = (), log=None) -> tuple[list, list[str]]:
    """Ask the model for K win predicates; return (accepted GoalInstances, verification failure reasons)."""
    log = log or (lambda *a, **k: None)
    codes: list = []

    def gen(k: int) -> None:
        try:
            msgs = goal_prompt(ev, roled, current, current_grid, rejected=rejected, feedback=feedback, alternative=k)
            reply = llm.chat(msgs, purpose="goal_template", use_cache=(k == 0))
            code = llm.extract_code(reply)
            if code:
                codes.append(code)
            else:
                log(f"goal proposal {k}: no code block")
        except Exception as e:
            log(f"goal proposal {k} failed: {e!r}")
    threads = [threading.Thread(target=gen, args=(k,), daemon=True) for k in range(K)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=float(getattr(llm, "cfg", {}).get("timeout", 300)) + 30)
    accepted, reasons = [], []
    for code in codes:
        goal = sandbox.load_goal(code)
        if goal is None:
            reasons.append(f"the code did not load: {sandbox.last_error or 'build_goal missing'}"); continue
        ok, why = verify_goal(goal, roled, ev, current)
        if ok:
            goal.origin = "llm"; goal.code = code; goal.confidence = 0.9; goal.template = f"win:{goal.name}"
            accepted.append(goal)
        else:
            reasons.append(why)
    return accepted, reasons
