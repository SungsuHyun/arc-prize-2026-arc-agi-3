"""goal/filter.py — level-up back-inference filter (spec §9): keep predicates true right before/after a LEVEL_UP
and false in the previous 10 distinct states."""
from __future__ import annotations

from typing import Iterable

from ..core.types import Transition
from .templates import GoalInstance


def filter_by_levelup(G: list[GoalInstance], log: Iterable[Transition], *, prior_window: int = 10, roles_fn=None, model=None) -> list[GoalInstance]:
    """At a LEVEL_UP the frame returned is the NEXT level's board, so the winning state is never observed directly: a goal
    is kept when it holds on the last state of the level, or on the model's prediction of the winning action's result."""
    log = list(log)
    for idx, t in enumerate(log):
        if t.status_change not in ("LEVEL_UP", "WIN"):
            continue
        before = roles_fn(t.before) if roles_fn else t.before
        win_states = [before]
        if model is not None:
            try:
                pred = model.predict(t.before, t.action)
                if pred is not None:
                    win_states.append(roles_fn(pred) if roles_fn else pred)
            except Exception:
                pass
        G = [g for g in G if any(g.is_goal(w) for w in win_states)]
        prior = [x for x in log[:idx] if x.before.frame_hash != t.before.frame_hash and x.level == t.level]
        seen = set(); distinct = []
        for x in reversed(prior):
            if x.before.frame_hash in seen:
                continue
            seen.add(x.before.frame_hash); distinct.append(x)
            if len(distinct) >= prior_window:
                break
        G = [g for g in G if not any(g.is_goal(roles_fn(x.before) if roles_fn else x.before) for x in distinct)]
        for g in G:
            g.confidence = min(1.0, g.confidence + 0.3)
    return G
