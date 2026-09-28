"""wml/context.py — context selection for LLM prompts (spec §14): never the whole log. Fill up to `cap` transitions:
1. all violated transitions (newest first); 2. three representative transitions per action class (distinct diffs first);
3. transitions right before/after a level change; 4. the most recent ones."""
from __future__ import annotations

from typing import Iterable, Optional

from ..core.types import Transition
from ..perception.summarize import summarize


def select_context(log: list[Transition], violations: Iterable[str] = (), cap: int = 40) -> list[Transition]:
    viol = set(violations)
    chosen: list[Transition] = []; ids: set[str] = set()

    def add(t: Transition) -> bool:
        if len(chosen) >= cap or t.id in ids or t.action.type == "RESET":
            return False
        chosen.append(t); ids.add(t.id)
        return True

    for t in reversed(log):
        if t.id in viol:
            add(t)
    by_class: dict[str, list[Transition]] = {}
    for t in log:
        by_class.setdefault(t.action.key, []).append(t)
    for key, ts in by_class.items():
        seen_diffs: set[tuple] = set(); n = 0
        for t in reversed(ts):
            sig = (tuple(sorted(t.diff.kinds())), tuple(sorted(v for _, v in t.diff.moved)), t.diff.is_noop)
            if sig in seen_diffs and n >= 1:
                continue
            seen_diffs.add(sig)
            if add(t):
                n += 1
            if n >= 3:
                break
    for i, t in enumerate(log):
        if t.status_change in ("LEVEL_UP", "WIN"):
            add(t)
            if i + 1 < len(log):
                add(log[i + 1])
    for t in reversed(log):
        if len(chosen) >= cap:
            break
        add(t)
    chosen.sort(key=lambda t: (t.level, t.step_idx))
    return chosen


def observation_lines(ts: list[Transition], perception=None) -> list[str]:
    return [f"[{t.id}] " + summarize(t) for t in ts]
