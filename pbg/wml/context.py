"""wml/context.py — context selection for LLM prompts (spec §14): never the whole log. Fill up to `cap` transitions:
1. all violated transitions (newest first); 2. three representative transitions per action class (distinct diffs first);
3. transitions right before/after a level change; 4. the most recent ones."""
from __future__ import annotations

from typing import Iterable, Optional

from ..core.types import Transition
from ..perception.summarize import summarize, transform_crops


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


def observation_lines(ts: list[Transition], perception=None, max_crops: int = 6) -> list[str]:
    """One summary line per transition; the first transitions that reshape/recolour an object per action key also get
    before/after pixel crops (a summary line cannot show a rotation or a stamp)."""
    out = []; shown: set[tuple[str, int]] = set()
    for t in ts:
        out.append(f"[{t.id}] " + summarize(t))
        ids = [x[0] for x in t.diff.reshaped] + [x[0] for x in t.diff.recolored]
        key = (t.action.key, ids[0]) if ids else None
        if key and key not in shown and len(shown) < max_crops:
            crops = transform_crops(t)
            if crops:
                shown.add(key); out += crops
    return out
