"""wml/experiment.py — information-gain experiment selection (spec §8): pick the action on which the hypotheses'
predictions disagree most (entropy of the prediction partition, weighted by hypothesis score)."""
from __future__ import annotations

import math
from typing import Optional

from ..core.contracts import Hypothesis
from ..core.types import Action, Scene


def entropy_of_partition(preds: list, weights: list[float]) -> float:
    groups: dict[str, float] = {}
    for p, w in zip(preds, weights):
        key = "?" if p is None else repr(sorted(o.identity() for o in p.objects)) + repr(sorted((k, repr(v)) for k, v in p.aux.items() if not k.startswith("_")))
        groups[key] = groups.get(key, 0.0) + max(w, 1e-6)
    tot = sum(groups.values())
    if tot <= 0 or len(groups) < 2:
        return 0.0
    return -sum(v / tot * math.log(v / tot, 2) for v in groups.values())


def candidate_actions(scene: Scene, available: list[Action], semantics=None, extra_clicks: list[tuple[int, int]] = ()) -> list[Action]:
    out: list[Action] = []
    for a in available:
        if a.type == "BUTTON":
            out.append(a)
        elif a.type == "CLICK":
            seen = set()
            strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
            for o in sorted(scene.objects, key=lambda o: -o.area):
                if o.region in strips:
                    continue
                c = o.center
                if c not in seen:
                    seen.add(c); out.append(Action.click(*c))
            for r in scene.regions:
                if r.kind_hint != "ui_strip" and r.center not in seen:
                    seen.add(r.center); out.append(Action.click(*r.center))
            for rc in extra_clicks:
                if tuple(rc) not in seen:
                    seen.add(tuple(rc)); out.append(Action.click(*rc))
    return out[:64]


def most_informative_action(H: list[Hypothesis], scene: Scene, available: list[Action], *, tau: float = 0.3, semantics=None,
                            extra_clicks=(), exclude: Optional[set] = None, state_key=None) -> Optional[Action]:
    """`exclude` = {(frame_hash, action label)} already run: an experiment is never repeated from the same state."""
    if len(H) < 2:
        return None
    best, best_gain = None, 0.0
    weights = [max(h.score, 0.05) for h in H]
    for a in candidate_actions(scene, available, semantics, extra_clicks):
        if exclude and ((state_key if state_key is not None else scene.frame_hash), a.label()) in exclude:
            continue
        preds = []
        for h in H:
            try:
                preds.append(h.model.predict(scene, a))
            except Exception:
                preds.append(None)
        gain = entropy_of_partition(preds, weights)
        if gain > best_gain:
            best, best_gain = a, gain
    return best if best_gain > tau else None
