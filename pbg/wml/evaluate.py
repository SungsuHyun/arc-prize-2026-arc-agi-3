"""wml/evaluate.py — verification of a world model against the WHOLE transition log (spec §8).

score = exactly predicted transitions / all transitions; coverage = transitions with a non-UNKNOWN prediction.
Per-rule confidence: 1 - violations attributed to the rule / times the rule applied."""
from __future__ import annotations

from collections import Counter

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..core.contracts import Hypothesis, Rule, RuleModel, WorldModel, scene_equal
from ..core.types import Transition


@dataclass
class EvalResult:
    score: float
    coverage: float
    violations: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    rule_confidence: dict[str, float] = field(default_factory=dict)
    prediction_key: str = ""
    n: int = 0
    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)   # action key -> (correct, total)
    change_score: float = 0.0   # accuracy on transitions where the board changed (a "nothing happens" model scores 0 here)


def _applied_rules(model: WorldModel, t: Transition) -> list[str]:
    if not isinstance(model, RuleModel):
        return []
    s = model.with_roles(t.before)
    names = []
    for r in model.rules():
        try:
            if r.applies(s, t.action):
                names.append(r.name)
        except Exception:
            pass
    return names


def evaluate(model: WorldModel, log: Iterable[Transition], *, skip_reset: bool = True, ignore_ui: bool = True) -> EvalResult:
    log = [t for t in log if not (skip_reset and t.action.type == "RESET")]
    if log and getattr(model, "level_scoped", False):
        last = max(t.level for t in log)
        log = [t for t in log if t.level == last]
    if not log:
        return EvalResult(0.0, 0.0, n=0)
    correct = covered = 0; ch_ok = 0
    viol: list[str] = []; unk: list[str] = []
    applied_total: dict[str, int] = {}; applied_bad: dict[str, int] = {}
    keys = []
    per_class: dict[str, list[int]] = {}
    for t in log:
        try:
            pred = model.predict(t.before, t.action)
        except Exception:
            pred = None
        cls = per_class.setdefault(t.action.key, [0, 0]); cls[1] += 1
        if pred is None:
            unk.append(t.id); keys.append("?")
            continue
        covered += 1
        ok = _equal(pred, t.after, ignore_ui, t, model)
        if not t.diff.is_noop and not _equal(t.before, t.after, ignore_ui, t, model):
            ch_ok += int(ok)      # a change the evaluator can see (not a strip tick / 1-2 px mark)
        keys.append(hashlib.sha1(repr(sorted(o.identity() for o in pred.objects)).encode()).hexdigest()[:8])
        for name in _applied_rules(model, t):
            applied_total[name] = applied_total.get(name, 0) + 1
            if not ok:
                applied_bad[name] = applied_bad.get(name, 0) + 1
        if ok:
            correct += 1; cls[0] += 1
        else:
            viol.append(t.id)
    conf = {n: 1.0 - applied_bad.get(n, 0) / c for n, c in applied_total.items()}
    if isinstance(model, RuleModel):
        for r in model.rules():
            if r.name in conf:
                r.confidence = conf[r.name]
                r.evidence = [t.id for t in log if t.id not in viol and r.name in _applied_rules(model, t)][:50]
                r.violated_by = [t.id for t in log if t.id in viol and r.name in _applied_rules(model, t)][:50]
    unk_set = set(unk)
    ch_total_all = sum(1 for t in log if t.id not in unk_set and not t.diff.is_noop and not _equal(t.before, t.after, ignore_ui, t, model))
    return EvalResult(correct / len(log), covered / len(log), viol, unk, conf, hashlib.sha1("|".join(keys).encode()).hexdigest()[:12], len(log),
                      {k: (v[0], v[1]) for k, v in per_class.items()}, change_score=(ch_ok / ch_total_all) if ch_total_all else 1.0)


def _is_rect(o, scene):
    from .effects import _is_rect as f
    return f(o, scene)


def _effective_bbox(o, scene):
    from .effects import _effective_bbox as f
    return f(o, scene)


def _equal(pred, obs, ignore_ui: bool, t: Transition, model=None) -> bool:
    """Object-level equality; objects in ui_strip regions and objects the model calls 'indicator' are compared by
    existence only (display elements may recolour/tick without being part of the mechanics)."""
    if not ignore_ui:
        return scene_equal(pred, obs)
    bands = [r.bbox for r in list(obs.regions) + list(pred.regions) if r.kind_hint == "ui_strip"]
    loose_ids: set[int] = set()
    if model is not None:
        try:
            roles = model.assign_roles(obs)
            loose_ids = {i for i, r in roles.items() if r == "indicator"}
        except Exception:
            loose_ids = set()

    def in_band(o) -> bool:      # geometry, not region ids: region numbering can differ between the two scenes
        r0, c0, r1, c1 = o.bbox
        return any(b[0] <= r0 and b[1] <= c0 and r1 <= b[2] and c1 <= b[3] for b in bands)

    def covered(o, scene) -> bool:      # fully under another (larger) object: unobservable in the frame
        r0, c0, r1, c1 = o.bbox
        return any(p.id != o.id and p.area > o.area and p.bbox[0] <= r0 and p.bbox[1] <= c0 and r1 <= p.bbox[2] and c1 <= p.bbox[3]
                   for p in scene.objects)

    def strict(o, scene):
        return not in_band(o) and o.id not in loose_ids and o.area > 2 and not covered(o, scene)   # 1-2 px marks/ticks are display elements

    def key(o, scene):
        # a bar/panel is compared by colour and extent (extended under the caps drawn over its ends): a button drawn
        # on top of it, or a piece cut out of it, is not a different bar
        if _is_rect(o, scene) and max(o.height, o.width) >= 8:
            return (int(o.color), tuple(_effective_bbox(o, scene)), "bar")
        return o.identity()
    pa = [o for o in pred.objects if strict(o, pred)]; ob_ = [o for o in obs.objects if strict(o, obs)]
    a = sorted(key(o, pred) for o in pa); b = sorted(key(o, obs) for o in ob_)
    if a == b:
        return True
    ca, cb = Counter(a), Counter(b)
    only_a = list((ca - cb).elements()); only_b = list((cb - ca).elements())
    ka = {key(o, pred): o for o in pa}; kb = {key(o, obs): o for o in ob_}

    def hidden(o, other):        # under an object of the OTHER scene (a bar drawn over a button)
        r0, c0, r1, c1 = o.bbox
        return any(p.area > o.area and p.color != o.color and p.bbox[0] <= r0 and p.bbox[1] <= c0 and r1 <= p.bbox[2] and c1 <= p.bbox[3]
                   for p in other.objects)
    only_a = [k for k in only_a if not (k in ka and hidden(ka[k], obs))]
    only_b = [k for k in only_b if not (k in kb and hidden(kb[k], pred))]
    return not only_a and not only_b


def promotable(res: EvalResult, min_transitions: int = 20, min_per_class: int = 3) -> bool:
    """Promotion condition (spec §8): score == 1.0, >= 20 transitions, >= 3 observations per action class."""
    if res.score < 1.0 or res.n < min_transitions:
        return False
    return all(tot >= min_per_class for _, tot in res.per_class.values())


def make_hypothesis(model: WorldModel, res: EvalResult, code: str = "", name: str = "", origin: str = "llm", recent_mismatches: int = 0) -> Hypothesis:
    return Hypothesis(model, res.score, res.coverage, list(res.violations), code, name, promotable(res), res.prediction_key, origin, recent_mismatches,
                      change_score=res.change_score)


def dedupe(H: list[Hypothesis]) -> list[Hypothesis]:
    seen: dict[str, Hypothesis] = {}
    for h in H:
        # prior-composed variants are deduplicated by name only: an LLM model that predicts the same log can still
        # generalise differently (a permutation table vs a uniform shift), and the composed one must stay available
        key = f"induced:{h.name}" if h.origin == "induced" else (h.prediction_key or h.name)
        if key in seen:
            keep = h if h.sort_key() > seen[key].sort_key() else seen[key]
            keep.recent_mismatches = max(h.recent_mismatches, seen[key].recent_mismatches)   # live mispredictions are not forgotten
            seen[key] = keep
        else:
            seen[key] = h
    return list(seen.values())
