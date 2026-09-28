"""wml/evaluate.py — verification of a world model against the WHOLE transition log (spec §8).

score = exactly predicted transitions / all transitions; coverage = transitions with a non-UNKNOWN prediction.
Per-rule confidence: 1 - violations attributed to the rule / times the rule applied."""
from __future__ import annotations

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
    correct = covered = 0
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
    return EvalResult(correct / len(log), covered / len(log), viol, unk, conf, hashlib.sha1("|".join(keys).encode()).hexdigest()[:12], len(log),
                      {k: (v[0], v[1]) for k, v in per_class.items()})


def _equal(pred, obs, ignore_ui: bool, t: Transition, model=None) -> bool:
    """Object-level equality; objects in ui_strip regions and objects the model calls 'indicator' are compared by
    existence only (display elements may recolour/tick without being part of the mechanics)."""
    if not ignore_ui:
        return scene_equal(pred, obs)
    strips = {r.id for r in obs.regions if r.kind_hint == "ui_strip"}
    loose_ids: set[int] = set()
    if model is not None:
        try:
            roles = model.assign_roles(obs)
            loose_ids = {i for i, r in roles.items() if r == "indicator"}
        except Exception:
            loose_ids = set()
    if not strips and not loose_ids:
        return scene_equal(pred, obs)
    def strict(o):
        return o.region not in strips and o.id not in loose_ids
    a = sorted(o.identity() for o in pred.objects if strict(o)); b = sorted(o.identity() for o in obs.objects if strict(o))
    return a == b


def promotable(res: EvalResult, min_transitions: int = 20, min_per_class: int = 3) -> bool:
    """Promotion condition (spec §8): score == 1.0, >= 20 transitions, >= 3 observations per action class."""
    if res.score < 1.0 or res.n < min_transitions:
        return False
    return all(tot >= min_per_class for _, tot in res.per_class.values())


def make_hypothesis(model: WorldModel, res: EvalResult, code: str = "", name: str = "", origin: str = "llm", recent_mismatches: int = 0) -> Hypothesis:
    return Hypothesis(model, res.score, res.coverage, list(res.violations), code, name, promotable(res), res.prediction_key, origin, recent_mismatches)


def dedupe(H: list[Hypothesis]) -> list[Hypothesis]:
    seen: dict[str, Hypothesis] = {}
    for h in H:
        key = h.prediction_key or h.name
        if key not in seen or h.sort_key() > seen[key].sort_key():
            seen[key] = h
    return list(seen.values())
