"""core/contracts.py — interfaces every world model / goal must satisfy (spec §4). The verifier calls only these."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from .types import Action, Scene


@runtime_checkable
class WorldModel(Protocol):
    def assign_roles(self, scene: Scene) -> dict[int, str]: ...
    def predict(self, scene: Scene, action: Action) -> Optional[Scene]: ...   # None == UNKNOWN
    def rules(self) -> list["Rule"]: ...


@dataclass
class Rule:
    name: str
    applies: Callable[[Scene, Action], bool]
    effect: Callable[[Scene, Action], Optional[Scene]]     # None == UNKNOWN for this transition
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)     # supporting transition ids
    violated_by: list[str] = field(default_factory=list)  # failing transition ids (filled by evaluate)
    source: str = "llm"                                   # llm | prior | induced | transferred

    def __repr__(self) -> str:  # pragma: no cover
        return f"Rule({self.name}, conf={self.confidence:.2f})"


@runtime_checkable
class Goal(Protocol):
    name: str
    confidence: float

    def is_goal(self, scene: Scene) -> bool: ...
    def progress(self, scene: Scene) -> float: ...   # 0.0..1.0 heuristic


class RuleModel:
    """Standard WorldModel: a role assigner + an ordered rule list applied sequentially.

    predict() applies every rule whose applies() is true, in order; each effect receives the scene produced by the
    previous one (rules therefore compose: move, then collision cancel, then counter). An effect returning None makes
    the whole prediction UNKNOWN. If no rule applies the result is `default` ("unknown" -> None, "noop" -> unchanged)."""

    def __init__(self, rules: list[Rule], role_fn: Optional[Callable[[Scene], dict[int, str]]] = None, default: str = "unknown",
                 name: str = "rule_model"):
        self._rules = list(rules)
        self._role_fn = role_fn
        self.default = default
        self.name = name
        self.verified = False
        self.level_scoped = False      # True: positional rules learned for the current level -> verified on that level's log only

    def assign_roles(self, scene: Scene) -> dict[int, str]:
        if self._role_fn is None:
            return {o.id: (o.role or "unknown") for o in scene.objects}
        return dict(self._role_fn(scene))

    def with_roles(self, scene: Scene) -> Scene:
        roles = self.assign_roles(scene)
        objs = []
        for o in scene.objects:
            role = roles.get(o.id, o.role)
            if role != o.role:
                o = o.moved(0, 0); o.role = role
            objs.append(o)
        return scene.copy(objects=objs)

    def predict(self, scene: Scene, action: Action) -> Optional[Scene]:
        s = self.with_roles(scene)
        s.aux["_before"] = scene
        applied = False
        for rule in self._rules:
            try:
                if rule.applies(s, action):
                    applied = True
                    nxt = rule.effect(s, action)
                    if nxt is None:
                        return None
                    nxt.aux["_before"] = scene
                    s = nxt
            except Exception as e:  # a broken rule = unknown prediction, never a crash
                s.aux["_error"] = f"{rule.name}: {type(e).__name__}: {e}"
                return None
        if not applied and self.default == "unknown":
            return None
        s.aux.pop("_before", None)
        return s

    def rules(self) -> list[Rule]:
        return self._rules

    def rule(self, name: str) -> Optional[Rule]:
        for r in self._rules:
            if r.name == name:
                return r
        return None


@dataclass
class Hypothesis:
    model: WorldModel
    score: float = 0.0          # transition-level accuracy on the whole log
    coverage: float = 0.0       # fraction of transitions with a non-UNKNOWN prediction
    violations: list[str] = field(default_factory=list)   # failing transition ids
    code: str = ""              # source that built the model (LLM / induction)
    name: str = ""
    verified: bool = False
    prediction_key: str = ""    # hash of all predictions on the log (dedupe)
    origin: str = "llm"         # llm | induced | transferred | prior
    recent_mismatches: int = 0  # mispredictions during execution since the last re-verification that changed the model

    def sort_key(self) -> tuple:
        return (self.score, self.coverage)

    def plan_weight(self) -> float:
        """Weight used when choosing plans: verified log accuracy, penalised for live mispredictions."""
        return max(0.05, self.score - 0.15 * self.recent_mismatches)


def scene_equal(a: Optional[Scene], b: Optional[Scene], *, ignore_roles: bool = True) -> bool:
    """Object-level equality used by evaluation and execution monitoring (spec §8): same multiset of
    (colour, bbox, shape_sig). Roles are ignored because the observed scene has none until the model assigns them."""
    if a is None or b is None:
        return False
    return sorted(o.identity() for o in a.objects) == sorted(o.identity() for o in b.objects)


def scene_mismatch(a: Scene, b: Scene) -> dict:
    """Which objects differ between a predicted and an observed scene (for prompts and diagnostics)."""
    ia = {o.identity(): o for o in a.objects}; ib = {o.identity(): o for o in b.objects}
    only_pred = [o for k, o in ia.items() if k not in ib]; only_obs = [o for k, o in ib.items() if k not in ia]
    return {"only_predicted": [(o.id, o.color, o.bbox) for o in only_pred], "only_observed": [(o.id, o.color, o.bbox) for o in only_obs]}
