"""orchestrator/knowledge.py — per-game knowledge asset that ACCUMULATES across levels. Later levels of a game build on
the earlier ones (same buttons, same kind of goal, new elements added), so what a level taught must not be re-learned
with actions: which click classes react, which goal template won, which goal predicates held without a level-up.
Persisted in the memory store (semantic/<game>/knowledge.json) so a second run of the game starts with it."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from ..probe.clickmap import ClickMap


class LevelKnowledge:
    def __init__(self) -> None:
        self.clicks = ClickMap()
        self.goal_wins: Counter = Counter()        # template -> levels it completed
        self.goal_fails: Counter = Counter()       # template -> times its predicate held without a level-up
        self.won_goals: list[dict] = []            # [{level, name, template, actions}]
        self.demoted: dict[str, float] = {}        # goal name -> penalty (carried into the next level)
        self.levels_seen: set[int] = set()
        # class-level click rules verified on earlier levels: trigger class -> {mover colours, displacements seen, n}.
        # Positions change between levels, classes do not: on a new level ONE click per trigger object is enough to
        # re-instantiate the rule (the displacement must be one already seen for the class)
        self.click_rules: dict[str, dict] = {}
        self.effect_rules: dict[str, dict] = {}    # trigger class -> {signatures: [...], n}: joint effects seen on earlier levels

    # ── goal confidence prior for THIS game (stronger than the global usage stats) ──
    def goal_stats(self, global_stats: Optional[dict] = None) -> dict:
        out = dict(global_stats or {})
        for name, wins in self.goal_wins.items():
            fails = self.goal_fails.get(name, 0)
            # a template that completed a level of this game starts the next level at 0.75 (+clue, +structure bonus),
            # a failed one lower; kept below 1.0 so the structure bonus (same axis as the winning goal) can still rank
            out[name] = {"games_used": 4 * (wins + fails), "games_verified": 3 * (wins + fails) if wins else 0}
        return out

    def record_rules(self, model) -> int:
        """Harvest class-level click rules from a verified model (rules carry `meta` set by the induction)."""
        n = 0
        for r in getattr(model, "rules", lambda: [])():
            meta = getattr(r, "meta", None)
            if meta and meta.get("kind") == "effect" and r.confidence >= 0.9:
                e = self.effect_rules.setdefault(meta["trigger_class"], {"signatures": [], "n": 0})
                if meta["signature"] not in e["signatures"]:
                    e["signatures"].append(meta["signature"])
                e["n"] += 1; n += 1
                continue
            if not meta or meta.get("kind") != "click_shift" or r.confidence < 0.9:
                continue
            e = self.click_rules.setdefault(meta["trigger_class"], {"mover_colors": [], "displacements": [], "n": 0})
            for c in meta.get("mover_colors", []):
                if c not in e["mover_colors"]:
                    e["mover_colors"].append(c)
            d = list(meta["displacement"])
            if d not in e["displacements"]:
                e["displacements"].append(d)
            e["n"] += 1; n += 1
        return n

    def goal_structure_bonus(self, goal) -> float:
        """+0.1 for a goal with the same template AND the same structural parameters (axis, kind) as a goal that won
        an earlier level; ids and colours are level-specific and ignored."""
        for w in self.won_goals:
            if w.get("template") == goal.template and w.get("structure") == _structure(goal.params):
                return 0.1
        return 0.0

    def record_win(self, level: int, goal, actions: int) -> None:
        if goal is None:
            return
        self.goal_wins[goal.template] += 1
        self.won_goals.append({"level": level, "name": goal.name, "template": goal.template, "actions": actions, "structure": _structure(goal.params)})

    def record_fail(self, goal) -> None:
        if goal is not None:
            self.goal_fails[goal.template] += 1

    # ── persistence ──
    def to_json(self) -> dict:
        return {"clicks": self.clicks.to_json(), "goal_wins": dict(self.goal_wins), "goal_fails": dict(self.goal_fails),
                "won_goals": self.won_goals, "demoted": self.demoted, "levels_seen": sorted(self.levels_seen), "click_rules": self.click_rules, "effect_rules": self.effect_rules}

    @classmethod
    def from_json(cls, d: dict) -> "LevelKnowledge":
        k = cls()
        k.clicks = ClickMap.from_json(d.get("clicks", {}))
        k.goal_wins = Counter(d.get("goal_wins", {})); k.goal_fails = Counter(d.get("goal_fails", {}))
        k.won_goals = list(d.get("won_goals", [])); k.demoted = dict(d.get("demoted", {}))
        k.levels_seen = set(d.get("levels_seen", [])); k.click_rules = dict(d.get("click_rules", {})); k.effect_rules = dict(d.get("effect_rules", {}))
        return k

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=1, default=str))

    @classmethod
    def load(cls, path: Path) -> "LevelKnowledge":
        try:
            return cls.from_json(json.loads(path.read_text())) if path.exists() else cls()
        except Exception:
            return cls()

    def summary(self) -> str:
        return f"clicks[{self.clicks.summary()}] wins={dict(self.goal_wins)} fails={dict(self.goal_fails)} rules={list(self.click_rules)}"


def _structure(params: dict) -> dict:
    """Structural (level-independent) goal parameters: string values that are not roles/regions/ids."""
    out = {}
    for k, v in (params or {}).items():
        if k in ("color", "colors", "target", "canvas", "canvas_bbox", "a", "b", "src", "dst", "id"):
            continue
        if isinstance(v, str) and not v.startswith(("custom:", "R")):
            out[k] = v
    return out
