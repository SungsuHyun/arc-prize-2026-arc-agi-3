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

    # ── goal confidence prior for THIS game (stronger than the global usage stats) ──
    def goal_stats(self, global_stats: Optional[dict] = None) -> dict:
        out = dict(global_stats or {})
        for name, wins in self.goal_wins.items():
            fails = self.goal_fails.get(name, 0)
            # a template that completed a level of this game starts the next level at 0.85 (+clue), a failed one lower
            out[name] = {"games_used": wins + fails, "games_verified": round(0.85 * (wins + fails)) if wins else 0}
        return out

    def record_win(self, level: int, goal, actions: int) -> None:
        if goal is None:
            return
        self.goal_wins[goal.template] += 1
        self.won_goals.append({"level": level, "name": goal.name, "template": goal.template, "actions": actions})

    def record_fail(self, goal) -> None:
        if goal is not None:
            self.goal_fails[goal.template] += 1

    # ── persistence ──
    def to_json(self) -> dict:
        return {"clicks": self.clicks.to_json(), "goal_wins": dict(self.goal_wins), "goal_fails": dict(self.goal_fails),
                "won_goals": self.won_goals, "demoted": self.demoted, "levels_seen": sorted(self.levels_seen)}

    @classmethod
    def from_json(cls, d: dict) -> "LevelKnowledge":
        k = cls()
        k.clicks = ClickMap.from_json(d.get("clicks", {}))
        k.goal_wins = Counter(d.get("goal_wins", {})); k.goal_fails = Counter(d.get("goal_fails", {}))
        k.won_goals = list(d.get("won_goals", [])); k.demoted = dict(d.get("demoted", {}))
        k.levels_seen = set(d.get("levels_seen", []))
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
        return f"clicks[{self.clicks.summary()}] wins={dict(self.goal_wins)} fails={dict(self.goal_fails)}"
