"""core/budget.py — action budget accounting and the category caps of spec §12."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

PER_LEVEL_FRACTIONS = ("level_probe",)   # fractions that apply per level rather than cumulatively

DEFAULT_BUDGET = {"total": 2000, "initial_probe": 0.12, "experiment": 0.25, "level_probe": 0.05, "reprobe": 0.08,
                  "low_budget_fraction": 0.20, "llm_calls_max": 60,
                  "per_level": {"reprobe": 32, "experiment": 12, "level_probe": 20, "initial_clicks": 16}}


def load_budget_config(path: Optional[Path] = None) -> dict:
    cfg = dict(DEFAULT_BUDGET)
    p = path or Path(__file__).resolve().parents[1] / "configs" / "budget.yaml"
    if p.exists():
        cfg.update(yaml.safe_load(p.read_text()) or {})
    return cfg


class Budget:
    """Tracks actions used per category against fractional caps of the total B.

    cap(kind) -> remaining actions allowed for that category; allows(kind) -> at least one action is allowed;
    charge(kind, n) records usage. Categories: initial_probe (once per game), level_probe (per level), experiment
    (cumulative), reprobe (cumulative), plan (everything else, uncapped)."""

    def __init__(self, total: Optional[int] = None, config: Optional[dict] = None):
        self.cfg = dict(config or load_budget_config())
        self.total = int(total or self.cfg["total"])
        self.used_by: dict[str, int] = {}
        self.level_used: dict[str, dict[int, int]] = {}
        self.llm_calls = 0

    # ── accounting ──
    def used(self) -> int:
        return sum(self.used_by.values())

    def remaining(self) -> int:
        return max(0, self.total - self.used())

    def fraction_left(self) -> float:
        return self.remaining() / max(1, self.total)

    def low(self) -> bool:
        return self.fraction_left() <= float(self.cfg["low_budget_fraction"])

    def charge(self, kind: str, n: int = 1, level: Optional[int] = None) -> None:
        self.used_by[kind] = self.used_by.get(kind, 0) + n
        if level is not None:
            lv = self.level_used.setdefault(kind, {}); lv[level] = lv.get(level, 0) + n

    def reset_level(self, level: int, kind: str) -> None:
        """A new attempt (after RESET) gets the per-level allowance of `kind` again."""
        self.level_used.setdefault(kind, {})[level] = 0

    def sync(self, env_actions_used: int) -> None:
        """Reconcile with the environment's own action counter (resets and engine-side counting): any actions the
        environment counted that we did not charge go to the 'other' category so the total is never exceeded."""
        extra = env_actions_used - self.used()
        if extra > 0:
            self.used_by["other"] = self.used_by.get("other", 0) + extra

    def cap(self, kind: str, level: Optional[int] = None) -> int:
        """Remaining allowance for `kind`: the fraction-of-B cap (cumulative) AND, when configured, an absolute per-level
        cap (spec §12 rows are fractions of B; per-level absolute caps keep exploration near the level's own scale)."""
        frac = self.cfg.get(kind)
        if frac is None:
            return self.remaining()
        if kind in PER_LEVEL_FRACTIONS and level is not None:
            limit = int(frac * self.total) - self.level_used.get(kind, {}).get(level, 0)   # "per level" rows of spec §12
        else:
            limit = int(frac * self.total) - self.used_by.get(kind, 0)
        per_level = (self.cfg.get("per_level") or {}).get(kind)
        if per_level is not None and level is not None:
            limit = min(limit, int(per_level) - self.level_used.get(kind, {}).get(level, 0))
        return max(0, min(limit, self.remaining()))

    def allows(self, kind: str, n: int = 1, level: Optional[int] = None) -> bool:
        return self.cap(kind, level) >= n and self.remaining() >= n

    def allows_llm(self) -> bool:
        return self.llm_calls < int(self.cfg.get("llm_calls_max", 60))

    def snapshot(self) -> dict:
        return {"total": self.total, "used": self.used(), "remaining": self.remaining(), "by_kind": dict(self.used_by), "llm_calls": self.llm_calls}
