"""env/wrapper.py — Env Wrapper (spec §5): wraps the environment API so upper layers never see the raw interface.

* multi-frame responses: the last stable frame is `after`, the rest go to intermediate_frames
* click coordinates are grid cells (calibrate_click confirms scale/offset)
* determinism: stochasticity_score = fraction of (frame_hash, action) repeats whose result hash differed
* budget metering: actions used, remaining, whether RESET costs an action
* step() never raises: environment errors land in RawTransition.error
* every raw request/response is appended to episodic/{game_id}/raw.jsonl (input of ReplayEnv)"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..core.types import Action, Frame, grid_hash, grid_to_rows

BUTTON_IDS = (1, 2, 3, 4, 5, 7)
CLICK_ID = 6


@dataclass
class EnvStatus:
    state: Literal["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]
    level: int
    levels_total: Optional[int]
    actions_used: int
    budget_total: Optional[int]
    stochasticity_score: float
    reset_costs_action: Optional[bool] = None

    def to_json(self) -> dict:
        return self.__dict__.copy()


@dataclass
class RawTransition:
    before: Optional[Frame]
    action: Action
    after: Optional[Frame]
    status: EnvStatus
    status_change: Optional[Literal["LEVEL_UP", "WIN", "GAME_OVER"]] = None
    intermediate_frames: list[Frame] = field(default_factory=list)
    available_actions: list[Action] = field(default_factory=list)
    error: Optional[str] = None
    step_idx: int = 0
    level: int = 1


def _state_name(raw_state) -> str:
    return str(raw_state).split(".")[-1]


def stable_frames(grids: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (stable_after, intermediates). Stable = the last of two consecutive identical frames if any; else the last frame."""
    if not grids:
        raise ValueError("empty frame list")
    if len(grids) == 1:
        return grids[0], []
    for i in range(len(grids) - 1, 0, -1):
        if np.array_equal(grids[i], grids[i - 1]):
            return grids[i], grids[:i - 1]
    return grids[-1], grids[:-1]


class EnvWrapper:
    """Abstract interface (spec §5). Concrete: ArcadeEnv (offline/online arc_agi engine) and ReplayEnv."""

    game_id: str = ""

    def reset(self, game_id: Optional[str] = None) -> Frame: raise NotImplementedError
    def step(self, action: Action) -> RawTransition: raise NotImplementedError
    def available_actions(self) -> list[Action]: raise NotImplementedError
    def status(self) -> EnvStatus: raise NotImplementedError
    def calibrate_click(self) -> dict: return {"scale": 1, "offset": (0, 0), "verified": False}
    def close(self) -> None: pass


class ArcadeEnv(EnvWrapper):
    """arc_agi / arcengine environment. `env` is the object returned by Arcade.make(game_id)."""

    def __init__(self, env, game_id: str, *, log_dir: Optional[Path] = None, budget_total: Optional[int] = None):
        self.env, self.game_id = env, game_id
        self.log_path = (Path(log_dir) / "raw.jsonl") if log_dir else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.budget_total = budget_total
        self.frame: Optional[Frame] = None
        self.state = "NOT_PLAYED"
        self.level, self.levels_total = 1, None
        self._avail: list[int] = []
        self.actions_used = 0
        self.step_idx = 0
        self._results: dict[tuple[str, str], set[str]] = {}
        self._repeat_total = self._repeat_diff = 0
        self.reset_costs_action: Optional[bool] = None
        self.t0 = time.time()
        self.click_calibration = {"scale": 1, "offset": (0, 0), "verified": False}

    # ── engine glue ──
    @staticmethod
    def _engine_action(action: Action):
        from arcengine import GameAction
        if action.type == "RESET":
            return GameAction.RESET, None
        if action.type == "CLICK":
            ga = GameAction.ACTION6
            ga.set_data({"x": int(action.col), "y": int(action.row)})
            return ga, ga.action_data.model_dump()
        return getattr(GameAction, f"ACTION{action.id}"), None

    def _apply(self, raw, action: Action) -> tuple[Frame, list[Frame], Optional[str]]:
        prev_level, prev_state = self.level, self.state
        self.state = _state_name(raw.state)
        self.levels_total = int(raw.win_levels or 0) or self.levels_total
        self.level = int(raw.levels_completed) + 1
        self._avail = [int(a) for a in (raw.available_actions or [])]
        grids = [np.asarray(g, dtype=np.int8) for g in (raw.frame or [])]
        if not grids and self.frame is not None:
            grids = [self.frame.grid]
        after_grid, inter = stable_frames(grids)
        after = Frame(after_grid, self.level, self.step_idx)
        inters = [Frame(g, self.level, self.step_idx) for g in inter]
        change = None
        if self.state == "WIN":
            change = "WIN"
        elif self.state == "GAME_OVER":
            change = "GAME_OVER"
        elif self.level > prev_level and prev_state != "NOT_PLAYED":
            change = "LEVEL_UP"
        self._record({"kind": action.type.lower() if action.type == "RESET" else "step", "step_idx": self.step_idx, "action": action.to_json(),
                      "frames": [grid_to_rows(g) for g in grids], "state": self.state, "level": self.level, "levels_total": self.levels_total,
                      "available_actions": self._avail, "status_change": change, "t": round(time.time() - self.t0, 3)})
        return after, inters, change

    def _record(self, rec: dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # ── interface ──
    def reset(self, game_id: Optional[str] = None) -> Frame:
        from arcengine import GameAction
        before_used = self.actions_used
        raw = self.env.step(GameAction.RESET)
        self.frame, _, _ = self._apply(raw, Action.reset())
        # the engine counts RESET as an action in its action counters; we keep our own metering
        if self.state != "NOT_PLAYED" and before_used > 0 and self.reset_costs_action is None:
            self.reset_costs_action = True
        return self.frame

    def step(self, action: Action) -> RawTransition:
        before = self.frame
        st = self.status()
        if action.type == "RESET":
            try:
                frame = self.reset()
            except Exception as e:
                return RawTransition(before, action, before, st, error=f"{type(e).__name__}: {e}", step_idx=self.step_idx, level=self.level)
            self.actions_used += 1
            return RawTransition(before, action, frame, self.status(), None, [], self.available_actions(), step_idx=self.step_idx, level=self.level)
        if action.type == "CLICK" and CLICK_ID not in self._avail or action.type == "BUTTON" and action.id not in self._avail:
            return RawTransition(before, action, before, st, error=f"action {action.label()} not available ({self._avail})", step_idx=self.step_idx, level=self.level)
        try:
            ga, data = self._engine_action(action)
            raw = self.env.step(ga, data=data) if data is not None else self.env.step(ga)
        except Exception as e:
            return RawTransition(before, action, before, st, error=f"{type(e).__name__}: {e}", step_idx=self.step_idx, level=self.level)
        self.step_idx += 1; self.actions_used += 1
        level_before = self.level
        after, inters, change = self._apply(raw, action)
        self.frame = after
        if before is not None:
            key = (before.hash, action.label())
            seen = self._results.setdefault(key, set())
            if seen:
                self._repeat_total += 1
                if after.hash not in seen:
                    self._repeat_diff += 1
            seen.add(after.hash)
        return RawTransition(before, action, after, self.status(), change, inters, self.available_actions(), step_idx=self.step_idx, level=level_before)

    def available_actions(self) -> list[Action]:
        out = [Action.button(i) for i in self._avail if i in BUTTON_IDS]
        if CLICK_ID in self._avail:
            out.append(Action("CLICK"))
        return out

    def status(self) -> EnvStatus:
        score = self._repeat_diff / self._repeat_total if self._repeat_total else 0.0
        return EnvStatus(self.state, self.level, self.levels_total, self.actions_used, self.budget_total, round(score, 4), self.reset_costs_action)

    def calibrate_click(self) -> dict:
        """The arcengine click space is the 64x64 grid itself (x=col, y=row, 0..63): scale 1, offset 0.
        Verified structurally (no action spent): the API contract states cell units, so corner clicks are not needed."""
        self.click_calibration = {"scale": 1, "offset": (0, 0), "verified": True, "method": "api-contract"}
        return self.click_calibration
