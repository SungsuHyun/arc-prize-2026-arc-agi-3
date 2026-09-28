"""orchestrator/session.py — one game session: env + perception + memory + budget glue. `act()` is the single place
an action is executed: it parses the frame, builds the Transition, appends it to memory, charges the budget and keeps
the current scene. Probe and Planner only see this object."""
from __future__ import annotations

import time
from typing import Optional

from ..core.budget import Budget
from ..core.types import Action, Scene, Transition
from ..env.calibrate import ClickMonitor
from ..env.wrapper import EnvWrapper
from ..memory.store import Memory
from ..perception import Perception


class Session:
    def __init__(self, env: EnvWrapper, game_id: str, perception: Perception, memory: Memory, budget: Budget, *, log=None,
                 max_seconds: Optional[float] = None, deadline: Optional[float] = None):
        self.env, self.game_id, self.perception, self.memory, self.budget = env, game_id, perception, memory, budget
        self.log = log or (lambda *a, **k: None)
        self.scene: Optional[Scene] = None
        self.frame = None
        self.level = 1
        self.step_idx = 0
        self.transitions: list[Transition] = []
        self.level_actions: dict[int, int] = {}
        self.click_monitor = ClickMonitor()
        self.t0 = time.time()
        self.deadline = deadline or (self.t0 + max_seconds if max_seconds else None)
        self.reset_allowed = True
        self.errors = 0

    # ── lifecycle ──
    def start(self) -> Scene:
        self.frame = self.env.reset(self.game_id)
        self.level = self.env.status().level
        self.scene = self.perception.parse(self.frame, None)
        self.budget.sync(self.env.status().actions_used)
        return self.scene

    def status(self):
        return self.env.status()

    def finished(self) -> bool:
        st = self.env.status().state
        return st == "WIN" or self.budget.remaining() <= 0 or self.timed_out()

    def timed_out(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def available_actions(self) -> list[Action]:
        return self.env.available_actions()

    def actions_used(self) -> int:
        return self.env.status().actions_used

    # ── the one action path ──
    def act(self, action: Action, kind: str = "plan") -> Optional[Transition]:
        if self.budget.remaining() <= 0 or self.timed_out():
            return None
        if action.type == "RESET":
            level = self.level
            frame = self.env.reset(self.game_id)
            self.budget.charge(kind, 1, level)
            before = self.scene
            self.frame = frame
            self.scene = self.perception.parse(frame, None)   # tracking restarts after a reset
            self.step_idx += 1
            t = self.perception.transition(before, action, self.scene, before_frame=None, after_frame=frame, status_change=None,
                                           tid=f"{self.game_id}:{level}:{self.step_idx}", level=level, step_idx=self.step_idx)
            self.transitions.append(t); self.memory.append(t, self.game_id)
            self.level = self.env.status().level
            self.budget.sync(self.env.status().actions_used)
            return t
        if self.env.status().state == "GAME_OVER":
            return None          # the caller must RESET first
        rt = self.env.step(action)
        if rt.error:
            self.errors += 1
            self.log(f"env error on {action.label()}: {rt.error}")
            if "not available" in rt.error:
                return None
            return None
        level = rt.level
        self.budget.charge(kind, 1, level)
        self.level_actions[level] = self.level_actions.get(level, 0) + 1
        self.step_idx += 1
        before = self.scene
        after = self.perception.parse(rt.after, before)
        t = self.perception.transition(before, action, after, before_frame=rt.before, after_frame=rt.after, status_change=rt.status_change,
                                       intermediate=rt.intermediate_frames, tid=f"{self.game_id}:{level}:{self.step_idx}", level=level, step_idx=self.step_idx)
        if self.perception.merge_confirmed:
            self.perception.refit(self.transitions)
            after = t.after
        self.transitions.append(t); self.memory.append(t, self.game_id)
        self.click_monitor.observe(t)
        self.budget.sync(self.env.status().actions_used)
        self.scene, self.frame = after, rt.after
        if rt.status_change == "LEVEL_UP":
            self.level = self.env.status().level
            # new level: tracking ids restart with the new board
            self.scene = self.perception.parse(rt.after, None)
        return t

    def level_log(self, level: Optional[int] = None) -> list[Transition]:
        lv = self.level if level is None else level
        return [t for t in self.transitions if t.level == lv]
