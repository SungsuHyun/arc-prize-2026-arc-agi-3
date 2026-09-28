"""Thin environment wrapper: step, reset, frames and the transition record. No model, no sandbox."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from arcengine import GameAction

from arcnav.frame import Frame

MODEL_TO_ENGINE = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5", "MOUSE": "ACTION6", "ACTION7": "ACTION7"}
ENGINE_TO_MODEL = {v: k for k, v in MODEL_TO_ENGINE.items()}
MOVE_KEYS = ("UP", "DOWN", "LEFT", "RIGHT")
INTERACT_KEYS = ("SPACE", "ACTION7")

Action = Union[str, dict]   # "UP" | {"action": "MOUSE", "row": r, "col": c}


def action_label(act: Action) -> str:
    if isinstance(act, dict):
        return f"MOUSE({act['row']},{act['col']})"
    return act


@dataclass
class Transition:
    action: Action          # str for keys, dict {"action": "MOUSE", "row", "col"} for clicks (what arcnav.rules expects)
    before_frame: Frame
    after_frame: Frame
    level: int
    attempt: int            # attempt index on that level (increments on every reset)
    step: int


class Game:
    def __init__(self, env, game_id: str):
        self.env, self.game_id = env, game_id
        self.frame: Optional[Frame] = None
        self.level, self.levels_total, self.state = 1, 0, "NOT_PLAYED"
        self.valid_actions: list[str] = []
        self.actions_used = self.level_actions = self.step_no = 0
        self.attempt = 0
        self.transitions: list[Transition] = []
        self.level_action_log: list[int] = []
        self.t0 = time.time()
        self.record_path: Optional[Path] = None   # set by the agent: every reset/step is appended here as one JSON line (eval viewer replay)

    def _record(self, rec: dict) -> None:
        if self.record_path is None:
            return
        rec["t"] = round(time.time() - self.t0, 2)
        rec["hash"] = hashlib.sha1(self.frame.ascii.encode()).hexdigest()[:12] if self.frame else None
        rec["levels_total"] = self.levels_total; rec["state"] = self.state
        with open(self.record_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ── engine ────────────────────────────────────────────────────────────
    def _apply(self, raw) -> None:
        self.state = str(raw.state).split(".")[-1]
        self.levels_total = int(raw.win_levels or 0)
        self.level = int(raw.levels_completed) + 1
        self.valid_actions = [ENGINE_TO_MODEL.get(f"ACTION{a}", f"ACTION{a}") for a in (raw.available_actions or [])]
        grids = [g.tolist() if hasattr(g, "tolist") else g for g in raw.frame]
        if grids:
            self.frame = Frame(grids[-1], step=self.step_no, level=self.level)

    def reset(self) -> None:
        self._apply(self.env.step(GameAction.RESET))
        self.attempt += 1
        self.level_actions = 0
        self._record({"kind": "reset", "step": self.step_no, "level": self.level, "attempt": self.attempt})

    def step(self, act: Action) -> dict:
        """Execute one action. Returns {changed, level_completed, game_over, won}. Never resets by itself."""
        name = act["action"] if isinstance(act, dict) else act
        if self.valid_actions and name not in self.valid_actions:
            return {"changed": False, "level_completed": False, "game_over": False, "won": False, "invalid": True}
        ga = GameAction.from_name(MODEL_TO_ENGINE[name]) if hasattr(GameAction, "from_name") else GameAction[MODEL_TO_ENGINE[name]]
        data = {"x": int(act["col"]), "y": int(act["row"])} if name == "MOUSE" else {}
        if data:
            ga.set_data(data)
        before, prev_level = self.frame, self.level
        raw = self.env.step(ga, data=ga.action_data.model_dump())
        self.step_no += 1; self.actions_used += 1; self.level_actions += 1
        self._apply(raw)
        label = {"action": "MOUSE", "row": int(act["row"]), "col": int(act["col"])} if name == "MOUSE" else name
        self.transitions.append(Transition(label, before, self.frame, prev_level, self.attempt, self.step_no))
        completed = self.level > prev_level or self.state == "WIN"
        if completed:
            self.level_action_log.append(self.level_actions); self.level_actions = 0; self.attempt += 1
        self._record({"kind": "step", "step": self.step_no, "action": label, "level": prev_level, "attempt": self.transitions[-1].attempt,
                      "changed": before.ascii != self.frame.ascii, "level_completed": completed, "game_over": self.state == "GAME_OVER", "won": self.state == "WIN"})
        return {"changed": before.ascii != self.frame.ascii, "level_completed": completed, "game_over": self.state == "GAME_OVER",
                "won": self.state == "WIN", "invalid": False}

    # ── views ─────────────────────────────────────────────────────────────
    def level_transitions(self, level: Optional[int] = None) -> list[Transition]:
        lv = self.level if level is None else level
        return [t for t in self.transitions if t.level == lv]

    def attempt_transitions(self) -> list[Transition]:
        """Transitions of the current attempt on the current level (since the last reset / level start)."""
        return [t for t in self.transitions if t.level == self.level and t.attempt == self.attempt]

    def has_move_keys(self) -> bool:
        return any(a in self.valid_actions for a in MOVE_KEYS)
