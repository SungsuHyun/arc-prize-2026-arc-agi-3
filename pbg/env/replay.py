"""env/replay.py — ReplayEnv: the EnvWrapper interface over a recorded raw.jsonl (spec §5, §16).

The log is a sequence of records {kind: reset|step, action, frames, state, level, ...}. step(action) serves the next
record when the requested action equals the recorded one; a different action is an error (RawTransition.error) unless
`lenient` is set, in which case the recorded action is served anyway (used to drive module pipelines over human logs)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..core.types import Action, Frame, rows_to_grid
from .wrapper import BUTTON_IDS, CLICK_ID, EnvStatus, EnvWrapper, RawTransition, stable_frames


def read_raw_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class ReplayEnv(EnvWrapper):
    def __init__(self, records: list[dict] | Path, game_id: str = "", *, lenient: bool = False):
        self.records = read_raw_log(records) if isinstance(records, (str, Path)) else list(records)
        self.game_id = game_id or (self.records[0].get("game_id", "") if self.records else "")
        self.lenient = lenient
        self.pos = 0
        self.frame: Optional[Frame] = None
        self.state, self.level, self.levels_total = "NOT_PLAYED", 1, None
        self._avail: list[int] = []
        self.actions_used = 0
        self.step_idx = 0

    def _serve(self, rec: dict, action: Action) -> tuple[Frame, list[Frame], Optional[str]]:
        self.state = rec.get("state", "NOT_FINISHED")
        self.level = int(rec.get("level", 1)); self.levels_total = rec.get("levels_total")
        self._avail = [int(a) for a in rec.get("available_actions", [])]
        grids = [rows_to_grid(rows) for rows in rec["frames"]]
        after_grid, inter = stable_frames(grids)
        return Frame(after_grid, self.level, self.step_idx), [Frame(g, self.level, self.step_idx) for g in inter], rec.get("status_change")

    def next_action(self) -> Optional[Action]:
        """The action the log takes next (None at the end). Lets a pipeline follow a human log step by step."""
        while self.pos < len(self.records) and self.records[self.pos]["kind"] == "reset":
            if self.frame is None:
                break
            self.pos += 1
        if self.pos >= len(self.records):
            return None
        return Action.from_json(self.records[self.pos]["action"])

    def reset(self, game_id: Optional[str] = None) -> Frame:
        while self.pos < len(self.records) and self.records[self.pos]["kind"] != "reset":
            self.pos += 1
        if self.pos >= len(self.records):
            raise EOFError("replay log has no further reset")
        rec = self.records[self.pos]; self.pos += 1
        self.frame, _, _ = self._serve(rec, Action.reset())
        return self.frame

    def step(self, action: Action) -> RawTransition:
        before, st = self.frame, self.status()
        if self.pos >= len(self.records):
            return RawTransition(before, action, before, st, error="end of replay log", step_idx=self.step_idx, level=self.level)
        rec = self.records[self.pos]
        recorded = Action.from_json(rec["action"])
        if rec["kind"] == "reset":
            if action.type != "RESET" and not self.lenient:
                return RawTransition(before, action, before, st, error="replay expects RESET", step_idx=self.step_idx, level=self.level)
            frame = self.reset(); self.actions_used += 1
            return RawTransition(before, Action.reset(), frame, self.status(), None, [], self.available_actions(), step_idx=self.step_idx, level=self.level)
        if recorded != action and not self.lenient:
            return RawTransition(before, action, before, st, error=f"replay expects {recorded.label()}, got {action.label()}", step_idx=self.step_idx, level=self.level)
        self.pos += 1; self.step_idx += 1; self.actions_used += 1
        level_before = self.level
        after, inters, change = self._serve(rec, recorded)
        self.frame = after
        return RawTransition(before, recorded, after, self.status(), change, inters, self.available_actions(), step_idx=self.step_idx, level=level_before)

    def available_actions(self) -> list[Action]:
        out = [Action.button(i) for i in self._avail if i in BUTTON_IDS]
        if CLICK_ID in self._avail:
            out.append(Action("CLICK"))
        return out

    def status(self) -> EnvStatus:
        return EnvStatus(self.state, self.level, self.levels_total, self.actions_used, None, 0.0, None)

    def exhausted(self) -> bool:
        return self.pos >= len(self.records)
