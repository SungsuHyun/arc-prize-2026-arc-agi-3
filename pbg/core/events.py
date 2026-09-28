"""core/events.py — state-transition event log (events.jsonl; spec §12, input of harness/attribution)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class EventLog:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, frm: str, to: str, reason: str, *, transition_id: Optional[str] = None, budget_used: int = 0, **extra) -> dict:
        if not reason:
            raise ValueError("every state transition needs a reason (spec §12 acceptance)")
        ev = {"ts": time.time(), "from": frm, "to": to, "reason": reason, "transition_id": transition_id, "budget_used": budget_used, **extra}
        self.events.append(ev)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev

    @staticmethod
    def read(path: Path) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
