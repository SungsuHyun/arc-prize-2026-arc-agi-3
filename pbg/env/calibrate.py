"""env/calibrate.py — click calibration (spec §5). For a pixel-space environment two corner clicks would fix scale and
offset; the arcengine API already takes cell coordinates, so calibration is a contract check plus a runtime monitor:
`ClickMonitor` records which object responded to each click and flags misalignment when the responder is never the
object under the cursor (spec §15 'click coordinate misalignment')."""
from __future__ import annotations

from typing import Optional

from ..core.types import Action, Scene, Transition


def object_under(scene: Scene, r: int, c: int) -> Optional[int]:
    for o in scene.objects:
        r0, c0, r1, c1 = o.bbox
        if r0 <= r < r1 and c0 <= c < c1 and o.mask[r - r0, c - c0]:
            return o.id
    return None


class ClickMonitor:
    def __init__(self, window: int = 6):
        self.window = window
        self.hits: list[bool] = []

    def observe(self, t: Transition) -> None:
        if t.action.type != "CLICK" or t.diff.is_noop:
            return
        under = object_under(t.before, t.action.row, t.action.col)
        responders = {i for i, _ in t.diff.moved} | {i for i, _, _ in t.diff.recolored} | {o.id for o in t.diff.disappeared} | {i for i, _, _ in t.diff.reshaped}
        self.hits.append(under is not None and under in responders)
        self.hits = self.hits[-self.window:]

    def misaligned(self) -> bool:
        return len(self.hits) >= self.window and not any(self.hits)
