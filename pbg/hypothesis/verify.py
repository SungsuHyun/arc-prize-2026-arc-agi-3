"""hypothesis/verify.py — pixel-level verification of a game hypothesis on a level's log. No object-size leniency, no
role-based exceptions: the predicted next grid must equal the real next grid outside the hypothesis' own ignore boxes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.types import Scene, Transition
from .contract import frame_text


@dataclass
class Verdict:
    accuracy: float            # correct / verifiable transitions
    coverage: float            # predicted (not None) / verifiable
    n: int
    correct: int
    unknown: int
    counterexamples: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def usable(self, min_acc: float = 0.8, min_n: int = 3) -> bool:
        return self.n >= min_n and self.accuracy >= min_acc


def _boxes(scene: Scene, hyp_ignore) -> list:
    boxes = [tuple(r.bbox) for r in scene.regions if r.kind_hint == "ui_strip"]
    try:
        boxes += [tuple(int(v) for v in b) for b in (hyp_ignore(scene) or [])]
    except Exception:
        pass
    return boxes


def pixel_equal(pred: Scene, obs_grid: np.ndarray, boxes: list, tol: int = 0) -> tuple[bool, int, Optional[tuple]]:
    pg = pred.render(); og = np.asarray(obs_grid)
    if pg.shape != og.shape:
        return False, 10 ** 6, None
    mask = pg != og
    for (r0, c0, r1, c1) in boxes:
        mask[r0:r1, c0:c1] = False
    n = int(mask.sum())
    if n <= tol:
        return True, n, None
    ys, xs = np.nonzero(mask)
    return False, n, (max(0, int(ys.min()) - 1), max(0, int(xs.min()) - 1), min(og.shape[0], int(ys.max()) + 2), min(og.shape[1], int(xs.max()) + 2))


def verify(model, log: list[Transition], hyp_ignore=None, *, tol: int = 0, max_examples: int = 4) -> Verdict:
    usable = [t for t in log if t.action.type != "RESET" and t.before_frame is not None and t.after_frame is not None and not t.status_change]
    if not usable:
        return Verdict(0.0, 0.0, 0, 0, 0)
    correct = unknown = 0; ex: list[str] = []; viol: list[str] = []
    for t in usable:
        try:
            pred = model.predict(t.before, t.action)
        except Exception as e:
            pred = None
            if len(ex) < max_examples:
                ex.append(f"[{t.id}] {t.action.label()}: the model raised {type(e).__name__}: {str(e)[:120]}")
        if pred is None:
            unknown += 1; continue
        boxes = _boxes(t.before, hyp_ignore) if hyp_ignore else [tuple(r.bbox) for r in t.before.regions if r.kind_hint == "ui_strip"]
        ok, n, win = pixel_equal(pred, t.after_frame.grid, boxes, tol)
        if ok:
            correct += 1; continue
        viol.append(t.id)
        if len(ex) < max_examples and win is not None:
            r0, c0, r1, c1 = win
            if (r1 - r0) <= 24 and (c1 - c0) <= 40:
                ex.append(f"[{t.id}] {t.action.label()}: {n} px wrong in rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1}\n    predicted:\n"
                          + "\n".join("      " + l for l in frame_text(pred.render(), (r0, r1), (c0, c1)).splitlines())
                          + "\n    real:\n" + "\n".join("      " + l for l in frame_text(np.asarray(t.after_frame.grid), (r0, r1), (c0, c1)).splitlines()))
            else:
                ex.append(f"[{t.id}] {t.action.label()}: {n} px wrong over rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1} (window too large to print)")
    n = len(usable)
    return Verdict(correct / n, (n - unknown) / n, n, correct, unknown, ex, viol)
