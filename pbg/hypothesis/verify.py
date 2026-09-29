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
    approx: float = 0.0        # mean closeness over verifiable transitions (1 exact, IoU of changed regions otherwise, 0 unknown)
    changed: int = 0           # transitions where the real board changed (outside the ignore boxes)
    change_correct: int = 0    # ... of which the model predicted exactly

    @property
    def change_accuracy(self) -> float:
        """Accuracy on the transitions that changed the board: a model that predicts "nothing happens" scores 0 here
        (it is exact on every no-op and useless for planning — the trap that cost tn36 its level, docs/029)."""
        return self.change_correct / self.changed if self.changed else 0.0

    def rank_key(self) -> tuple:
        return (self.change_accuracy, self.accuracy, self.coverage)

    def usable(self, min_acc: float = 0.8, min_n: int = 3, min_change: float = 0.5) -> bool:
        return self.n >= min_n and self.accuracy >= min_acc and self.changed >= 1 and self.change_accuracy >= min_change

    def approximate(self, min_approx: float = 0.5, min_cov: float = 0.6, min_n: int = 2) -> bool:
        """Good enough to act on one step at a time (closed loop), not to trust a long plan."""
        return self.n >= min_n and self.coverage >= min_cov and self.approx >= min_approx


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


def closeness(pred: Scene, before_grid, after_grid, boxes: list) -> float:
    """How close a wrong prediction is: IoU of the predicted and the real changed-pixel sets, times the colour agreement
    on the real changed pixels (moving the right thing roughly right scores well; changing nothing scores 0)."""
    pg = pred.render(); bg = np.asarray(before_grid); ag = np.asarray(after_grid)
    if pg.shape != ag.shape:
        return 0.0
    pc = pg != bg; rc = ag != bg
    for (r0, c0, r1, c1) in boxes:
        pc[r0:r1, c0:c1] = False; rc[r0:r1, c0:c1] = False
    if not rc.any():
        return 1.0 if not pc.any() else 0.0
    union = (pc | rc).sum()
    iou = float((pc & rc).sum()) / float(union) if union else 0.0
    # spatial tolerance: a change predicted within 3 cells of the real one counts half
    if iou < 0.5 and pc.any():
        ry, rx = np.nonzero(rc); py, px = np.nonzero(pc)
        d = abs(float(ry.mean()) - float(py.mean())) + abs(float(rx.mean()) - float(px.mean()))
        iou = max(iou, 0.5 * max(0.0, 1.0 - d / 12.0))
    colour = float((pg[rc] == ag[rc]).mean())
    return max(0.0, min(1.0, 0.7 * iou + 0.3 * colour))


def verify(model, log: list[Transition], hyp_ignore=None, *, tol: int = 0, max_examples: int = 4) -> Verdict:
    usable = [t for t in log if t.action.type != "RESET" and t.before_frame is not None and t.after_frame is not None and not t.status_change]
    if not usable:
        return Verdict(0.0, 0.0, 0, 0, 0)
    correct = unknown = 0; ex: list[str] = []; viol: list[str] = []; close = 0.0
    changed = change_correct = 0
    for t in usable:
        boxes0 = _boxes(t.before, hyp_ignore) if hyp_ignore else [tuple(r.bbox) for r in t.before.regions if r.kind_hint == "ui_strip"]
        real = np.asarray(t.before_frame.grid) != np.asarray(t.after_frame.grid)
        for (r0, c0, r1, c1) in boxes0:
            real[r0:r1, c0:c1] = False
        did_change = bool(real.any())
        if did_change:
            changed += 1
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
            correct += 1; close += 1.0
            if did_change:
                change_correct += 1
            continue
        close += closeness(pred, t.before_frame.grid, t.after_frame.grid, boxes)
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
    return Verdict(correct / n, (n - unknown) / n, n, correct, unknown, ex, viol, approx=close / n, changed=changed, change_correct=change_correct)
