"""probe/hypotheses.py — look first, then test: role and relation hypotheses read off the FIRST frame, and the fewest
clicks that confirm or refute them.

Roles are guessed from appearance (a set of identical small squares = buttons; a solid bar attached to a wall or border =
a bar/tank; a small multi-colour piece = a mover; a small mark whose colour a mover carries = a marker; a thin long line
= a wall). Relations follow from geometry (a button controls the bars nearest to it; a mover rides the bar it stands on;
a marker belongs to the mover of its colour). The probe then clicks ONE button candidate at a time: a change in the
predicted bars confirms the hypothesis (and one observation is enough to instantiate its effect rule); no change or a
change elsewhere refutes it and the roles are revised. Objects hypothesised as non-interactive are not clicked at all
unless every trigger hypothesis fails."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..core.types import Action, Object, Scene, Transition
from .clickmap import object_key
from .semantics import core_diff


@dataclass
class RoleHyp:
    role: str
    p: float
    why: str
    confirmed: Optional[bool] = None


def _solid(o: Object) -> bool:
    return o.mask is not None and min(o.height, o.width) >= 2 and float(o.mask.mean()) >= 0.75


def _gap(a: tuple, b: tuple) -> int:
    """Manhattan gap between two bboxes (0 when they touch or overlap)."""
    dr = max(0, max(a[0], b[0]) - min(a[2], b[2])); dc = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return dr + dc


class BoardHypotheses:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.roles: dict[int, RoleHyp] = {}
        self.controls: dict[int, list[int]] = {}      # button -> bars it probably moves (nearest first)
        self.rides: dict[int, int] = {}               # mover -> bar it stands on
        self.marker_of: dict[int, int] = {}           # marker -> mover with the same colour
        self.tested: dict[int, str] = {}              # button -> "confirmed" | "refuted"
        self.controls_other: dict[int, bool] = {}     # button -> its test click changed objects other than itself
        # ids can be re-assigned when the level is re-parsed after a test click (a panel turned out to be a bar):
        # objects are recognised by colour + bbox, which the same frame keeps
        self._key2id = {(o.color, tuple(o.bbox)): o.id for o in scene.objects}
        self._infer()

    def _orig(self, o: Object) -> int:
        return self._key2id.get((o.color, tuple(o.bbox)), o.id)

    # ── appearance -> roles ──
    def _infer(self) -> None:
        s = self.scene
        strips = {r.id for r in s.regions if r.kind_hint == "ui_strip"}
        objs = [o for o in s.objects if o.region not in strips and o.area >= 2]
        shape_count = Counter((o.color, o.shape_sig) for o in objs)
        multi_colors = {c for o in objs if len(o.colors) > 1 for c in o.colors}
        small_single = [o for o in objs if len(o.colors) == 1 and o.area <= 12]
        for o in objs:
            single = len(o.colors) == 1
            if single and _solid(o) and o.area <= 36 and shape_count[(o.color, o.shape_sig)] >= 2 and abs(o.height - o.width) <= 2:
                self.roles[o.id] = RoleHyp("button", 0.8, f"{shape_count[(o.color, o.shape_sig)]} identical small squares of colour {o.color}")
            elif not single and o.area <= 60:
                self.roles[o.id] = RoleHyp("mover", 0.7, "small multi-colour piece")
            elif single and o.area <= 12 and (o.color in multi_colors or any(q.color == o.color and q.id != o.id and (q.height, q.width) != (o.height, o.width) for q in small_single)):
                self.roles[o.id] = RoleHyp("marker", 0.7, f"small mark of colour {o.color} that a piece also carries")
            elif single and min(o.height, o.width) <= 4 and max(o.height, o.width) >= 12 and not _solid(o) is False:
                self.roles[o.id] = RoleHyp("wall", 0.6, "thin long line")
            elif single and _solid(o) and o.area >= 12:
                r0, c0, r1, c1 = o.bbox
                border = r0 == 0 or c0 == 0 or r1 == s.grid_shape[0] or c1 == s.grid_shape[1]
                self.roles[o.id] = RoleHyp("bar", 0.6 if border else 0.4, "solid block" + (" attached to the border" if border else ""))
            elif single and _solid(o) and o.area <= 36:
                self.roles[o.id] = RoleHyp("button", 0.4, "lone small square")
            else:
                self.roles[o.id] = RoleHyp("decoration", 0.3, "no pattern")
        # walls that are thin lines vs bars: a bar next to a wall keeps 'bar'
        by = {o.id: o for o in objs}
        bars = [i for i, h in self.roles.items() if h.role == "bar"]
        for i, h in self.roles.items():
            o = by[i]
            if h.role == "button":
                near = sorted(bars, key=lambda b: _gap(o.bbox, by[b].bbox))
                self.controls[i] = [b for b in near if _gap(o.bbox, by[b].bbox) <= 12][:2]
            elif h.role == "mover":
                touching = [b for b in bars if _gap(o.bbox, by[b].bbox) == 0]
                if touching:
                    self.rides[i] = touching[0]
            elif h.role == "marker":
                mv = [m for m, hm in self.roles.items() if hm.role == "mover" and o.color in by[m].colors]
                if mv:
                    self.marker_of[i] = mv[0]

    # ── what to test ──
    def test_actions(self) -> list[Action]:
        """One click per button CLASS first (23 identical pieces are one hypothesis, not 23); the other members of a class
        are tested only after its representative proved to control OTHER objects (arrow buttons each move their own bar),
        never when clicking it changed the clicked object itself (pieces, pattern cells)."""
        by = {o.id: o for o in self.scene.objects}
        cands = [(h.p + (0.1 if self.controls.get(i) else 0.0), i) for i, h in self.roles.items() if h.role == "button" and i not in self.tested and i in by]
        cands.sort(reverse=True)
        out = []; seen_class: set = set()
        for _, i in cands:
            cls = (by[i].color, by[i].shape_sig)
            if cls in seen_class:
                continue
            tested_same = [j for j in self.tested if j in by and (by[j].color, by[j].shape_sig) == cls]
            if tested_same and not any(self.controls_other.get(j) for j in tested_same):
                continue          # the class representative changed itself or nothing: no per-object testing
            seen_class.add(cls) if not tested_same else None
            out.append(Action.click(*by[i].center))
        return out

    def fallback_actions(self) -> list[Action]:
        """Only when no trigger hypothesis held: the other object classes, one each, decorations last."""
        by = {o.id: o for o in self.scene.objects}
        order = {"marker": 0, "mover": 1, "bar": 2, "wall": 3, "decoration": 4}
        seen: set[str] = set(); out = []
        for i, h in sorted(self.roles.items(), key=lambda x: order.get(x[1].role, 5)):
            o = by.get(i)
            if o is None or h.role == "button" or object_key(o) in seen:
                continue
            seen.add(object_key(o)); out.append(Action.click(*o.center))
        return out

    def non_interactive_classes(self) -> set[str]:
        by = {o.id: o for o in self.scene.objects}
        return {object_key(by[i]) for i, h in self.roles.items() if h.role in ("bar", "wall", "marker", "mover", "decoration") and i in by}

    # ── evidence ──
    def observe(self, t: Transition) -> Optional[str]:
        """Update the hypotheses with one probe transition; returns a one-line verdict for the event log."""
        a = t.action
        if a.type != "CLICK":
            return None
        by = {o.id: o for o in t.before.objects}
        hit = next((o for o in t.before.objects if o.bbox[0] <= a.row < o.bbox[2] and o.bbox[1] <= a.col < o.bbox[3] and o.mask[a.row - o.bbox[0], a.col - o.bbox[1]]), None)
        if hit is None:
            return None
        c = core_diff(t)
        changed_ids = {i for i, _ in c["moved"]} | {x[0] for x in c["recolored"] + c["reshaped"]} | {o.id for o in c["disappeared"]}
        changed = {self._orig(by[i]) for i in changed_ids if i in by}
        hid = self._orig(hit)
        h = self.roles.get(hid)
        if h is None:
            return None
        if h.role == "button":
            if changed:
                expected = set(self.controls.get(hid, [])) | {m for m, b in self.rides.items() if b in self.controls.get(hid, [])}
                agree = bool(expected & changed)
                self.tested[hid] = "confirmed"; h.confirmed = True; h.p = 1.0
                self.controls_other[hid] = bool(changed - {hid})
                if not agree and expected:
                    # it is a trigger, but of other objects than the geometry suggested: keep the observed targets
                    self.controls[hid] = [i for i in changed if self.roles.get(i, RoleHyp("", 0, "")).role == "bar"]
                orig_by = {o.id: o for o in self.scene.objects}
                what = ", ".join(f"{orig_by[i].color}@{orig_by[i].bbox[0]},{orig_by[i].bbox[1]}" for i in sorted(changed) if i in orig_by)[:80]
                return f"button {hid} confirmed ({'as predicted' if agree else 'other targets'}): changed {what}"
            self.tested[hid] = "refuted"; h.confirmed = False; h.p *= 0.3
            return f"button {hid} refuted: no change (blocked or not a trigger)"
        if changed:
            # a supposedly passive object reacted: it is a trigger after all
            self.roles[hid] = RoleHyp("button", 0.9, f"reacted when clicked (was {h.role})", confirmed=True)
            self.tested[hid] = "confirmed"
            return f"{h.role} {hid} reacted -> promoted to button"
        return f"{h.role} {hid} inert as expected"

    def confirmed_triggers(self) -> set[tuple]:
        by = {o.id: o for o in self.scene.objects}
        return {tuple(by[i].bbox) for i, v in self.tested.items() if v == "confirmed" and i in by}

    def summary(self) -> str:
        cnt = Counter(h.role for h in self.roles.values())
        rel = f"controls={len([b for b in self.controls.values() if b])} rides={len(self.rides)} markers={len(self.marker_of)}"
        return f"roles {dict(cnt)}; {rel}; tested {dict(Counter(self.tested.values()))}"
