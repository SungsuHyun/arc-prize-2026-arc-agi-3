"""perception/ — pixels -> Scene (regions + tracked objects), Diff, summaries. Deterministic, no LLM (spec §6)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..core.types import Frame, Object, Scene, Transition
from .diff import compute_diff
from .regions import find_regions, stabilize_regions
from .render import grid_png, render_scene
from .segment import fuse, rect_composites, segment_objects
from .summarize import scene_text, summarize
from .track import Tracker

DEFAULT_CONFIG = {"region_min_area_ratio": 0.05, "connectivity": 4, "merge_confirm_count": 3, "track_cost_max": 20.0,
                  "noop_pixel_threshold": 0, "ui_strip_max_thickness": 3}


def load_perception_config(path: Optional[Path] = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    p = path or Path(__file__).resolve().parents[1] / "configs" / "perception.yaml"
    if p.exists():
        cfg.update(yaml.safe_load(p.read_text()) or {})
    return cfg


class Perception:
    """Stateful per game: keeps the tracker, the merge candidates and the confirmed merge log."""

    def __init__(self, config: Optional[dict] = None):
        self.cfg = {**load_perception_config(), **(config or {})}
        self.tracker = Tracker(float(self.cfg["track_cost_max"]))
        self._pair_moves: dict[tuple[int, int], int] = {}   # (id_a, id_b) -> co-movement count
        self._merged: list[tuple[int, int]] = []             # confirmed merges (id_a, id_b)
        self._groups: list[set[int]] = []
        self._solo: set[int] = set()                         # part ids seen moving on their own: never part of a composite
        self._dynamic: set[tuple] = set()                    # (colour, axis, lo, hi): panels seen changing extent -> objects, not regions
        self.reparse_needed = False                          # a panel was just found dynamic: callers re-parse the level's transitions
        self.timings: list[float] = []
        self.merge_confirmed = False                         # a merge was confirmed OR a composite dissolved: callers refit past transitions

    # ── main entry points ──
    def parse(self, frame: Frame, prev: Optional[Scene]) -> Scene:
        t0 = time.perf_counter()
        grid = np.asarray(frame.grid, dtype=np.int8)
        regions, global_bg, meta = find_regions(grid, min_area_ratio=float(self.cfg["region_min_area_ratio"]),
                                               ui_strip_max_thickness=int(self.cfg["ui_strip_max_thickness"]))
        raw_regions = [(r.bg_color, tuple(r.bbox)) for r in regions if r.id != "R0" and r.kind_hint != "ui_strip"]
        regions = [r for r in regions if not self._is_dynamic(r)]
        if prev is not None:
            regions = stabilize_regions(regions, [r for r in prev.regions if not self._is_dynamic(r)])
        min_area = max(4, int(float(self.cfg["region_min_area_ratio"]) * grid.shape[0] * grid.shape[1]))
        objs, adj = segment_objects(grid, regions, global_bg, meta["region_masks"], connectivity=int(self.cfg["connectivity"]), min_region_area=min_area)
        prev_parts = [p for o in prev.objects for p in (o.parts or [o])] if prev else None   # fused objects are tracked by their parts
        assign = self.tracker.match(prev_parts, objs)
        tracked: list[Object] = []
        for i, o in enumerate(objs):
            o.id = assign[i]; tracked.append(o)
        # merge candidates: adjacent different-colour components, by tracking id
        cand = {(min(assign[a], assign[b]), max(assign[a], assign[b])) for a, b in adj}
        # rectangle composites (target pattern, canvas, framed button): fused right away, dissolved if a part moves alone
        grouped = {i for g in self._groups for i in g}
        tracked = self._fuse_composites(tracked, [(assign[a], assign[b]) for a, b in adj], grid, regions, grouped)
        # confirmed merges: fuse groups whose members are all present
        if self._groups:
            by_id = {o.id: o for o in tracked}
            fused: list[Object] = []; consumed: set[int] = set()
            for grp in self._groups:
                members = [by_id[i] for i in grp if i in by_id]
                if len(members) >= 2:
                    fused.append(fuse(members, grid, regions)); consumed |= {m.id for m in members}
            tracked = [o for o in tracked if o.id not in consumed] + fused
        tracked.sort(key=lambda o: o.id)
        scene = Scene(frame.hash, tuple(grid.shape), regions, tracked, {"_global_bg": global_bg, "_merge_candidates": sorted(cand), "_raw_regions": raw_regions})
        self.timings.append(time.perf_counter() - t0)
        return scene

    def diff(self, before: Scene, after: Scene, before_grid: Optional[np.ndarray] = None, after_grid: Optional[np.ndarray] = None):
        d = compute_diff(before, after, before_grid=before_grid, after_grid=after_grid, noop_pixel_threshold=int(self.cfg["noop_pixel_threshold"]))
        self.merge_confirmed = self._observe_comovement(before, d)
        return d

    def _observe_comovement(self, before: Scene, d) -> bool:
        """Merge confirmation (spec §6 step 3): adjacent pairs that moved by the same displacement 3 times become one object.
        Returns True when a new merge was confirmed."""
        if not d.moved:
            return False
        reshaped = {x[0] for x in d.reshaped}
        moved = {i: v for i, v in d.moved if i not in reshaped}     # a bar whose end moved with the mover was resized, not carried
        new = False
        for a, b in before.aux.get("_merge_candidates", []):
            if a in moved and b in moved and moved[a] == moved[b]:
                self._pair_moves[(a, b)] = self._pair_moves.get((a, b), 0) + 1
                if self._pair_moves[(a, b)] >= int(self.cfg["merge_confirm_count"]) and (a, b) not in self._merged:
                    self._merged.append((a, b)); self._add_group(a, b); new = True
        return new

    def _is_dynamic(self, r) -> bool:
        if r.id == "R0" or r.kind_hint == "ui_strip" or not self._dynamic:
            return False
        r0, c0, r1, c1 = r.bbox
        return (r.bg_color, "row", r0, r1) in self._dynamic or (r.bg_color, "col", c0, c1) in self._dynamic

    def _detect_dynamic(self, before: Scene, after: Scene) -> set[tuple]:
        """Raw background panels whose extent changed between two consecutive frames (same colour, same fixed axis) are
        resizable bars: they carry state and must be objects."""
        new: set[tuple] = set()
        rb = before.aux.get("_raw_regions", []); ra = after.aux.get("_raw_regions", [])
        for color, (r0, c0, r1, c1) in rb:
            same = [(s0, d0, s1, d1) for col, (s0, d0, s1, d1) in ra if col == color and not (s1 <= r0 or r1 <= s0 or d1 <= c0 or c1 <= d0)]
            if not same:
                continue
            for (s0, d0, s1, d1) in same:
                if (s0, s1) == (r0, r1) and (d0, d1) != (c0, c1):
                    new.add((color, "row", r0, r1))
                elif (d0, d1) == (c0, c1) and (s0, s1) != (r0, r1):
                    new.add((color, "col", c0, c1))
        return new - self._dynamic

    def reparse_level(self, transitions: list[Transition]):
        """Re-parse a level's transitions from their frames (after a panel turned out to be dynamic) with tracking run
        again in order. Returns the final scene (the caller's current scene)."""
        scene = None
        for t in transitions:
            if t.before_frame is None or t.action.type == "RESET":
                scene = self.parse(t.after_frame, None) if t.after_frame is not None else scene
                if scene is not None:
                    t.after = scene
                continue
            before = scene if (scene is not None and scene.frame_hash == t.before_frame.hash) else self.parse(t.before_frame, None)
            after = self.parse(t.after_frame, before)
            d = compute_diff(before, after, before_grid=t.before_frame.grid, after_grid=t.after_frame.grid, noop_pixel_threshold=int(self.cfg["noop_pixel_threshold"]))
            t.before, t.after, t.diff = before, after, d
            scene = after
        return scene

    def _fuse_composites(self, objs: list[Object], adj, grid, regions, extra_excluded=()) -> list[Object]:
        comps = rect_composites(objs, adj, regions, self._solo | set(extra_excluded), min_side=int(self.cfg.get("composite_min_side", 2)),
                                min_area=int(self.cfg.get("composite_min_area", 6)), min_share=float(self.cfg.get("composite_min_share", 0.2)))
        if not comps:
            return objs
        consumed = {m.id for c in comps for m in c}
        fused = []
        for c in comps:
            f = fuse(c, grid, regions); f.composite = True; fused.append(f)
        return sorted([o for o in objs if o.id not in consumed] + fused, key=lambda o: o.id)

    def _dissolve(self, scene: Scene) -> Scene:
        """Break composites that contain a part seen moving on its own; the remaining parts may re-form a composite."""
        if not self._solo or not any(o.composite and any(p.id in self._solo for p in o.parts) for o in scene.objects):
            return scene
        objs: list[Object] = []; freed: list[Object] = []
        for o in scene.objects:
            if o.composite and any(p.id in self._solo for p in o.parts):
                freed.extend(o.parts)
            else:
                objs.append(o)
        objs = self._fuse_composites(objs + freed, None, None, scene.regions)
        return scene.copy(objects=sorted(objs, key=lambda o: o.id))

    def _solo_movers(self, before: Scene, after: Scene) -> set[int]:
        """Parts of a composite that moved while another part of the same composite stayed put."""
        def parts_map(scene):
            return {p.id: p for o in scene.objects for p in (o.parts or [o])}
        pb, pa = parts_map(before), parts_map(after)
        new: set[int] = set()
        for scene_c, other in ((before, pa), (after, pb)):
            for o in scene_c.objects:
                if not o.composite:
                    continue
                disp = {}; changed = set()
                for p in o.parts:
                    q = other.get(p.id)
                    if q is None:
                        continue
                    if q.shape_sig == p.shape_sig and q.area == p.area:
                        disp[p.id] = (q.bbox[0] - p.bbox[0], q.bbox[1] - p.bbox[1])
                    else:
                        changed.add(p.id)          # resized / redrawn part (a bar next to a button is not a pattern half)
                if len(disp) >= 2 and any(v == (0, 0) for v in disp.values()):
                    new |= {i for i, v in disp.items() if v != (0, 0)}
                if changed and any(v == (0, 0) for v in disp.values()):
                    new |= changed
        return new - self._solo

    def apply_groups(self, scene: Scene) -> Scene:
        """Fuse confirmed groups inside an already-parsed scene (used to refit earlier scenes after a confirmation)."""
        if not self._groups:
            return scene
        by_id = {o.id: o for o in scene.objects}
        fused: list[Object] = []; consumed: set[int] = set()
        for grp in self._groups:
            members = [by_id[i] for i in grp if i in by_id]
            if len(members) >= 2:
                fused.append(fuse(members, None, scene.regions)); consumed |= {m.id for m in members}
        if not consumed:
            return scene
        objs = sorted([o for o in scene.objects if o.id not in consumed] + fused, key=lambda o: o.id)
        return scene.copy(objects=objs)

    def refit(self, transitions: list[Transition]) -> int:
        """Re-fuse confirmed groups in past transitions (spec §15: reparse past scenes after a merge). Returns #changed."""
        n = 0
        for t in transitions:
            b, a = self._dissolve(self.apply_groups(t.before)), self._dissolve(self.apply_groups(t.after))
            if len(b.objects) != len(t.before.objects) or len(a.objects) != len(t.after.objects):
                t.before, t.after = b, a
                t.diff = compute_diff(b, a, before_grid=None if t.before_frame is None else t.before_frame.grid,
                                      after_grid=None if t.after_frame is None else t.after_frame.grid, noop_pixel_threshold=int(self.cfg["noop_pixel_threshold"]))
                n += 1
        return n

    def _add_group(self, a: int, b: int) -> None:
        ga = next((g for g in self._groups if a in g), None); gb = next((g for g in self._groups if b in g), None)
        if ga and gb and ga is not gb:
            ga |= gb; self._groups.remove(gb)
        elif ga:
            ga.add(b)
        elif gb:
            gb.add(a)
        else:
            self._groups.append({a, b})

    def summarize(self, t: Transition, max_len: int = 200) -> str:
        return summarize(t, max_len)

    def render(self, scene: Scene, scale: int = 8) -> bytes:
        return render_scene(scene, scale)

    def merge_log(self) -> list[tuple[int, int]]:
        return list(self._merged)

    def transition(self, before: Scene, action, after: Scene, *, before_frame: Optional[Frame] = None, after_frame: Optional[Frame] = None,
                   status_change=None, intermediate=(), tid: str = "", level: int = 1, step_idx: int = 0) -> Transition:
        """Build the Transition (diff + merge bookkeeping). Callers must continue from `t.after`, not from the scene they
        passed in: when a merge is confirmed here both scenes are re-fused and `t.after` is the canonical current scene."""
        bg = None if before_frame is None else before_frame.grid; ag = None if after_frame is None else after_frame.grid
        d = self.diff(before, after, bg, ag)
        if status_change is None:
            dyn = self._detect_dynamic(before, after)
            if dyn:
                self._dynamic |= dyn; self.reparse_needed = True
        solo = self._solo_movers(before, after)
        if solo:
            # a composite's part moved alone (an agent next to a wall, a marker under a button): it was never a pattern
            self._solo |= solo
            before, after = self._dissolve(before), self._dissolve(after)
            d = compute_diff(before, after, before_grid=bg, after_grid=ag, noop_pixel_threshold=int(self.cfg["noop_pixel_threshold"]))
            self.merge_confirmed = True
        if self.merge_confirmed:
            # a merge was just confirmed: fuse both scenes of this transition so the diff sees one object
            before, after = self.apply_groups(before), self.apply_groups(after)
            d = compute_diff(before, after, before_grid=bg, after_grid=ag, noop_pixel_threshold=int(self.cfg["noop_pixel_threshold"]))
        return Transition(before, action, after, d, status_change, list(intermediate), tid, level, step_idx, before_frame, after_frame)


__all__ = ["Perception", "load_perception_config", "summarize", "scene_text", "grid_png", "render_scene", "compute_diff"]
