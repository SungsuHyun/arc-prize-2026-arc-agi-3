"""Program-made plan candidates: short action sequences that make a live win predicate true on the learned effect model.

  match:   Equal predicate     -> click every editable mark that differs from the template (toggle rule known)
  press:   Inside predicate    -> BFS over button presses on the solved slot permutations until the blocks sit in the frames
  marker:  Inside predicate    -> move the selected marker so that the coupled piece lands in the outline
The model only chooses among them; every action is still verified one by one when executed.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .candidates import Candidate
from .entities import Scene
from .predict import Evidence
from . import goals2


def make_plans(game, ev: Evidence, preds: list, *, max_depth: int = 24, max_states: int = 20000) -> list[Candidate]:
    out: list[Candidate] = []
    sc = ev.scene
    for eid, p in preds:
        try:
            if isinstance(p, goals2.Equal):
                c = _plan_match(ev, sc, p, eid)
            elif isinstance(p, goals2.Inside):
                c = _plan_press(ev, sc, p, eid, max_depth, max_states) or _plan_marker(ev, sc, p, eid)
            elif isinstance(p, (goals2.CollectAll, goals2.Reach, goals2.CollectReach)):
                c = _plan_move(ev, sc, p, eid)
            else:
                c = None
        except Exception as e:  # a planner bug must never stop play
            c = Candidate(f"plan:{eid}", "info", [], f"planner error {type(e).__name__}", priority=999)
        if c is not None:
            out.append(c)
    return out


def _plan_match(ev: Evidence, sc: Scene, p: goals2.Equal, eid: str) -> Optional[Candidate]:
    a, b = goals2._region_by_spec(sc, p.r1), goals2._region_by_spec(sc, p.r2)
    if a is None or b is None:
        return None
    ma, mb = goals2._mask(sc, a), goals2._mask(sc, b)
    if ma == mb:
        return None
    ra, rb = sc.region_named(a), sc.region_named(b)
    diff = {k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)}
    # objects of the editable area that cover differing cells
    targets = []
    for o in sc.objs:
        if o.region != a or o.hud or not o.cells:
            continue
        if any((r - ra.bbox[0], c - ra.bbox[1]) in diff for r, c in o.cells):
            targets.append(o)
    if not targets:
        return None
    acts = []
    for o in targets[:24]:
        r, c = o.center
        if sc.frame.grid[r][c] != o.color:
            r, c = o.cells[0]
        acts.append({"action": "MOUSE", "row": r, "col": c})
    return Candidate(f"plan:match({eid})", "plan", acts, f"click the {len(targets)} marks in P{a} that differ from the template P{b} (toggle each once) -> {eid} should hold",
                     priority=5, pred_kind="plan")


def _plan_press(ev: Evidence, sc: Scene, p: goals2.Inside, eid: str, max_depth: int, max_states: int) -> Optional[Candidate]:
    allf = goals2.find_frames(sc)
    pairs = [(c, c) for c in sorted({f.color for f in allf})] if p.color == p.frame else [(p.color, p.frame)]
    frames = [f for f in allf if any(f.color == fc for _, fc in pairs)]
    dot_ids = {i for f in frames for i in f.dots}
    blocks = [o for o in sc.objs if any(o.color == bc for bc, _ in pairs) and not o.hud and o.size >= 4 and o.id not in dot_ids]
    if not frames or not blocks:
        return None
    buttons = [(key, ev.cs.sigma(key)) for key in ev.cs.presses]
    buttons = [(key, sg) for key, sg in buttons if any(q != q0 for q0, q in sg.items())]
    if not buttons:
        # nothing learned on this level yet: press a button that moved things on an earlier level (same colour), if any
        known_cols = {key[0] for key in ev.cs_all.presses}
        cand = [o for o in sc.objs if o.color in known_cols and not o.hud and o.size >= 8]
        if cand:
            b = min(cand, key=lambda o: ev.cs.succ_n.get(o.key, 0))
            return Candidate(f"plan:learn({eid})", "plan", [{"action": "MOUSE", "row": b.center[0], "col": b.center[1]}],
                             f"press button #{b.id} (colour {b.color} buttons moved blocks on an earlier level) to learn its track on this level", priority=7, pred_kind="plan")
        return None
    button_objs = {}
    for key, _ in buttons:
        o = next((x for x in sc.objs if x.key == key), None)
        if o is not None:
            button_objs[key] = o
    buttons = [(key, sg) for key, sg in buttons if key in button_objs]
    if not buttons:
        return None
    movable = set()
    for _, sg in buttons:
        movable |= {q0 for q0, q in sg.items() if q != q0} | {q for q0, q in sg.items() if q != q0}
    if not any(o.bbox in movable for o in blocks):
        return None
    static = {o.bbox: (o.color, o.shape) for o in sc.objs if not o.hud and o.bbox not in movable}
    start = frozenset((o.bbox, o.color, o.shape) for o in sc.objs if not o.hud and o.bbox in movable)
    need = min(len(blocks), len(frames))

    fcol = {bc: fc for bc, fc in pairs}

    def goal(state) -> bool:
        k = 0
        for bbox, col, sh in state:
            if col in fcol and any(f.color == fcol[col] and f.inner[0] - 2 <= bbox[0] and f.inner[1] - 2 <= bbox[1] and bbox[2] <= f.inner[2] + 2 and bbox[3] <= f.inner[3] + 2
                                   and abs((bbox[0] + bbox[2]) / 2 - (f.inner[0] + f.inner[2]) / 2) <= 2 and abs((bbox[1] + bbox[3]) / 2 - (f.inner[1] + f.inner[3]) / 2) <= 2 for f in frames):
                k += 1
        return k >= need

    def step(state, sg):
        new = set()
        for bbox, col, sh in state:
            q = sg.get(bbox)
            if q is None and bbox in movable and bbox in sg:
                return None
            if q is None:
                # not on this button's track (or unsolved): a slot on this track with unknown successor makes the press unpredictable
                if bbox in {q0 for q0 in sg}:
                    return None
                new.add((bbox, col, sh))
            else:
                new.add((q, col, sh))
        return frozenset(new)

    if goal(start):
        return None
    q = deque([(start, [])]); seen = {start}; n = 0
    while q and n < max_states:
        state, path = q.popleft(); n += 1
        if len(path) >= max_depth:
            continue
        for key, sg in buttons:
            nxt = step(state, sg)
            if nxt is None or nxt in seen:
                continue
            seen.add(nxt)
            if goal(nxt):
                seq = path + [key]
                acts = [{"action": "MOUSE", "row": button_objs[k].center[0], "col": button_objs[k].center[1]} for k in seq]
                for a in acts:
                    if sc.frame.grid[a["row"]][a["col"]] != sc.obj_at(a["row"], a["col"]).color if sc.obj_at(a["row"], a["col"]) else True:
                        pass
                from collections import Counter
                cnt = Counter(f"#{button_objs[k].id}" for k in seq)
                return Candidate(f"plan:press({eid})", "plan", acts, f"{len(seq)} button presses ({', '.join(f'{b}x{c}' for b, c in cnt.items())}) move the colour-{p.color} blocks into the colour-{p.frame} frames on the learned tracks -> {eid} should hold",
                                 priority=5, pred_kind="plan")
            q.append((nxt, path + [key]))
    # no sequence on what is solved: press the least-explored button-like object (same colours as known buttons) so that every track gets solved
    known_cols = {key[0] for key in ev.cs.presses} | {key[0] for key in ev.cs_all.presses}
    cand = [o for o in sc.objs if o.color in known_cols and not o.hud and o.size >= 8]
    if not cand:
        return None
    b = min(cand, key=lambda o: (ev.cs.succ_n.get(o.key, 0), o.center))
    return Candidate(f"plan:learn({eid})", "plan", [{"action": "MOUSE", "row": b.center[0], "col": b.center[1]}],
                     f"press button #{b.id} ({ev.cs.succ_n.get(b.key, 0)} presses so far) to learn its track; no press sequence up to {max_depth} exists yet on the solved tracks ({n} states searched)",
                     priority=7, pred_kind="plan")


def _plan_marker(ev: Evidence, sc: Scene, p: goals2.Inside, eid: str) -> Optional[Candidate]:
    allf = goals2.find_frames(sc)
    for bc, fc in ([(c, c) for c in sorted({f.color for f in allf})] if p.color == p.frame else [(p.color, p.frame)]):
        c = _plan_marker_pair(ev, sc, p, eid, bc, fc, allf)
        if c is not None and c.kind == "plan":
            return c
    return None


def _plan_marker_pair(ev: Evidence, sc: Scene, p: goals2.Inside, eid: str, bc: int, fc: int, allf: list) -> Optional[Candidate]:
    frames = [f for f in allf if f.color == fc]
    dot_ids = {i for f in frames for i in f.dots}
    pieces = [o for o in sc.objs if o.color == bc and not o.hud and o.size >= 4 and o.id not in dot_ids and min(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]) >= 2
              and not any(f.contains(o) for f in frames)]
    if not frames or not pieces:
        return None
    # the marker colour that moves on clicks (cursor rule) and its coupling with the piece colour
    markers = []
    for (col, rg), cnt in ev.cs.cursor.items():
        k, n = cnt.most_common(1)[0]
        if k != -1 and n * 2 > sum(cnt.values()):
            markers.append(k)
    # objects whose click makes the marker jump onto them (selection): the other anchors
    selectable = []
    for (col, rg), cnt in ev.cs.select.items():
        if cnt.get(True, 0) > cnt.get(False, 0):
            selectable.extend(o for o in sc.objs if o.color == col and o.region == rg and not o.hud and o.size >= 4)
    for k in set(markers):
        cp = ev.cs.coupling(k, bc)
        if not cp:
            continue
        (rr, rc), _ = cp
        if float(rr) == 0 or float(rc) == 0:
            continue
        mks = [o for o in sc.objs if o.color == k and not o.hud and o.size >= 4]
        if not mks:
            continue
        mk = mks[0]
        piece = min(pieces, key=lambda o: min(abs(o.center[0] - (f.inner[0] + f.inner[2]) / 2) + abs(o.center[1] - (f.inner[1] + f.inner[3]) / 2) for f in frames))
        f = min(frames, key=lambda f: abs(piece.center[0] - (f.inner[0] + f.inner[2]) / 2) + abs(piece.center[1] - (f.inner[1] + f.inner[3]) / 2))
        need_r = (f.inner[0] + f.inner[2]) / 2 - piece.center[0]; need_c = (f.inner[1] + f.inner[3]) / 2 - piece.center[1]
        floor = sc.region_at(mk.center[0], mk.center[1])
        # option 1: move the selected marker itself; option 2: select another anchor first, then move it
        options = [(None, mk)] + [(a, a) for a in selectable if a.id != mk.id]
        blocked = []
        for sel, anchor in options:
            tr, tc = int(round(anchor.center[0] + need_r / float(rr))), int(round(anchor.center[1] + need_c / float(rc)))
            if not (0 <= tr < 64 and 0 <= tc < 64) or sc.region_at(tr, tc) != floor or sc.obj_at(tr, tc) is not None:
                blocked.append((anchor.id, tr, tc)); continue
            acts = ([{"action": "MOUSE", "row": sel.center[0], "col": sel.center[1]}] if sel is not None else []) + [{"action": "MOUSE", "row": tr, "col": tc}]
            how = f"select anchor #{sel.id} then click ({tr},{tc})" if sel is not None else f"click ({tr},{tc})"
            return Candidate(f"plan:marker({eid})", "plan", acts, f"{how}: the colour-{k} marker goes to ({tr},{tc}) and piece #{piece.id} (coupled x{rr}) lands in the colour-{fc} frame at ({f.inner[0]},{f.inner[1]}) -> {eid} should hold",
                             priority=5, pred_kind="plan")
        if blocked:
            return Candidate(f"plan:marker({eid})", "info", [], f"to put piece #{piece.id} in the frame an anchor would have to reach " + ", ".join(f"({r},{c})" for _, r, c in blocked[:3]) + " — not on the floor (wall or occupied)", priority=999)
    return None


def _plan_move(ev: Evidence, sc: Scene, p, eid: str) -> Optional[Candidate]:
    """Walk to the nearest object of the colour the predicate needs (collect first, then reach), on the learned floor map."""
    nav = ev.nav
    if nav is None or not ev.moves or not ev.avatar:
        return None
    if isinstance(p, goals2.CollectReach):
        ok1 = goals2.CollectAll(p.color).evaluate(sc, ev)[0]
        colour, what = (p.reach, "reach") if ok1 else (p.color, "collect")
    elif isinstance(p, goals2.CollectAll):
        colour, what = p.color, "collect"
    else:
        colour, what = p.color, "reach"
    box = goals2._avatar_box(ev)
    targets = [o for o in sc.objs if o.color == colour and not o.hud and o.size <= 400 and not (box and goals2._touching(box, o, 0))]
    if not targets:
        return None
    best = None
    for o in sorted(targets, key=lambda o: abs(o.center[0] - box[0]) + abs(o.center[1] - box[1]))[:8]:
        path = nav.path_to(o.center[0], o.center[1], ignore_wall_target=True)
        if path:
            if best is None or len(path) < len(best[1]):
                best = (o, path)
    if best is None:
        return Candidate(f"plan:{what}({eid})", "info", [], f"no known path from the avatar to a colour-{colour} object yet (explore the floor first)", priority=999)
    o, path = best
    gauge = f"; gauge {ev.gauge['per_action'] * len(path):+}" if ev.gauge else ""
    return Candidate(f"plan:{what}({eid})", "plan", list(path), f"walk {len(path)} steps ({''.join(a[0] for a in path)}) to colour-{colour} object #{o.id} at ({o.center[0]},{o.center[1]}){gauge} -> {eid}: {'it should vanish' if what == 'collect' else 'the avatar touches it'}",
                     priority=6, pred_kind="plan")
