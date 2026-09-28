"""Deterministic predictor: from the recorded transitions, say what each action should do on the current board.

Evidence is rebuilt from the game record (no model): movement deltas / walls / gauge (arcnav.nav, arcnav.rules), a
position-relative click-effect model (arcnav.effects), no-op keys, and win facts from completed levels (arcnav.goals).
The same evidence is exported as `facts()` so the rulebook can score the model's hypotheses against it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from arcnav.effects import ClickModel
from arcnav.frame import Frame, masked_ascii, summarize_diff
from arcnav.nav import NavHelper, extract_objects
from arcnav import goals as goal_inference

from .env import Game, Transition, MOVE_KEYS, action_label
from .entities import Obj, Scene, build_scene, match
from .clicks import effect_summary, effect_text, key_effect, ClickStats, change_class, record, _edge_only_diff


@dataclass
class Prediction:
    kind: str                 # move | blocked | noop | board | hud | unknown | level
    text: str
    avatar_to: Optional[tuple] = None      # (row, col) of the avatar's bbox top-left after a move
    board: Optional[list] = None           # predicted grid (board kind)
    rules: list = field(default_factory=list)   # (kind, params) the prediction relies on
    confidence: str = "hypothesis"         # confirmed | hypothesis | none
    moves: Optional[dict] = None           # objects kind: before-object id -> predicted new bbox
    coupled: Optional[list] = None         # cursor kind: [(colour, predicted (dr, dc))] for objects coupled to the marker
    effect: Optional[tuple] = None         # effects kind: coarse per-colour summary (clicks.effect_summary)


@dataclass
class Verdict:
    ok: Optional[bool]        # True = as predicted, False = mismatch, None = no prediction to check (observation)
    text: str
    diff: dict = field(default_factory=dict)


class Evidence:
    """Everything the harness knows for sure at this moment, derived from the transitions."""

    def __init__(self, game: Game, distrust: Optional[dict] = None, custom_rules: Optional[list] = None):
        self.game = game
        self.custom_rules = custom_rules or []   # (name, fn, accuracy, n): model-written, replay-verified transition rules
        self.distrust = distrust or {}     # click colour -> number of failed board predictions on this level (agent-maintained)
        self.frame: Frame = game.frame
        cur = game.attempt_transitions()
        self.nav = NavHelper(cur, self.frame) if game.has_move_keys() else None   # click-only games need no movement model
        try:
            self.rules = induce(game, self.nav)
        except Exception:
            self.rules = []
        self.click = ClickModel()
        for t in game.transitions:
            if isinstance(t.action, dict):
                self.click.observe(t.before_frame.grid, t.after_frame.grid, t.action["row"], t.action["col"])
        self.click.fit()
        self.moves: dict = dict(self.nav.moves) if self.nav else {}
        self.avatar = self.nav.avatar() if self.nav else None
        self.wall_colors = {r["params"]["color"] for r in self.rules if r["kind"] == "wall" and r["support"] > r["counter"]}
        self.ambiguous_colors = {r["params"]["color"]: (r["support"], r["counter"]) for r in self.rules if r["kind"] == "wall" and r["counter"] >= 1}
        # a key is a no-op only while every press did nothing; once it did something the effect is conditional (state-dependent)
        self.noop_keys = {r["params"]["action"] for r in self.rules if r["kind"] == "noop" and r["support"] >= 1 and r["counter"] == 0}
        self.conditional_keys = {r["params"]["action"]: (r["support"], r["counter"]) for r in self.rules if r["kind"] == "noop" and r["counter"] >= 1}
        self.flaky_moves = {r["params"]["action"]: (r["support"], r["counter"]) for r in self.rules if r["kind"] == "move" and r["counter"] >= max(2, r["support"])}
        self.hazard_colors = {r["params"]["color"] for r in self.rules if r["kind"] == "hazard"}
        self.gauge = self.nav.gauge() if self.nav else None
        # click statistics keyed by (colour, region) and by button object: the current level's records first, all levels as fallback
        self.scene: Scene = build_scene(self.frame)
        self.distrust = {k if isinstance(k, tuple) else (k, None): v for k, v in self.distrust.items()}
        self.cs = ClickStats(); self.cs_all = ClickStats()
        for t in game.transitions:
            if isinstance(t.action, dict):
                rec = record(t)
                (self.cs if t.level == game.level else self.cs_all).add(rec)
        # keys that move several objects at once (rotations, scrolls, conveyors driven by a key) are learned like buttons
        for t in game.level_transitions():
            if isinstance(t.action, str) and t.action not in self.moves:
                eff = key_effect(t)
                self.cs.add_effect(("KEY", t.action), eff if eff is not None else ())
                if eff is None or sum(1 for k, sg, mv in eff if mv) < 1:
                    continue
                sa, sb = build_scene(t.before_frame), build_scene(t.after_frame)
                m = match(sa, sb)
                if len(m["moved"]) >= 2:
                    occ_b = {o.bbox: (o.color, o.shape) for o in sa.objs if not o.hud and o.size >= 2 and 1 < o.center[0] < 62 and 1 < o.center[1] < 62}
                    occ_a = {o.bbox: (o.color, o.shape) for o in sb.objs if not o.hud and o.size >= 2 and 1 < o.center[0] < 62 and 1 < o.center[1] < 62}
                    self.cs.add_key_press(t.action, occ_b, occ_a)
        if game.level_transitions():   # the offset click-effect model too: refit on this level once it has clicks
            lvl_clicks = [t for t in game.level_transitions() if isinstance(t.action, dict)]
            if lvl_clicks:
                self.click = ClickModel()
                for t in lvl_clicks:
                    self.click.observe(t.before_frame.grid, t.after_frame.grid, t.action["row"], t.action["col"])
                self.click.fit(min_support=0.8)

    def _stat(self, name: str, colour: int, region: int):
        """(counter, exact_region?) from this level's stats, else from all levels."""
        v, exact = getattr(self.cs, name)(colour, region)
        if v:
            return v, exact, True
        v, exact = getattr(self.cs_all, name)(colour, region)
        return v, exact, False

    def click_key(self, r: int, c: int) -> tuple:
        o = self.scene.obj_around(r, c)
        return (o.color if o is not None else self.frame.grid[r][c], o.region if o else self.scene.label[r][c]), o

    # ── facts for the rulebook ───────────────────────────────────────────
    def facts(self) -> list[dict]:
        out = [dict(r) for r in self.rules]
        if self.avatar:
            av = self.avatar; x0, y0, x1, y1 = av["bbox_xyxy"]
            out.append({"kind": "avatar", "params": {"colors": av["colors"]}, "support": 2, "counter": 0,
                        "text": f"the avatar (what the movement keys move) is the colour-{av['colors']} block, {y1 - y0 + 1}x{x1 - x0 + 1} px, now at rows {y0}-{y1} cols {x0}-{x1}"})
        # per (colour, region) click facts from this level (fallback: all levels)
        stats = self.cs if self.cs.n else self.cs_all
        for (c, rg), h in sorted(stats.hist.items()):
            n = sum(h.values()); where = f" in P{rg}" if rg >= 0 else ""
            rgl = self.scene.region_named(rg)
            rgtxt = f" (P{rg} = the colour-{rgl.color} area)" if rgl else ""
            if h.get("world", 0) == 0:
                what = "changes only the HUD/counter" if h.get("hud") else "changes nothing"
                out.append({"kind": "click", "params": {"color": c, "region": rg, "changes": []}, "support": n, "counter": 0,
                            "text": f"clicking colour {c}{where}{rgtxt} {what} ({n} clicks)"})
                continue
            parts = []
            vt = stats.vanish.get((c, rg)); cu = stats.cursor.get((c, rg))
            if vt and vt.most_common(1)[0][0] != -1 and vt.most_common(1)[0][1] * 2 > n:
                k, m = vt.most_common(1)[0]; parts.append(f"the clicked object turns into colour {k} ({m}/{n})")
            if cu and cu.most_common(1)[0][0] != -1 and cu.most_common(1)[0][1] * 2 > n:
                k, m = cu.most_common(1)[0]
                sel = stats.select.get((c, rg), Counter())
                parts.append(f"the colour-{k} marker {'jumps onto the clicked object (selection)' if sel.get(True, 0) > sel.get(False, 0) else 'moves to the clicked cell'} ({m}/{n})")
            if h.get("hud", 0) + h.get("none", 0):
                parts.append(f"{h.get('hud', 0) + h.get('none', 0)}/{n} clicks changed nothing in the playfield")
            if not parts:
                parts.append(f"changes the playfield ({h.get('world', 0)}/{n}), effect not yet modelled")
            out.append({"kind": "click", "params": {"color": c, "region": rg}, "support": n, "counter": 0,
                        "text": f"clicking colour {c}{where}{rgtxt}: " + "; ".join(parts)})
        for b, n in stats.succ_n.items():
            sg = stats.sigma(b); moving = sum(1 for p_, q in sg.items() if p_ != q)
            out.append({"kind": "button", "params": {"color": b[0], "region": b[1], "bbox": list(b[2])}, "support": n, "counter": 0,
                        "text": f"the colour-{b[0]} button at rows {b[2][0]}-{b[2][2]} cols {b[2][1]}-{b[2][3]} shifts every block on its track one slot "
                                f"({moving} slot transitions solved from {n} presses; blocks return to their slot after a full cycle)"})
        for (mk, oc), cnt in stats.coupled.items():
            r = stats.coupling(mk, oc)
            if r:
                (rr, rc), m = r
                out.append({"kind": "coupled", "params": {"marker": mk, "color": oc, "ratio": [str(rr), str(rc)]}, "support": m, "counter": sum(cnt.values()) - m,
                            "text": f"when the colour-{mk} marker moves by (dr, dc), each colour-{oc} object moves by ({rr}*dr, {rc}*dc) — it sits at a fixed fraction (e.g. the centroid) of the markers"})
        return out

    def win_facts(self, level: int) -> list[dict]:
        """Win hypotheses from the attempt that completed `level` (goals.infer), as rulebook facts."""
        same = self.game.level_transitions(level)
        if not same:
            return []
        last_attempt = same[-1].attempt
        attempt = [t for t in same if t.attempt == last_attempt]
        try:
            hyps = goal_inference.infer(attempt, level, movement_keys=self.game.has_move_keys() or bool(self.moves))
        except Exception:
            hyps = []
        out = []
        for h in hyps:
            if h.get("type") in ("reach", "collect_reach", "collect_all", "click_sequence", "merge"):
                p = {k: v for k, v in h.items() if k in ("reach", "collect", "collect_count", "sequence", "colors")}
                out.append({"kind": h["type"], "params": p, "support": 1, "counter": 0, "text": h["text"]})
        return out

    # ── prediction ───────────────────────────────────────────────────────
    def _block_colour(self, row: int, col: int) -> Optional[int]:
        if not (0 <= row < 64 and 0 <= col < 64):
            return None
        return self.frame.grid[row][col]

    def predict(self, act) -> Prediction:
        grid = self.frame.grid
        for name, fn, acc, n in self.custom_rules:
            try:
                out = fn([row[:] for row in grid], act)
            except Exception:
                out = None
            if out is not None and len(out) == 64:
                changed = sum(1 for i in range(64) for j in range(64) if out[i][j] != grid[i][j])
                return Prediction("board", f"model rule `{name}` ({acc:.0%} on {n} replays): {changed} cells change", board=out, rules=[("other", {"name": name})], confidence="confirmed" if acc >= 0.9 else "hypothesis")
        if isinstance(act, dict):
            r, c = act["row"], act["col"]; colour = grid[r][c]
            (colour, region), obj = self.click_key(r, c)
            where = f" in P{region}" if region >= 0 else ""
            rule = ("click", {"color": colour, "region": region})
            h, exact, this_level = self._stat("hist_of", colour, region)
            n = sum(h.values()) if h else 0
            if h and h.get("world", 0) == 0 and (exact or n >= 3):
                kind = "hud" if h.get("hud") else "noop"
                return Prediction(kind, f"colour {colour}{where} clicked {n}x before: " + ("only the HUD changed" if kind == "hud" else "nothing changed"),
                                  rules=[rule], confidence="confirmed" if n >= 2 else "hypothesis")
            if obj is not None and self.cs.is_dead(obj.key):
                return Prediction("hud", f"this object was clicked {self.cs.dead[obj.key]}x on this level with no effect in the playfield", rules=[rule], confidence="confirmed")
            ep = self._effect_pred(("BTN", obj.bbox) if obj is not None else ("CELL", colour, region), f"colour {colour}{where}" + (f" button at ({obj.bbox[0]},{obj.bbox[1]})" if obj is not None else ""), rule)
            if ep is not None and self.distrust.get((colour, region), 0) >= 2:
                return ep
            if self.distrust.get((colour, region), 0) >= 2:
                return Prediction("unknown", f"colour {colour}{where}: the effect model was wrong {self.distrust[(colour, region)]}x on this level; effect not predictable yet (clicked {n}x)", rules=[rule], confidence="none")
            # button with learned slot successors (conveyor / permutation)
            if obj is not None and self.cs.succ_n.get(obj.key):
                moves = {}; known = 0; unknown = 0; slots = self.cs.slots(obj.key); tent = self.cs.tentative(obj.key); guesses = {}
                for o in self.scene.objs:
                    if o.hud or o.id == obj.id:
                        continue
                    nb = self.cs.successor(obj.key, o.color, o.bbox)
                    if nb is not None and nb != o.bbox:
                        moves[o.id] = nb; known += 1
                    elif nb is None and o.bbox in slots:
                        unknown += 1
                        if o.bbox in tent and tent[o.bbox] != o.bbox:
                            guesses[o.id] = tent[o.bbox]
                if moves and known * 2 >= known + unknown:
                    return Prediction("objects", f"button #{obj.id}: {known} objects shift one slot along their tracks (" + ", ".join(f"#{i}->({b[0]},{b[1]})" for i, b in list(moves.items())[:4]) + (" ..." if len(moves) > 4 else ")") + (f"; {unknown} objects on slots not yet solved" + (", probably " + ", ".join(f"#{i}->({b[0]},{b[1]})" for i, b in list(guesses.items())[:3]) if guesses else "") if unknown else ""),
                                      moves=moves, rules=[("button", {"color": colour, "region": region})], confidence="confirmed" if self.cs.succ_n[obj.key] >= 2 else "hypothesis")
                if moves:
                    return Prediction("unknown", f"button #{obj.id}: {known} objects would shift along known tracks but {unknown} sit on slots whose next slot is still unknown", rules=[rule], confidence="none")
                return Prediction("unknown", f"button #{obj.id}: moved objects before ({self.cs.succ_n[obj.key]} presses), track not yet solved", rules=[rule], confidence="none")
            cu, cu_exact, _ = self._stat("cursor_of", colour, region)
            if cu and cu.most_common(1)[0][0] != -1 and cu.most_common(1)[0][1] * 2 > sum(cu.values()):
                k = cu.most_common(1)[0][0]
                markers = [o for o in self.scene.objs if o.color == k and not o.hud and o.size >= 4]
                here = [o for o in markers if o.bbox[0] <= r <= o.bbox[2] and o.bbox[1] <= c <= o.bbox[3]]
                if here:
                    return Prediction("unknown", f"the colour-{k} marker is already at ({r},{c}); clicking here again has an unknown effect", confidence="none")
                sel = self.cs.select.get((colour, region)) or self.cs_all.select.get((colour, region)) or Counter()
                if obj is not None and obj.size >= 4 and sel.get(True, 0) > sel.get(False, 0):
                    return Prediction("cursor", f"the colour-{k} marker jumps onto the clicked object #{obj.id} (selection)", rules=[rule], confidence="confirmed" if sum(cu.values()) >= 2 else "hypothesis", avatar_to=(r, c))
                coupled = []
                if markers:
                    mk = min(markers, key=lambda o: abs(o.center[0] - r) + abs(o.center[1] - c))
                    dr, dc = r - mk.center[0], c - mk.center[1]
                    for oc in {o.color for o in self.scene.objs if not o.hud and o.color != k}:
                        cp = self.cs.coupling(k, oc)
                        if cp:
                            (rr, rc), _ = cp; coupled.append((oc, (round(float(rr) * dr), round(float(rc) * dc))))
                txt = f"the colour-{k} marker moves to the click point ({r},{c})"
                if coupled:
                    txt += "; coupled: " + ", ".join(f"colour {oc} objects move by ({d[0]:+},{d[1]:+})" for oc, d in coupled)
                else:
                    txt += f" (other cells may change too: {h.get('world', 0) if h else 0} world changes seen)"
                return Prediction("cursor", txt, rules=[rule], confidence="confirmed" if sum(cu.values()) >= 2 else "hypothesis", avatar_to=(r, c), coupled=coupled or None)
            vt, _, _ = self._stat("vanish_of", colour, region)
            shifts = self.cs.shifts_of(colour, region)
            if shifts:
                moves = {}
                for o in self.scene.objs:
                    for (k, dr, dc) in shifts:
                        if o.color == k and not o.hud and o.size >= 4:
                            moves[o.id] = (o.bbox[0] + dr, o.bbox[1] + dc, o.bbox[2] + dr, o.bbox[3] + dc)
                if moves:
                    return Prediction("objects", f"colour {colour}{where} click: " + ", ".join(f"colour-{k} objects move ({dr:+},{dc:+})" for k, dr, dc in shifts) + f" ({len(moves)} objects)",
                                      moves=moves, rules=[rule], confidence="confirmed" if self.cs.shift_n[(colour, region)] >= 3 else "hypothesis")
            hit = self.scene.obj_at(r, c)
            if vt and vt.most_common(1)[0][0] != -1 and vt.most_common(1)[0][1] * 2 > sum(vt.values()) and hit is not None and hit.cells:
                k = vt.most_common(1)[0][0]
                board = [row[:] for row in grid]
                for rr, cc in hit.cells:
                    board[rr][cc] = k
                return Prediction("board", f"the clicked colour-{colour} object #{hit.id}{where} ({hit.size} cells) turns into colour {k}", board=board,
                                  rules=[rule], confidence="confirmed" if sum(vt.values()) >= 2 else "hypothesis")
            board = self.click.predict(grid, r, c) if self.click.clicks[colour] >= 5 else None
            if board is not None:
                changed = sum(1 for i in range(64) for j in range(64) if board[i][j] != grid[i][j])
                pairs = Counter((grid[i][j], board[i][j]) for i in range(64) for j in range(64) if board[i][j] != grid[i][j])
                conf = "confirmed" if self.click.clicks[colour] >= 2 else "hypothesis"
                return Prediction("board", f"changes {changed} cells near ({r},{c}): " + ", ".join(f"{b}->{a}x{n}" for (b, a), n in pairs.most_common(3)),
                                  board=board, rules=[rule], confidence=conf)
            if ep is not None:
                return ep
            if h:
                return Prediction("unknown", f"colour {colour}{where} clicked {n}x before with no consistent effect model", rules=[rule], confidence="none")
            return Prediction("unknown", f"colour {colour}{where} never clicked" + ("" if this_level else " on this level"), confidence="none")
        name = act
        bkey = ("KEY", name)
        if self.cs.succ_n.get(bkey) and name not in self.moves:
            sg = self.cs.sigma(bkey); slots = self.cs.slots(bkey)
            moves = {}; unknown = 0
            for o in self.scene.objs:
                if o.hud:
                    continue
                nb = sg.get(o.bbox)
                if nb is not None and nb != o.bbox:
                    moves[o.id] = nb
                elif nb is None and o.bbox in slots:
                    unknown += 1
            if moves and len(moves) * 2 >= len(moves) + unknown:
                return Prediction("objects", f"{name}: {len(moves)} objects move along fixed tracks (learned from {self.cs.succ_n[bkey]} presses)" + (f"; {unknown} on unsolved slots" if unknown else ""),
                                  moves=moves, rules=[("button", {"color": -1, "region": -1})], confidence="confirmed" if self.cs.succ_n[bkey] >= 2 else "hypothesis")
            return Prediction("unknown", f"{name}: moves several objects at once ({self.cs.succ_n[bkey]} presses); tracks not solved yet", confidence="none")
        if name in self.conditional_keys and name not in self.moves:
            n0, n1 = self.conditional_keys[name]
            ep = self._effect_pred(("KEY", name), name, None)
            if n0 >= 5 and n0 >= 4 * n1:
                return Prediction("noop", f"{name}: changed nothing {n0}x, changed the board only {n1}x — treat as a no-op in the current state (some unknown condition enables it)" + (f"; when it works: {ep.text}" if ep else ""),
                                  rules=[("noop", {"action": name})], confidence="hypothesis")
            return Prediction("unknown", f"{name}: changed nothing {n0}x but changed the board {n1}x — the effect depends on a condition (state) not yet known" + (f"; when it works: {ep.text}" if ep else ""), confidence="none")
        if name in self.flaky_moves:
            n0, n1 = self.flaky_moves[name]
            return Prediction("unknown", f"{name}: the learned move delta fit {n0}x but failed {n1}x — the avatar or step size is not identified reliably", confidence="none")
        if name in self.noop_keys and name not in self.moves:
            return Prediction("noop", f"{name} changed nothing before", rules=[("noop", {"action": name})],
                              confidence="confirmed" if any(r["kind"] == "noop" and r["params"]["action"] == name and r["support"] >= 2 for r in self.rules) else "hypothesis")
        if name in self.moves and self.avatar:
            dx, dy = self.moves[name]
            x0, y0, x1, y1 = self.avatar["bbox_xyxy"]
            nx0, ny0, nx1, ny1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            if nx0 < 0 or ny0 < 0 or nx1 > 63 or ny1 > 63:
                return Prediction("blocked", f"{name}: avatar at the board edge, stays at ({y0},{x0})", avatar_to=(y0, x0), rules=[("move", {"action": name})], confidence="hypothesis")
            # cells the avatar would newly cover
            new_cells = [(rr, cc) for rr in range(ny0, ny1 + 1) for cc in range(nx0, nx1 + 1) if not (x0 <= cc <= x1 and y0 <= rr <= y1)]
            cols = Counter(grid[rr][cc] for rr, cc in new_cells)
            hit = [c for c in cols if c in self.wall_colors]
            haz = [c for c in cols if c in self.hazard_colors]
            amb = [c for c in cols if c in self.ambiguous_colors and c not in self.wall_colors]
            if amb and not hit and not haz:
                b, k = self.ambiguous_colors[amb[0]]
                return Prediction("unknown", f"{name}: colour {amb[0]} ahead — blocked {b}x but entered {k}x before; outcome undetermined (some condition decides)", confidence="none")
            gauge_txt = f"; gauge {self.gauge['per_action']:+}" if self.gauge else ""
            conf = "confirmed" if any(r["kind"] == "move" and r["params"]["action"] == name and r["support"] >= 2 for r in self.rules) else "hypothesis"
            if haz:
                return Prediction("move", f"{name}: enters colour {haz[0]} = HAZARD (reset expected)", avatar_to=(ny0, nx0), rules=[("move", {"action": name}), ("hazard", {"color": haz[0]})], confidence=conf)
            if hit:
                return Prediction("blocked", f"{name}: blocked by wall colour {hit[0]}, avatar stays at ({y0},{x0})", avatar_to=(y0, x0),
                                  rules=[("move", {"action": name}), ("wall", {"color": hit[0]})], confidence=conf)
            # unknown terrain: the block ahead is neither known floor nor known wall
            ahead = cols.most_common(1)[0][0] if cols else None
            floor = self.nav.floor_colors if self.nav else set()
            if ahead is not None and ahead not in floor and ahead != self.nav.background:
                return Prediction("move", f"{name}: avatar to ({ny0},{nx0}) onto colour {ahead} (untested terrain: may block or trigger something){gauge_txt}",
                                  avatar_to=(ny0, nx0), rules=[("move", {"action": name})], confidence="hypothesis")
            return Prediction("move", f"{name}: avatar moves ({dy:+},{dx:+}) to ({ny0},{nx0}){gauge_txt}", avatar_to=(ny0, nx0), rules=[("move", {"action": name})], confidence=conf)
        if name in self.moves:
            return Prediction("unknown", f"{name}: moves something by {self.moves[name]} (avatar not identified)", confidence="none")
        pressed = sum(1 for t in self.game.transitions if t.action == name)
        ep = self._effect_pred(("KEY", name), name, None)
        if ep is not None:
            return ep
        return Prediction("unknown", f"{name} never pressed" if not pressed else f"{name} pressed {pressed}x, effect not modelled", confidence="none")

    def _key_of_bbox(self, bbox):
        o = next((o for o in self.scene.objs if o.bbox == bbox), None)
        return (o.color, o.region) if o is not None else None

    def _effect_pred(self, ek, label: str, rule) -> Optional[Prediction]:
        got = self.cs.effect_of(ek)
        if got is None:
            return None
        eff, m, n = got
        if not eff or (any(mv for k, sg, mv in eff) and m < 3):
            return None            # empty summaries and weakly supported movement summaries are left to the other models
        if ek[0] != "KEY":
            cur = self.cs.cursor.get((ek[2], ek[3]) if ek[0] == "CELL" else self._key_of_bbox(ek[1]))
            if cur and any(k != -1 for k in cur):
                return None        # marker/cursor games: the click position decides, not the button
        txt = f"{label}: {effect_text(eff)} ({m}/{n} times so far)"
        return Prediction("effects" if eff else "hud", txt, effect=eff, rules=[rule] if rule else [], confidence="confirmed" if m >= 3 and m == n else "hypothesis")

    # ── synthesised post-board (for goal inference) ──────────────────────
    def synth_board(self, pred: Prediction, act) -> Optional[list]:
        """The board a prediction implies: board kind as is; objects / cursor kinds by moving objects on a copy of the grid."""
        if pred.kind == "board":
            return pred.board
        moves: dict = {}
        if pred.kind == "objects" and pred.moves:
            moves = dict(pred.moves)
        elif pred.kind == "cursor" and isinstance(act, dict):
            k = None
            for o in self.scene.objs:
                pass
            markers = [o for o in self.scene.objs if not o.hud and o.size >= 4 and ("marker" in pred.text) and f"colour-{o.color} marker" in pred.text]
            if markers:
                mk = min(markers, key=lambda o: abs(o.center[0] - act["row"]) + abs(o.center[1] - act["col"]))
                dr, dc = act["row"] - mk.center[0], act["col"] - mk.center[1]
                moves[mk.id] = (mk.bbox[0] + dr, mk.bbox[1] + dc, mk.bbox[2] + dr, mk.bbox[3] + dc)
                for oc, (pdr, pdc) in (pred.coupled or []):
                    for o in self.scene.objs:
                        if o.color == oc and not o.hud and o.size >= 4 and min(o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]) >= 2:
                            moves[o.id] = (o.bbox[0] + pdr, o.bbox[1] + pdc, o.bbox[2] + pdr, o.bbox[3] + pdc)
        if not moves:
            return None
        grid = [row[:] for row in self.frame.grid]
        for i, nb in moves.items():
            o = self.scene.objs[i]
            fill = self.scene.region_named(o.region).color if o.region >= 0 and self.scene.region_named(o.region) else self.frame.background
            for r, c in (o.cells or []):
                grid[r][c] = fill
        for i, nb in moves.items():
            o = self.scene.objs[i]
            for r, c in (o.cells or []):
                rr, cc = r + nb[0] - o.bbox[0], c + nb[1] - o.bbox[1]
                if 0 <= rr < 64 and 0 <= cc < 64:
                    grid[rr][cc] = o.color
        return grid

    # ── checking ─────────────────────────────────────────────────────────
    def check(self, pred: Prediction, t: Transition, res: dict) -> Verdict:
        d = summarize_diff(t.before_frame, t.after_frame)
        world = change_class(t.before_frame, t.after_frame) == "world"
        summary = _diff_text(d, world)
        if res.get("level_completed"):
            return Verdict(None, "LEVEL COMPLETED — " + summary, d)
        if res.get("game_over"):
            return Verdict(False if pred.kind not in ("unknown",) and "HAZARD" not in pred.text else None, "GAME OVER — " + summary, d)
        if pred.kind == "unknown":
            return Verdict(None, "observed: " + summary, d)
        if pred.kind in ("move", "blocked"):
            av_after = NavHelper(self.game.attempt_transitions(), t.after_frame).avatar()
            if av_after is None:
                return Verdict(None, "avatar lost after the move; observed: " + summary, d)
            got = (av_after["bbox_xyxy"][1], av_after["bbox_xyxy"][0])
            if "HAZARD" in pred.text:
                return Verdict(None, "hazard step; observed: " + summary, d)
            ok = got == tuple(pred.avatar_to)
            extra = "" if not world or pred.kind == "blocked" else ""
            # a move that also changed something else in the world (collect/toggle) is not a mismatch of the move rule, but worth noting
            side = [x for x in (d.get("disappeared") or []) + (d.get("appeared") or [])]
            note = f"; side effect: {summary}" if ok and side else ""
            return Verdict(ok, ("as predicted" if ok else f"MISMATCH: avatar at {got}, predicted {tuple(pred.avatar_to)}") + note, d)
        if pred.kind == "cursor":
            rec = record(t)
            ok = rec.marker is not None
            note = ""
            if ok and pred.coupled:
                mdr, mdc = rec.marker_to[0] - rec.marker.bbox[0], rec.marker_to[1] - rec.marker.bbox[1]
                for oc, (pdr, pdc) in pred.coupled:
                    got = [(dr, dc) for oa, ob, dr, dc in rec.moved if oa.color == oc and oa.size >= 4]
                    if not got:
                        ok = False; note += f"; colour {oc} did not move (predicted ({pdr:+},{pdc:+}))"
                    elif not any(abs(dr - pdr) <= 1 and abs(dc - pdc) <= 1 for dr, dc in got):
                        ok = False; note += f"; colour {oc} moved {got[0]} (predicted ({pdr:+},{pdc:+}))"
            return Verdict(ok, ("as predicted (marker moved to the click" + ("; coupled objects as predicted" if pred.coupled else "") + "); " if ok else "MISMATCH: " + ("no marker moved to the click; " if rec.marker is None else "marker moved but" + note + "; ")) + summary, d)
        if pred.kind == "effects":
            got = effect_summary(build_scene(t.before_frame), build_scene(t.after_frame)) if world else ()
            want = {k: (sg, mv) for k, sg, mv in pred.effect}; have = {k: (sg, mv) for k, sg, mv in got}
            bad = []
            for k in set(want) | set(have):
                w, h_ = want.get(k), have.get(k)
                if w is None:
                    bad.append(f"colour {k} changed unexpectedly")
                elif h_ is None:
                    bad.append(f"colour {k} did not change")
                elif w[0] != h_[0] and w[0] and h_[0]:
                    bad.append(f"colour {k} " + ("grew" if h_[0] > 0 else "shrank") + " instead")
                elif w[1] and h_[1] and (abs(w[1][0] - h_[1][0]) > 1 or abs(w[1][1] - h_[1][1]) > 1):
                    bad.append(f"colour {k} moved {h_[1]} not {w[1]}")
            if not bad:
                return Verdict(True, "as predicted (" + effect_text(got) + "); " + summary, d)
            return Verdict(False, "MISMATCH: " + "; ".join(bad[:3]) + "; " + summary, d)
        if pred.kind == "objects":
            sb, sa_ = build_scene(t.before_frame), build_scene(t.after_frame)
            occ_after = {o.bbox: (o.color, o.shape) for o in sa_.objs if not o.hud}
            arriving = set(pred.moves.values())
            wrong = []
            for i, q in pred.moves.items():
                o = sb.objs[i]
                if occ_after.get(q) != (o.color, o.shape):
                    wrong.append((i, q, "slot not filled as predicted"))
                elif o.bbox not in arriving and occ_after.get(o.bbox) == (o.color, o.shape):
                    wrong.append((i, q, "did not leave its slot"))
            if not wrong:
                return Verdict(True, f"as predicted ({len(pred.moves)} objects shifted along their tracks)", d)
            i, q, why = wrong[0]
            return Verdict(False, f"MISMATCH: {len(wrong)}/{len(pred.moves)} objects not where predicted (e.g. #{i} colour {sb.objs[i].color} -> ({q[0]},{q[1]}): {why}); " + summary, d)
        if pred.kind == "noop":
            ok = not world and not d.get("changed_cells")
            return Verdict(ok, "as predicted (nothing changed)" if ok else "MISMATCH: predicted no change but " + summary, d)
        if pred.kind == "hud":
            ok = not world
            return Verdict(ok, "as predicted (HUD only)" if ok else "MISMATCH: predicted HUD-only but " + summary, d)
        if pred.kind == "board":
            pm = _mask_like(pred.board, t.after_frame); am = masked_ascii(t.after_frame)
            if pm == am or _edge_only_diff(pred.board, t.after_frame.grid):
                return Verdict(True, "as predicted (board matches)", d)
            # how far off?
            wrong = sum(1 for a, b in zip(pm, am) if a != b)
            return Verdict(False, f"MISMATCH: {wrong} cells differ from the predicted board; " + summary, d)
        return Verdict(None, "observed: " + summary, d)


def _mask_like(board, frame: Frame) -> str:
    return masked_ascii(Frame(board, step=frame.step, level=frame.level))


def _diff_text(d: dict, world: bool) -> str:
    if not d.get("changed_cells"):
        return "nothing changed"
    parts = [f"{d['changed_cells']} cells changed" + ("" if world else " (HUD only)")]
    if d.get("moved"):
        parts.append("moved " + ", ".join(f"colour {m['color']} {m['from']}->{m['to']}" for m in d["moved"][:3]))
    if d.get("disappeared"):
        parts.append("vanished " + ", ".join(f"colour {x['color']}({x['pixels']}px)@{x['center']}" for x in d["disappeared"][:3]))
    if d.get("appeared"):
        parts.append("appeared " + ", ".join(f"colour {x['color']}({x['pixels']}px)@{x['center']}" for x in d["appeared"][:3]))
    if d.get("bbox"):
        parts.append(f"in rows {d['bbox'][0]}-{d['bbox'][2]}, cols {d['bbox'][1]}-{d['bbox'][3]}")
    return "; ".join(parts)




def _avatar_in(grid, nav: NavHelper) -> Optional[dict]:
    """The avatar object (top player vote colour/size) in an arbitrary grid; None if not found."""
    objs, _ = extract_objects(grid)
    for (color, size), _ in nav.player_votes.most_common(3):
        for o in objs:
            if o["color"] == color and o["size"] == size:
                return o
    return None


def induce(game: Game, nav: Optional[NavHelper]) -> list[dict]:
    """Cheap O(n) rule induction over the current level's transitions (all attempts): move deltas, walls, no-ops, gauge,
    collectibles, hazards, refills. Each fact: {kind, params, support, counter, text}."""
    out: list[dict] = []
    trans = game.level_transitions()
    if not trans:
        return out
    moves = dict(nav.moves) if nav else {}
    ok: Counter = Counter(); blocked: Counter = Counter(); bad: Counter = Counter(); walls: Counter = Counter()
    noop: Counter = Counter(); pressed: Counter = Counter(); vanish: Counter = Counter(); hazard: Counter = Counter()
    entered: Counter = Counter()
    start_pos = None
    for t in trans:
        if isinstance(t.action, dict):
            continue
        a = t.action; pressed[a] += 1
        if t.before_frame.ascii == t.after_frame.ascii:
            noop[a] += 1
        if nav is None or a not in moves:
            continue
        ab, aa = _avatar_in(t.before_frame.grid, nav), _avatar_in(t.after_frame.grid, nav)
        if not ab or not aa:
            continue
        if start_pos is None:
            start_pos = ab["bbox"][:2]
        dx, dy = aa["bbox"][0] - ab["bbox"][0], aa["bbox"][1] - ab["bbox"][1]
        if (dx, dy) == moves[a]:
            ok[a] += 1
            x0, y0, x1, y1 = ab["bbox"]
            for yy in range(y0 + dy, y1 + dy + 1):
                for xx in range(x0 + dx, x1 + dx + 1):
                    if 0 <= yy < 64 and 0 <= xx < 64 and not (x0 <= xx <= x1 and y0 <= yy <= y1):
                        entered[t.before_frame.grid[yy][xx]] += 1
        elif (dx, dy) == (0, 0):
            blocked[a] += 1
            mx, my = moves[a]; x0, y0, x1, y1 = ab["bbox"]
            cells = [(yy, xx) for yy in range(y0 + my, y1 + my + 1) for xx in range(x0 + mx, x1 + mx + 1)
                     if 0 <= yy < 64 and 0 <= xx < 64 and not (x0 <= xx <= x1 and y0 <= yy <= y1)]
            if cells:
                walls[Counter(t.before_frame.grid[yy][xx] for yy, xx in cells).most_common(1)[0][0]] += 1
        else:
            if start_pos is not None and aa["bbox"][:2] == start_pos and abs(dx) + abs(dy) > 6:
                mx, my = moves[a]; x0, y0, x1, y1 = ab["bbox"]
                cells = [(yy, xx) for yy in range(y0 + my, y1 + my + 1) for xx in range(x0 + mx, x1 + mx + 1)
                         if 0 <= yy < 64 and 0 <= xx < 64 and not (x0 <= xx <= x1 and y0 <= yy <= y1)]
                if cells:
                    hazard[Counter(t.before_frame.grid[yy][xx] for yy, xx in cells).most_common(1)[0][0]] += 1
            else:
                bad[a] += 1
        # collectibles: a small object that vanished where the avatar arrived
        bo, _ = extract_objects(t.before_frame.grid); ao, _ = extract_objects(t.after_frame.grid)
        after_keys = {(o["color"], o["center"]) for o in ao}
        for o in bo:
            if o["size"] <= 60 and (o["color"], o["center"]) not in after_keys and o["color"] not in (aa["color"], ab["color"]):
                if abs(o["center"][1] - aa["center"][1]) <= 6 and abs(o["center"][0] - aa["center"][0]) <= 6:
                    vanish[o["color"]] += 1
    bg = game.frame.background
    for a, (dx, dy) in moves.items():
        if ok[a] or blocked[a] or bad[a]:
            out.append({"kind": "move", "params": {"action": a, "dx": dx, "dy": dy}, "support": ok[a], "counter": bad[a],
                        "text": f"{a} moves the avatar by (dx={dx}, dy={dy}) pixels [{ok[a]} moved, {blocked[a]} blocked]"})
    for c, n in walls.items():
        if c != bg or n >= 2:
            k = entered[c] // 5   # entered cells -> roughly moves (an avatar step covers ~5 cells of the colour)
            out.append({"kind": "wall", "params": {"color": c}, "support": n, "counter": k,
                        "text": f"the avatar cannot enter colour {c} (blocked {n}x" + (f", but entered it {k}x" if k else "") + ")"})
    for a, n in noop.items():
        if a not in moves or ok[a] == 0:
            out.append({"kind": "noop", "params": {"action": a}, "support": n, "counter": pressed[a] - n, "text": f"{a} changes nothing ({n}/{pressed[a]} presses)"})
    g = nav.gauge() if nav else None
    if g:
        out.append({"kind": "gauge", "params": {"color": g["color"], "per_action": g["per_action"]}, "support": 3, "counter": 0,
                    "text": f"the gauge (colour {g['color']}, edge strip) changes by {g['per_action']} per action" + (f"; about {g['actions_left']} actions left" if g.get("actions_left") is not None else "")})
    floor = nav.floor_colors if nav else set()
    for c, n in vanish.items():
        if c != bg and c not in floor:
            out.append({"kind": "collect", "params": {"color": c}, "support": n, "counter": 0, "text": f"touching a colour-{c} object makes it disappear ({n}x)"})
    for c, n in hazard.items():
        out.append({"kind": "hazard", "params": {"color": c}, "support": n, "counter": 0, "text": f"entering colour {c} sends the avatar back to its start ({n}x)"})
    return out

