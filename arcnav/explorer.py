"""Layer 0: model-free graph exploration of one game level.

The environment is deterministic and cheap; the model is slow and expensive. This module therefore explores
without any model: every visited board (HUD masked) is a node, every macro action tried from it is an edge,
and the next thing to try is always the highest-priority macro that has not been tried at the current node.
When the current node has nothing left, the explorer walks (over known edges) to the nearest node that has,
and when nothing reachable is left it resets the level (the start node is always known).

Macro actions (all generic, no game knowledge):
  goto+interact   walk to an unvisited target object and press the interaction key there (SPACE / ACTION7)
  interact        press an interaction key where the avatar stands
  click           click an object (one per 4x4 cell; smallest objects and one colour per round first)
  frontier        walk to the nearest unexplored walkable block
  key / key-run   press a movement key once / until the board stops changing (max 8)

The result of a level is the macro path that completed it (feeds goals.py / rules.py) plus the graph itself.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .frame import masked_ascii, summarize_diff
from .nav import NavHelper

INTERACT_KEYS = ("SPACE", "ACTION7")
MOVE_KEYS = ("UP", "DOWN", "LEFT", "RIGHT")
CHUNK = 24   # GameSession.execute caps one call at MAX_ACTIONS_PER_CALL


@dataclass
class Macro:
    kind: str                 # goto+interact | interact | click | frontier | key | key-run
    actions: list[dict]       # concrete engine actions
    label: str
    priority: int             # lower runs first

    def key(self) -> str:
        return self.label


@dataclass
class Node:
    state: str
    level: int
    tried: set = field(default_factory=set)          # macro labels tried here
    edges: dict = field(default_factory=dict)        # macro label -> (actions, result state | "GAME_OVER" | "LEVEL")
    macros_cache: Optional[list] = None


class Explorer:
    def __init__(self, session, *, log=print, max_actions_per_level: int = 300, max_seconds: float = 300.0):
        self.s = session
        self.log = log
        self.max_actions_per_level = max_actions_per_level
        self.max_seconds = max_seconds
        self.nodes: dict[str, Node] = {}
        self.hazard_labels: set[str] = set()     # (state, label) pairs that ended in game over
        self.solutions: dict[int, list[str]] = {}
        self.trace: list[str] = []
        self.label_hist: dict[tuple[int, str], list[int]] = {}   # (level, label) -> [tried, produced-same-state]
        self.frontier_actions: dict[int, int] = {}
        self.inv_tried: set = set()   # (level, inventory signature, label): a macro is tried once per world state, not once per avatar position
        self.effect_stats: dict[tuple[str, int], list[int]] = {}   # (kind, colour) -> [tries, effective]: which kinds of touches did anything, across levels

    def inv_sig(self) -> str:
        """World inventory: non-HUD objects except the avatar's own colours. Moving the avatar does not change it; collecting,
        toggling or opening something does. Novelty is judged against this, so 'goto(target)' is retried only after the world changed."""
        nav = self._nav(); skip_cols = set()
        try:
            av = nav.avatar() if nav else None
            skip_cols |= set(av["colors"]) if av else set()
            g = nav.gauge() if nav else None
            if g:
                skip_cols.add(g["color"])
        except Exception:
            pass
        items = []
        for n in self.s.frame.segmentation["nodes"]:
            if n["hud"] or n["color"] in skip_cols:
                continue
            r0, c0, r1, c1 = n["bbox"]
            thin = (r1 - r0 <= 2) or (c1 - c0 <= 2)
            if thin and (r0 <= 1 or r1 >= 62 or c0 <= 1 or c1 >= 62):
                continue   # gauge / counter pieces along the frame edge (their emptied part is not flagged as HUD)
            items.append((n["color"], n["pixels"], (r0, c0, r1, c1)))
        return f"L{self.s.level}:" + str(hash(tuple(sorted(items))))

    KIND_RANK = {"goto+interact": 0, "goto": 0, "interact": 1, "click": 2, "frontier": 3, "key": 4, "key-run": 5}
    KIND_RANK_MOVEMENT = {"goto+interact": 0, "goto": 0, "interact": 1, "frontier": 2, "key": 3, "key-run": 4, "click": 6}

    def goal_colors(self) -> dict[int, int]:
        """Colours that mattered in earlier levels' wins (from goals.py hypotheses), with a rank: the goal colour first,
        collectibles / clicked colours next. Soft transfer: they are tried first on the new level, nothing is replayed blindly."""
        out: dict[int, int] = {}
        for h in self.s.goal_hypotheses:
            if h.get("level", 0) >= self.s.level:
                continue
            for c in ([h.get("reach")] if h.get("reach") is not None else []):
                out.setdefault(int(c), 0)
            for c in ([h.get("collect")] if h.get("collect") is not None else []):
                out.setdefault(int(c), 1)
            for i, c in enumerate(dict.fromkeys(h.get("sequence") or [])):
                out.setdefault(int(c), 2 + i)
        return out

    def _macro_color(self, m: Macro) -> Optional[int]:
        import re
        mt = re.match(r"(?:goto|click)\((\d+)@", m.label)
        return int(mt.group(1)) if mt else None

    def _rank(self, level: int, m: Macro) -> tuple:
        """Colours that won earlier levels first, then goal-directed kinds (after the world changed, re-trying a target is worth
        more than mapping another corridor), then never-tried labels before tried ones, dead labels (same state twice) last."""
        h = self.label_hist.get((level, m.label), [0, 0])
        ranks = self.KIND_RANK_MOVEMENT if self._movement_game() else self.KIND_RANK
        dead = h[1] >= (1 if m.kind == "click" else 2)   # a click that changed nothing at all (HUD included) once is a dead button
        gc = self.goal_colors(); c = self._macro_color(m)
        goal_rank = gc.get(c, 99) if c is not None else 99
        eff = self._effect_ratio(m)   # colours whose touches did something on earlier levels first; known no-ops last
        return (1 if dead else 0, 0 if goal_rank < 99 else 1, goal_rank, ranks.get(m.kind, 9), 1 if eff < 0.2 else 0, -round(eff, 1), 1 if h[0] > 0 else 0, m.priority)

    def _movement_game(self) -> bool:
        nav = self._nav()
        return bool(nav and nav.moves and any(a in self.s.valid_actions for a in MOVE_KEYS))

    # ── state ────────────────────────────────────────────────────────────
    def state(self) -> str:
        return f"L{self.s.level}:" + str(hash(masked_ascii(self.s.frame)))

    def node(self) -> Node:
        st = self.state()
        n = self.nodes.get(st)
        if n is None:
            n = Node(st, self.s.level); self.nodes[st] = n
        return n

    # ── macro generation ─────────────────────────────────────────────────
    def _nav(self) -> Optional[NavHelper]:
        key = (len(self.s.host_transitions), self.s.attempt_start_index, id(self.s.frame))
        cached = getattr(self, "_nav_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        cur = [t for t in self.s.host_transitions[self.s.attempt_start_index:]
               if t.before_frame.level == t.after_frame.level == self.s.level]
        try:
            nav = NavHelper(cur, self.s.frame)
        except Exception:
            nav = None
        self._nav_cache = (key, nav)
        return nav

    def macros(self, node: Node) -> list[Macro]:
        valid = list(self.s.valid_actions)
        inter = [a for a in INTERACT_KEYS if a in valid]
        keys = [a for a in MOVE_KEYS if a in valid]
        out: list[Macro] = []
        nav = self._nav()
        cur = [t for t in self.s.host_transitions[self.s.attempt_start_index:] if t.after_frame.level == self.s.level]
        if nav and keys and nav.moves and nav.avatar():
            # targets: unvisited first, nearest first
            tg = [t for t in nav.targets(max_n=14) if t.get("path_len") is not None]
            tg.sort(key=lambda t: (bool(t.get("visited")), t["path_len"]))
            for i, t in enumerate(tg[:8]):
                path = nav.path_to(t["row"], t["col"])
                if path is None or len(path) > 30:
                    continue
                acts = [{"action": a} for a in path]
                if inter:
                    out.append(Macro("goto+interact", acts + [{"action": inter[0]}],
                                     f"goto({t['color']}@{t['row']},{t['col']})+{inter[0]}", 10 + i + (40 if t.get("visited") else 0)))
                else:
                    out.append(Macro("goto", acts, f"goto({t['color']}@{t['row']},{t['col']})", 10 + i + (40 if t.get("visited") else 0)))
            fr = nav.frontier()
            if fr is not None:
                path = nav.path_to(fr[0], fr[1])
                if path:
                    out.append(Macro("frontier", [{"action": a} for a in path], f"frontier({fr[0]},{fr[1]})", 30))
        for a in inter:
            out.append(Macro("interact", [{"action": a}], f"{a}", 20))
        if "MOUSE" in valid and self.s.frame is not None:
            clicked = {(t.action["row"] // 4, t.action["col"] // 4) for t in cur if isinstance(t.action, dict)}
            nodes = []
            for n in self.s.frame.segmentation["nodes"]:
                if n["hud"]:
                    continue
                r0, c0, r1, c1 = n["bbox"]
                thin = (r1 - r0 <= 2) or (c1 - c0 <= 2)
                if thin and (r0 <= 1 or r1 >= 62 or c0 <= 1 or c1 >= 62):
                    continue   # edge strips (gauge / counters) are never click targets
                if n["pixels"] <= 2 and keys:
                    continue   # 1-2 px specks in a movement game are marks, not buttons
                nodes.append(n)
            nodes.sort(key=lambda n: n["pixels"])
            click_base = 25 if not (keys and nav and nav.moves) else 70   # in a movement game clicking is the last resort
            seen_colors: set = set(); seen_cells: set = set()
            for j, n in enumerate(nodes[:60]):
                cell = (n["center"][0] // 4, n["center"][1] // 4)
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                first_of_color = n["color"] not in seen_colors
                seen_colors.add(n["color"])
                pr = click_base + (0 if first_of_color else 10) + (20 if cell in clicked else 0) + j // 8
                out.append(Macro("click", [{"action": "MOUSE", "row": n["center"][0], "col": n["center"][1]}],
                                 f"click({n['color']}@{n['center'][0]},{n['center'][1]})", pr))
        for a in keys:
            out.append(Macro("key", [{"action": a}], f"{a}", 50))
        for a in keys:
            out.append(Macro("key-run", [{"action": a}] * 8, f"{a}x8", 60))
        out.sort(key=lambda m: m.priority)
        return out

    # ── execution ────────────────────────────────────────────────────────
    def _execute(self, actions: list[dict]) -> dict:
        """Execute in chunks; stop at level completion / game over / invalid action."""
        res_all = {"executed_count": 0, "board_changed": False, "level_completed": False, "game_over": False, "stopped_reason": None}
        for i in range(0, len(actions), CHUNK):
            r = self.s.execute(actions[i:i + CHUNK])
            res_all["executed_count"] += r["executed_count"]; res_all["board_changed"] |= r["board_changed"]
            res_all["level_completed"] |= r["level_completed"]; res_all["game_over"] |= r["game_over"]
            res_all["stopped_reason"] = r["stopped_reason"]
            if r["level_completed"] or r["game_over"] or (r["stopped_reason"] and "only the first" not in r["stopped_reason"]):
                break
            if r["executed_count"] == 0:
                break
        return res_all

    def _effect_ratio(self, m: Macro) -> float:
        c = self._macro_color(m)
        if c is None:
            return 0.5
        st = self.effect_stats.get((m.kind, c))
        return 0.5 if not st or st[0] == 0 else st[1] / st[0]

    def _run_macro(self, node: Node, m: Macro) -> dict:
        actions = m.actions
        full_before = self.s.frame.ascii if self.s.frame is not None else None
        masked_before = masked_ascii(self.s.frame) if self.s.frame is not None else None
        if m.kind == "key-run":   # press until the board stops changing
            done = 0; last = None
            for _ in range(8):
                before = masked_ascii(self.s.frame)
                r = self.s.execute([actions[0]]); done += r["executed_count"]
                last = r
                if r["level_completed"] or r["game_over"] or masked_ascii(self.s.frame) == before or r["executed_count"] == 0:
                    break
            r = dict(last or {"executed_count": 0, "board_changed": False, "level_completed": False, "game_over": False, "stopped_reason": None})
            r["executed_count"] = done
        else:
            r = self._execute(actions)
        node.tried.add(m.label)
        result = "LEVEL" if r["level_completed"] else "GAME_OVER" if r["game_over"] else self.state()
        node.edges[m.label] = (actions, result)
        h = self.label_hist.setdefault((node.level, m.label), [0, 0]); h[0] += 1
        # effect grade: 2 = the masked world changed, 1 = only the HUD/counter changed (progress can hide there: vc33 pumps), 0 = nothing
        full_changed = self.s.frame is not None and full_before is not None and self.s.frame.ascii != full_before
        world_changed = self.s.frame is not None and masked_before is not None and masked_ascii(self.s.frame) != masked_before
        grade = 2 if (r["level_completed"] or world_changed) else 1 if full_changed else 0
        c = self._macro_color(m)
        if c is not None:
            st = self.effect_stats.setdefault((m.kind, c), [0, 0]); st[0] += 2; st[1] += grade   # ratio in [0, 1]
        if grade == 0:
            h[1] += 1   # nothing at all changed: a dead button
        if r["game_over"]:
            self.hazard_labels.add((node.state, m.label))
        self.trace.append(f"{m.label} -> {result if result in ('LEVEL', 'GAME_OVER') else ('same' if result == node.state else 'new' if result not in self.nodes else 'known')}")
        return r

    # ── search for something untried ─────────────────────────────────────
    def _plan_to_untried(self) -> Optional[list[tuple[str, list[dict], str]]]:
        """BFS over known edges to the nearest node that still has an untried macro. Returns the edge path."""
        start = self.state()
        prev: dict[str, tuple[Optional[str], Optional[str]]] = {start: (None, None)}
        q = deque([start])
        while q:
            st = q.popleft()
            n = self.nodes.get(st)
            if n is None:
                continue
            if st != start:
                ms = self.macros(n) if st == self.state() else None   # macros depend on the live frame: only evaluable at the live node
                # for non-live nodes fall back to the cached list generated when we were there
                ms = ms if ms is not None else (n.macros_cache or [])
                if any(m.label not in n.tried for m in ms):
                    path = []; cur = st
                    while prev[cur][0] is not None:
                        p, lab = prev[cur]; path.append((p, self.nodes[p].edges[lab][0], lab)); cur = p
                    return list(reversed(path))
            for lab, (acts, res) in n.edges.items():
                if res in ("LEVEL", "GAME_OVER") or res in prev:
                    continue
                prev[res] = (st, lab); q.append(res)
        return None

    # ── main loop ────────────────────────────────────────────────────────
    def play_level(self) -> dict:
        level = self.s.level
        t0 = time.time(); a0 = self.s.actions_used; macros_run = 0; resets = 0
        path_labels: list[str] = []
        while self.s.level == level and self.s.state != "WIN":
            if self.s.actions_used - a0 >= self.max_actions_per_level or time.time() - t0 > self.max_seconds:
                return {"level": level, "completed": False, "actions": self.s.actions_used - a0, "macros": macros_run, "resets": resets, "reason": "budget"}
            node = self.node()
            ms = self.macros(node); node.macros_cache = ms
            inv = self.inv_sig()   # world inventory (no avatar, gauge or edge strips): empirically better than the raw board for click games too (r11l, vc33)
            untried = [m for m in ms if m.label not in node.tried and (node.state, m.label) not in self.hazard_labels
                       and (level, inv, m.label) not in self.inv_tried]
            if self.frontier_actions.get(level, 0) > 0.4 * self.max_actions_per_level:
                untried = [m for m in untried if m.kind != "frontier"] or untried
            if untried:
                m = min(untried, key=lambda m: self._rank(level, m))   # never-tried-on-this-level labels first, dead labels last
                self.inv_tried.add((level, inv, m.label))
                r = self._run_macro(node, m); macros_run += 1
                if m.kind == "frontier":
                    self.frontier_actions[level] = self.frontier_actions.get(level, 0) + r["executed_count"]
                if r["level_completed"]:
                    path_labels.append(m.label)
                    self.solutions[level] = path_labels
                    return {"level": level, "completed": True, "actions": self.s.actions_used - a0, "macros": macros_run, "resets": resets, "path": path_labels}
                if r["game_over"]:
                    path_labels = []; resets += 1
                    continue
                if r["executed_count"] == 0:
                    node.tried.add(m.label)   # invalid here: never again at this node
                    continue
                path_labels.append(m.label)
                continue
            plan = self._plan_to_untried()
            if plan:
                ok = True
                for st, acts, lab in plan:
                    if self.state() != st:
                        ok = False; break
                    r = self._execute(acts)
                    path_labels.append(lab)
                    if r["level_completed"]:
                        self.solutions[level] = path_labels
                        return {"level": level, "completed": True, "actions": self.s.actions_used - a0, "macros": macros_run, "resets": resets, "path": path_labels}
                    if r["game_over"]:
                        ok = False; path_labels = []; resets += 1; break
                if not ok:
                    self.trace.append("replay diverged")
                continue
            # nothing reachable is untried: reset the level and continue from the start node (its edges are known)
            if resets >= 3:
                return {"level": level, "completed": False, "actions": self.s.actions_used - a0, "macros": macros_run, "resets": resets, "reason": "exhausted"}
            self.s.reset(); resets += 1; path_labels = []
            self.s.attempt_start_index = len(self.s.host_transitions)
            self.trace.append("reset (nothing untried reachable)")
            start = self.node()
            if all(m.label in start.tried for m in self.macros(start)):
                return {"level": level, "completed": False, "actions": self.s.actions_used - a0, "macros": macros_run, "resets": resets, "reason": "exhausted"}
        return {"level": level, "completed": self.s.level > level, "actions": self.s.actions_used - a0, "macros": macros_run, "resets": resets, "path": path_labels}

    def _layer1(self) -> Optional[dict]:
        """Before exploring a new level, replay what won the previous one (goal hypotheses from goals.py): reach/collect
        via the nav planner, click games via the recorded colour sequence. Cheap, deterministic, and it is exactly the
        knowledge transfer the model never managed."""
        from . import autopilot
        level0 = self.s.level; a0 = self.s.actions_used
        # rule-based autopilots (two-body merge when a mirrored body is confirmed) need a few observed moves first
        keys = [a for a in MOVE_KEYS if a in self.s.valid_actions]
        if keys and not [t for t in self.s.host_transitions[self.s.attempt_start_index:] if t.after_frame.level == level0]:
            self._execute([{"action": a} for a in keys])   # one probe per key: learns controls and feeds rule induction
            if self.s.level > level0:
                return {"level": level0, "completed": True, "actions": self.s.actions_used - a0, "macros": 0, "resets": 0, "path": ["probe"]}
        try:
            for _ in range(2):
                if self.s._maybe_autopilot() and self.s.level > level0:
                    self.trace.append("layer1 rules-autopilot completed the level")
                    return {"level": level0, "completed": True, "actions": self.s.actions_used - a0, "macros": 0, "resets": 0, "path": ["layer1:rules-autopilot"]}
        except Exception as e:
            self.trace.append(f"layer1 autopilot error {type(e).__name__}")
        hyps = [h for h in self.s.goal_hypotheses if h.get("level", 0) < self.s.level]
        if not hyps:
            return None
        for h in hyps[:3]:
            try:
                if h.get("type") in ("reach", "collect_reach", "collect_all"):
                    r = autopilot.run_reach(self.s, h, max_actions=80, log=lambda m: None)
                elif h.get("type") == "click_sequence":
                    r = autopilot.run_click_sequence(self.s, h, max_actions=40, log=lambda m: None)
                else:
                    continue
            except Exception as e:
                self.trace.append(f"layer1 error {type(e).__name__}"); continue
            self.trace.append(f"layer1 {h.get('type')}: {r.get('reason')} ({r.get('actions')} actions)")
            if r.get("completed") or self.s.level > level0:
                return {"level": level0, "completed": True, "actions": self.s.actions_used - a0, "macros": 0, "resets": 0, "path": [f"layer1:{h.get('type')}"]}
        return None

    def play(self, max_levels: int = 3, client=None, advisor_rounds: int = 6) -> list[dict]:
        out = []
        for _ in range(max_levels):
            if self.s.state == "WIN":
                break
            r = self._layer1()
            if r is None and client is not None and self.s.level > 1:
                from .advisor import advise_level
                level0 = self.s.level; a0 = self.s.actions_used
                adv = advise_level(self, client, rounds=advisor_rounds, log=self.log)
                if adv.get("completed"):
                    r = {"level": level0, "completed": True, "actions": self.s.actions_used - a0, "macros": 0, "resets": 0, "path": [f"advisor:{adv.get('hypothesis', '')[:60]}"]}
            if r is None:
                r = self.play_level()
            out.append(r)
            self.log(f"[explorer] level {r['level']}: {'COMPLETED' if r['completed'] else 'not completed'} in {r['actions']} actions, {r['macros']} macros, {r['resets']} resets" + (f" — {r.get('reason')}" if not r["completed"] else f" — path {r.get('path')}"))
            if not r["completed"]:
                break
        return out
