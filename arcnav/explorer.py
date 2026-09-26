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

    def _rank(self, level: int, m: Macro) -> tuple:
        h = self.label_hist.get((level, m.label), [0, 0])
        return (1 if h[1] >= 2 else 0, 1 if h[0] > 0 else 0, m.priority)

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
        cur = [t for t in self.s.host_transitions[self.s.attempt_start_index:]
               if t.before_frame.level == t.after_frame.level == self.s.level]
        try:
            return NavHelper(cur, self.s.frame)
        except Exception:
            return None

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
            nodes = [n for n in self.s.frame.segmentation["nodes"] if not n["hud"]]
            nodes.sort(key=lambda n: n["pixels"])
            seen_colors: set = set(); seen_cells: set = set()
            for j, n in enumerate(nodes[:60]):
                cell = (n["center"][0] // 4, n["center"][1] // 4)
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                first_of_color = n["color"] not in seen_colors
                seen_colors.add(n["color"])
                pr = 25 + (0 if first_of_color else 10) + (20 if cell in clicked else 0) + j // 8
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

    def _run_macro(self, node: Node, m: Macro) -> dict:
        actions = m.actions
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
        if result == node.state:
            h[1] += 1
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
            untried = [m for m in ms if m.label not in node.tried and (node.state, m.label) not in self.hazard_labels]
            if untried:
                m = min(untried, key=lambda m: self._rank(level, m))   # never-tried-on-this-level labels first, dead labels last
                r = self._run_macro(node, m); macros_run += 1
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

    def play(self, max_levels: int = 3) -> list[dict]:
        out = []
        for _ in range(max_levels):
            if self.s.state == "WIN":
                break
            r = self.play_level(); out.append(r)
            self.log(f"[explorer] level {r['level']}: {'COMPLETED' if r['completed'] else 'not completed'} in {r['actions']} actions, {r['macros']} macros, {r['resets']} resets" + (f" — {r.get('reason')}" if not r["completed"] else f" — path {r.get('path')}"))
            if not r["completed"]:
                break
        return out
