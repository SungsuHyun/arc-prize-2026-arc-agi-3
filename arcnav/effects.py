"""Layer 1 for click games: a position-relative click-effect model, a goal predicate learned from the winning board, and a
search over click sequences on the learned model (executed with per-step verification).

Model: for clicks on cells of colour c, each offset (dr, dc) within RADIUS carries a partial function before->after colour,
kept when it is consistent across the observed clicks. Far-away changes (HUD counters) are ignored by the simulator.
Goal predicate candidates come from the board just before the level-completing click: colours whose count dropped to zero,
plus the colour of the completing click (a 'submit' button).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

RADIUS = 8


class ClickModel:
    def __init__(self):
        self.obs: dict[tuple[int, int, int], Counter] = defaultdict(Counter)   # (click colour, dr, dc) -> Counter((before, after))
        self.clicks: Counter = Counter()                                      # click colour -> number of observed clicks
        self.far: Counter = Counter()                                         # click colour -> clicks with far (HUD) changes only
        self.rules: dict[int, dict[tuple[int, int], dict[int, int]]] = {}      # colour -> offset -> {before: after}

    def observe(self, before, after, r0: int, c0: int) -> None:
        c = before[r0][c0]; self.clicks[c] += 1
        local = 0
        for r in range(max(0, r0 - RADIUS), min(64, r0 + RADIUS + 1)):
            for cc in range(max(0, c0 - RADIUS), min(64, c0 + RADIUS + 1)):
                if before[r][cc] != after[r][cc]:
                    self.obs[(c, r - r0, cc - c0)][(before[r][cc], after[r][cc])] += 1; local += 1
        if local == 0:
            self.far[c] += 1

    def fit(self, min_support: float = 0.5) -> None:
        self.rules = {}
        for (c, dr, dc), cnt in self.obs.items():
            n = self.clicks[c]
            mapping: dict[int, int] = {}
            ok = True
            by_before: dict[int, Counter] = defaultdict(Counter)
            for (b, a), k in cnt.items():
                by_before[b][a] += k
            total = sum(cnt.values())
            for b, outs in by_before.items():
                a, k = outs.most_common(1)[0]
                if k < 0.7 * sum(outs.values()):
                    ok = False; break
                mapping[b] = a
            if ok and total >= max(1, min_support * n):
                self.rules.setdefault(c, {})[(dr, dc)] = mapping

    def predict(self, board, r0: int, c0: int):
        c = board[r0][c0]
        rules = self.rules.get(c)
        if not rules:
            return None
        out = [row[:] for row in board]; changed = 0
        for (dr, dc), mapping in rules.items():
            r, cc = r0 + dr, c0 + dc
            if 0 <= r < 64 and 0 <= cc < 64 and board[r][cc] in mapping:
                out[r][cc] = mapping[board[r][cc]]; changed += 1
        return out if changed else None

    def summary(self) -> str:
        rows = []
        for c, rules in sorted(self.rules.items()):
            maps = Counter()
            for m in rules.values():
                for b, a in m.items():
                    maps[(b, a)] += 1
            rows.append(f"click colour {c}: {len(rules)} offsets change, transitions {dict(maps.most_common(4))} ({self.clicks[c]} clicks, {self.far[c]} HUD-only)")
        return "\n".join(rows) if rows else "(no consistent click effects)"


def _counts(board, hud_rows: int = 2) -> Counter:
    cnt: Counter = Counter()
    for r in range(hud_rows, 64 - hud_rows):
        for v in board[r][hud_rows:64 - hud_rows]:
            cnt[v] += 1
    return cnt


def learn_goal(attempt: list) -> dict:
    """attempt: transitions of the winning attempt (last one completes the level). Returns predicate candidates."""
    clicks = [t for t in attempt if isinstance(t.action, dict)]
    if not clicks:
        return {}
    final = clicks[-1]
    first = attempt[0].before_frame.grid; pre = final.before_frame.grid
    c0, c1 = _counts(first), _counts(pre)
    zero = sorted(c for c, n in c0.items() if n >= 3 and c1.get(c, 0) == 0)
    submit_colour = pre[final.action["row"]][final.action["col"]]
    return {"zero_colours": zero, "submit_colour": submit_colour, "pre_counts": dict(c1)}


def satisfied(board, goal: dict) -> bool:
    if not goal or not goal.get("zero_colours"):
        return False
    cnt = _counts(board)
    return all(cnt.get(c, 0) == 0 for c in goal["zero_colours"])


def plan_clicks(model: ClickModel, board, goal: dict, candidates: list[tuple[int, int]], max_depth: int = 6, max_nodes: int = 20000) -> Optional[list[tuple[int, int]]]:
    """BFS over click sequences (candidate positions) on the learned model until the goal predicate holds."""
    from collections import deque
    key = lambda b: hash(tuple(tuple(r) for r in b))
    start = [row[:] for row in board]
    if satisfied(start, goal):
        return []
    # monotone removal goals ('every colour-k cell gone'): greedy — click each candidate whose predicted effect removes colour-k cells
    zero = set(goal.get("zero_colours") or [])
    if zero:
        b = start; plan = []
        for _ in range(60):
            cnt = _counts(b); remaining = sum(cnt.get(c, 0) for c in zero)
            if remaining == 0:
                return plan
            best = None
            for (r, c) in candidates:
                nb = model.predict(b, r, c)
                if nb is None:
                    continue
                rem = sum(_counts(nb).get(cc, 0) for cc in zero)
                if rem < remaining and (best is None or rem < best[0]):
                    best = (rem, (r, c), nb)
            if best is None:
                break
            plan.append(best[1]); b = best[2]
        if plan and satisfied(b, goal):
            return plan
    q = deque([(start, [])]); seen = {key(start)}; n = 0
    while q and n < max_nodes:
        b, path = q.popleft(); n += 1
        if len(path) >= max_depth:
            continue
        for (r, c) in candidates:
            nb = model.predict(b, r, c)
            if nb is None:
                continue
            k = key(nb)
            if k in seen:
                continue
            seen.add(k)
            if satisfied(nb, goal):
                return path + [(r, c)]
            q.append((nb, path + [(r, c)]))
    return None


def run_click_model(session, model: ClickModel, goal: dict, *, max_actions: int = 60, log=print) -> dict:
    """Level solver for click games: plan on the model, execute click by click, re-fit the model on surprises and re-plan;
    when the predicate holds, press the submit colour (if any). Bounded by max_actions."""
    level0 = session.level; start = session.actions_used; replans = 0
    while session.actions_used - start < max_actions and session.level == level0 and replans < 6:
        board = session.frame.grid
        nodes = [n for n in session.frame.segmentation["nodes"] if not n["hud"] and n["color"] in model.rules]
        candidates = [(n["center"][0], n["center"][1]) for n in nodes][:40]
        if satisfied(board, goal):
            sub = goal.get("submit_colour")
            btn = [n for n in session.frame.segmentation["nodes"] if not n["hud"] and n["color"] == sub]
            if btn:
                res = session.execute([{"action": "MOUSE", "row": btn[0]["center"][0], "col": btn[0]["center"][1]}])
                return {"completed": session.level > level0, "actions": session.actions_used - start, "reason": "predicate met, submit pressed"}
            return {"completed": False, "actions": session.actions_used - start, "reason": "predicate met but no submit button"}
        plan = plan_clicks(model, board, goal, candidates)
        if not plan:
            return {"completed": False, "actions": session.actions_used - start, "reason": f"no plan on the model ({len(candidates)} candidates)"}
        log(f"click-model: plan of {len(plan)} clicks")
        for (r, c) in plan:
            predicted = model.predict(session.frame.grid, r, c)
            before = session.frame.grid
            res = session.execute([{"action": "MOUSE", "row": r, "col": c}])
            if session.level > level0:
                return {"completed": True, "actions": session.actions_used - start, "reason": "level completed during the plan"}
            if res["game_over"]:
                return {"completed": False, "actions": session.actions_used - start, "reason": "game over"}
            model.observe(before, session.frame.grid, r, c)
            if predicted is None or predicted != session.frame.grid:
                model.fit(); replans += 1
                log("click-model: surprise -> re-fit and re-plan")
                break
    return {"completed": session.level > level0, "actions": session.actions_used - start, "reason": "budget/replans exhausted"}
