"""Autopilot for two-body (mirrored) merge games in cell space: lattice from the body, plan on cells,
execute step by step, verify against the prediction, refine passable colours, replan. No model calls."""
from __future__ import annotations

from typing import Optional

from .frame import texture_colors
from .nav import extract_objects
from .planner import lattice_from_body, cell_of, cell_colors, two_body_merge_cells

TF = {"same": (1, 1), "mirror_x": (1, -1), "mirror_y": (-1, 1), "mirror_xy": (-1, -1)}
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def _bodies(grid, color, min_size=4):
    objs, _ = extract_objects(grid)
    return sorted([o for o in objs if o["color"] == color and o["size"] >= min_size], key=lambda o: (o["center"][1], o["center"][0]))


def run_two_body(session, *, body_color: int, transform: str, floor_colors: set, max_replans: int = 8, max_actions: int = 120, log=print) -> dict:
    level0 = session.level; start_actions = session.actions_used
    grid = session.frame.grid; tex = texture_colors(grid)
    passable = set(floor_colors) | {body_color}; hazards = set(tex); forbidden = set(); blocked_cells = set()
    replans = 0
    while replans <= max_replans and session.actions_used - start_actions < max_actions:
        grid = session.frame.grid; b = _bodies(grid, body_color)
        if len(b) != 2:
            return {"completed": False, "actions": session.actions_used - start_actions, "replans": replans, "reason": f"{len(b)} bodies visible"}
        rows, cols = lattice_from_body(b[0]["bbox"])
        cells = cell_colors(grid, rows, cols)
        A, B = cell_of(b[0]["bbox"], rows, cols), cell_of(b[1]["bbox"], rows, cols)
        plan = two_body_merge_cells(cells, A, B, transform, passable, hazards, forbidden=forbidden, blocked=blocked_cells)
        log(f"autopilot: replan {replans}: lattice {len(rows)-1}x{len(cols)-1}, A{A} B{B}, passable={sorted(passable)}, hazards={sorted(hazards)}, forbidden={sorted(forbidden)}, plan={plan}")
        if not plan:
            return {"completed": False, "actions": session.actions_used - start_actions, "replans": replans, "reason": "no plan"}
        replans += 1
        tf = TF[transform]
        for k, act in enumerate(plan):
            dr, dc = MOVES[act]
            def stp(pos, r_, c_):
                r, c = pos[0] + r_, pos[1] + c_
                if not (0 <= r < len(cells) and 0 <= c < len(cells[0])):
                    return pos
                if cells[r][c][1] & hazards:
                    return (r, c)
                if not cells[r][c][1] <= passable or (r, c) in blocked_cells:
                    return pos
                return (r, c)
            pa, pb = stp(A, dr, dc), stp(B, tf[0] * dr, tf[1] * dc)
            res = session.execute([{"action": act}])
            if res["level_completed"] or session.level > level0:
                return {"completed": True, "actions": session.actions_used - start_actions, "replans": replans, "reason": "level completed"}
            if res["game_over"]:
                return {"completed": False, "actions": session.actions_used - start_actions, "replans": replans, "reason": "game over"}
            grid = session.frame.grid; b2 = _bodies(grid, body_color)
            if len(b2) != 2:   # bodies touched and segment as one: finish the next planned steps blindly
                rest = plan[k + 1:][:3]
                if rest:
                    res = session.execute([{"action": x} for x in rest])
                    if res["level_completed"] or session.level > level0:
                        return {"completed": True, "actions": session.actions_used - start_actions, "replans": replans, "reason": "level completed"}
                break
            cells = cell_colors(grid, rows, cols)
            c1, c2 = cell_of(b2[0]["bbox"], rows, cols), cell_of(b2[1]["bbox"], rows, cols)
            # identity: the bodies may cross; assign observed cells to predictions by total distance
            d = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])
            A2, B2 = ((c1, c2) if d(c1, pa) + d(c2, pb) <= d(c2, pa) + d(c1, pb) else (c2, c1))
            if A2 == pa and B2 == pb:
                A, B = A2, B2
                continue
            log(f"autopilot: divergence on {act}: predicted A{pa} B{pb}, actual A{A2} B{B2}")
            jumped = [(pred, act_pos, body, d) for pred, act_pos, body, d in ((pa, A2, A, (dr, dc)), (pb, B2, B, (tf[0] * dr, tf[1] * dc)))
                      if act_pos != pred and act_pos != body and act_pos != (body[0] + d[0], body[1] + d[1])]
            if jumped:   # a reset: blame the body whose target cell shows hazard texture; if none does, blame both
                blamed = [(body, d) for _, _, body, d in jumped
                          if 0 <= body[0] + d[0] < len(cells) and 0 <= body[1] + d[1] < len(cells[0]) and (cells[body[0] + d[0]][body[1] + d[1]][1] & hazards)]
                if not blamed:
                    blamed = [(body, d) for _, _, body, d in jumped]
                for body, d in blamed:
                    forbidden.add((body[0] + d[0], body[1] + d[1]))
                log(f"autopilot: reset; forbidding {sorted(forbidden)}")
                break
            for pred, act_pos, body, (r_, c_) in ((pa, A2, A, (dr, dc)), (pb, B2, B, (tf[0] * dr, tf[1] * dc))):
                if act_pos == pred:
                    continue
                target = (body[0] + r_, body[1] + c_)
                in_grid = 0 <= target[0] < len(cells) and 0 <= target[1] < len(cells[0])
                if act_pos == body and in_grid:        # blocked where we predicted a move
                    extra = cells[target[0]][target[1]][1] - passable
                    if extra:
                        log(f"autopilot: colours {sorted(extra)} block")   # already implied by the whitelist; nothing to change
                    else:
                        blocked_cells.add(target); log(f"autopilot: cell {target} blocks although it looks like floor")
                elif act_pos == target and in_grid:    # moved where we predicted a block
                    passable |= cells[target[0]][target[1]][1]; blocked_cells.discard(target); log(f"autopilot: colours {sorted(cells[target[0]][target[1]][1])} are passable")
                elif in_grid:
                    forbidden.add(target); log(f"autopilot: unexpected position after entering {target}; forbidding it")
            break
    return {"completed": False, "actions": session.actions_used - start_actions, "replans": replans, "reason": "budget exhausted"}



def run_reach(session, hypothesis: dict, *, max_actions: int = 60, log=print) -> dict:
    """Single-avatar autopilot for 'reach' / 'collect_reach' / 'collect_all' hypotheses: pick the next target from the
    hypothesis (collectibles nearest-first, then the goal colour), route with nav.path_to over known-walkable cells,
    execute at most 12 steps at a time, re-plan from the live board. Stops on level completion, game over, no path,
    or no movement. Never spends more than max_actions."""
    from .nav import NavHelper
    level0 = session.level; start = session.actions_used; stalls = 0; visited_targets = set()
    kind = hypothesis.get("type"); goal_c = hypothesis.get("reach"); coll_c = hypothesis.get("collect")
    while session.actions_used - start < max_actions and session.level == level0:
        cur = [t for t in session.host_transitions if t.before_frame.level == t.after_frame.level == session.level]
        nav = NavHelper(cur, session.frame)
        if not nav.moves or not nav.avatar():
            return {"completed": False, "actions": session.actions_used - start, "reason": "movement not learned"}
        targets = [t for t in nav.targets(max_n=20) if t.get("path_len")]
        want = None
        if kind in ("collect_reach", "collect_all") and coll_c is not None:
            left = [t for t in targets if t["color"] == coll_c and (t["row"], t["col"]) not in visited_targets]
            want = min(left, key=lambda t: t["path_len"]) if left else None
        if want is None and kind in ("reach", "collect_reach") and goal_c is not None:
            goals = [t for t in targets if t["color"] == goal_c]
            want = min(goals, key=lambda t: t["path_len"]) if goals else None
        if want is None:
            return {"completed": False, "actions": session.actions_used - start, "reason": "no reachable target for the hypothesis"}
        path = nav.path_to(want["row"], want["col"])
        if not path:
            return {"completed": False, "actions": session.actions_used - start, "reason": "no path"}
        before = nav.avatar(); step = path[:12]
        log(f"reach-autopilot: target colour {want['color']} at ({want['row']},{want['col']}), {len(path)} moves, executing {len(step)}")
        res = session.execute([{"action": a} for a in step])
        if res["level_completed"] or session.level > level0:
            return {"completed": True, "actions": session.actions_used - start, "reason": "level completed"}
        if res["game_over"]:
            return {"completed": False, "actions": session.actions_used - start, "reason": "game over"}
        after = NavHelper([t for t in session.host_transitions if t.before_frame.level == t.after_frame.level == session.level], session.frame).avatar()
        if after and before and (after["row"], after["col"]) == (before["row"], before["col"]):
            stalls += 1
            if stalls >= 2:
                return {"completed": False, "actions": session.actions_used - start, "reason": "avatar did not move (blocked target?)"}
        else:
            stalls = 0
        if len(step) == len(path):
            visited_targets.add((want["row"], want["col"]))
    return {"completed": False, "actions": session.actions_used - start, "reason": "budget exhausted" if session.level == level0 else "level completed"}
