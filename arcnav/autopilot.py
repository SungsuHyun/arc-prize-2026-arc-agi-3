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
    return sorted([o for o in objs if o["color"] == color and o["size"] >= min_size
                   and o["bbox"][2] - o["bbox"][0] >= 1 and o["bbox"][3] - o["bbox"][1] >= 1],   # not a 1-px-thin HUD strip
                  key=lambda o: (o["center"][1], o["center"][0]))


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


def run_click_sequence(session, hypothesis: dict, *, max_actions: int = 40, log=print) -> dict:
    """Click-game autopilot for a 'click_sequence' hypothesis learned on an earlier level: click objects of the recorded
    colours in the same order on the new level (positions differ; the nearest object of that colour to the previous click
    is chosen). Stops on level completion, game over, a missing colour, or the action budget."""
    level0 = session.level; start = session.actions_used
    seq = list(hypothesis.get("sequence") or [])
    if not seq or "MOUSE" not in session.valid_actions:
        return {"completed": False, "actions": 0, "reason": "not a click game / empty sequence"}
    # compress the recorded sequence into runs: a run of >= 2 clicks on colour c generalises to "click every colour-c object"
    runs: list[list] = []
    for c in seq:
        if runs and runs[-1][0] == c:
            runs[-1][1] += 1
        else:
            runs.append([c, 1])
    last = None
    for color, count in runs:
        clicked_cells: set = set()
        todo = 64 if count >= 2 else 1
        for _ in range(todo):
            if session.actions_used - start >= max_actions or session.level != level0:
                break
            nodes = [n for n in session.frame.segmentation["nodes"] if not n["hud"] and n["color"] == color
                     and (n["center"][0] // 4, n["center"][1] // 4) not in clicked_cells]
            if not nodes:
                if todo == 1:
                    return {"completed": False, "actions": session.actions_used - start, "reason": f"no colour-{color} object on this level"}
                break   # every colour-c object clicked: next run
            if last is not None:
                nodes.sort(key=lambda n: abs(n["center"][0] - last[0]) + abs(n["center"][1] - last[1]))
            else:
                nodes.sort(key=lambda n: n["pixels"])
            n = nodes[0]; last = n["center"]; clicked_cells.add((n["center"][0] // 4, n["center"][1] // 4))
            res = session.execute([{"action": "MOUSE", "row": n["center"][0], "col": n["center"][1]}])
            if res["level_completed"] or session.level > level0:
                return {"completed": True, "actions": session.actions_used - start, "reason": "level completed"}
            if res["game_over"]:
                return {"completed": False, "actions": session.actions_used - start, "reason": "game over"}
    return {"completed": False, "actions": session.actions_used - start, "reason": "sequence replayed without completion"}


def run_goal_search(session, *, goal_colors: list, collect_colors: list, refill_colors: list, hazard_colors: set,
                    max_actions: int = 120, log=print, refill_amounts: Optional[dict] = None) -> dict:
    """Generic movement-game level solver on induced knowledge (no game-specific code):
    1. go to the nearest goal-colour object (a door bumped earlier is still allowed as the final step);
    2. if arriving did not finish the level or no goal is reachable, collect the known collectible colours nearest-first
       (keys open doors), refuel at refill colours when the gauge is short, then retry the goal;
    3. stop on completion, game over, budget, or two rounds without any progress."""
    from .nav import NavHelper
    level0 = session.level; start = session.actions_used
    visited: set = set(); stale_rounds = 0
    refill_amounts = dict(refill_amounts or {})

    def nav_now():
        cur = [t for t in session.host_transitions if t.before_frame.level == t.after_frame.level == session.level]
        return NavHelper(cur, session.frame)

    def go(nav, t, final_any=False) -> Optional[dict]:
        path = nav.path_to(t["row"], t["col"], ignore_wall_target=final_any)
        if not path:
            return None
        g = nav.gauge()
        if g and g.get("actions_left") is not None and len(path) > int(g["actions_left"]) - 1:
            return {"short": True}
        res = session.execute([{"action": a} for a in path[:24]])
        return res

    def targets(nav, colors):
        return sorted([t for t in nav.targets(max_n=30) if t["color"] in colors and (t["row"], t["col"]) not in visited],
                      key=lambda t: (t.get("path_len") is None, t.get("path_len") or 999))

    def done(d):
        d["refill_colors"] = list(refill_colors); d["refill_amounts"] = dict(refill_amounts); return d
    while session.actions_used - start < max_actions and session.level == level0:
        nav = nav_now()
        if not nav.moves or not nav.avatar():
            return done({"completed": False, "actions": session.actions_used - start, "reason": "movement not learned"})
        progressed = False
        # refuel first when the gauge is nearly empty
        g = nav.gauge()
        need = min([t["path_len"] for t in nav.targets(max_n=30) if t["color"] in set(goal_colors) | set(collect_colors) and t.get("path_len")] or [6])
        if g and g.get("actions_left") is not None and int(g["actions_left"]) <= max(6, need + 2) and refill_colors:
            for t in targets(nav, set(refill_colors))[:2]:
                r = go(nav, t)
                if r and not r.get("short") and r["executed_count"]:
                    visited.add((t["row"], t["col"])); progressed = True
                    log(f"goal-search: refuel at colour {t['color']}"); break
            if session.level != level0:
                break
            nav = nav_now()
        # 0. a tight gauge with unknown objects around: learn the nearest unknown first (it is often the refill this level adds)
        if g and g.get("actions_left") is not None and int(g["actions_left"]) < 2 * need and not refill_colors:
            known = set(goal_colors) | set(collect_colors) | set(hazard_colors)
            unknown = sorted([t for t in nav.targets(max_n=30) if t["color"] not in known and t.get("path_len") and (t["row"], t["col"]) not in visited],
                             key=lambda t: t["path_len"])
            for t in unknown[:1]:
                r = go(nav, t)
                if r and not r.get("short") and r["executed_count"]:
                    visited.add((t["row"], t["col"])); progressed = True
                    g2 = nav_now().gauge()
                    log(f"goal-search: tight gauge ({g['actions_left']} left, need {need}); probed unknown colour {t['color']} -> gauge {g.get('size')}->{g2 and g2.get('size')}")
                    if session.level > level0:
                        return done({"completed": True, "actions": session.actions_used - start, "reason": "probe completed the level"})
                    if r["game_over"]:
                        return done({"completed": False, "actions": session.actions_used - start, "reason": "game over"})
                    # measured refill: what the gauge has beyond the plain decrement over the walk, in actions
                    try:
                        per = abs(float(g.get("per_action") or 1)); steps = r["executed_count"]
                        gained_px = (g2.get("size", 0) if g2 else 0) - (g.get("size", 0) - steps * per)
                        gained = int(round(gained_px / per)) if per else 0
                        if gained >= 1 and t["color"] not in refill_colors:
                            refill_colors.append(t["color"]); refill_amounts[t["color"]] = gained
                            log(f"goal-search: colour {t['color']} refills the gauge (+{gained} actions, measured)")
                    except Exception:
                        pass
                    continue
            if progressed:
                continue
        # 0b. resource-aware plan when a gauge is in play: refuel/collect/goal in one route (planner.plan_resources)
        if g and g.get("actions_left") is not None and goal_colors:
            from .planner import plan_resources
            cells = lambda cols: {nav.block_of(t["row"], t["col"]) for t in nav.targets(max_n=40) if t["color"] in cols}
            refill_amt = {c: a for c, a in (refill_amounts or {}).items()}
            rc = {nav.block_of(t["row"], t["col"]): int(refill_amt.get(t["color"], 4)) for t in nav.targets(max_n=40) if t["color"] in set(refill_colors)}
            plan = None
            for req in (True, False):
                plan = plan_resources(nav, goal_cells=cells(set(goal_colors)), collect_cells=cells(set(collect_colors)), refill_cells=rc,
                                      hazard_colors=set(hazard_colors), gauge_left=int(g["actions_left"]), require_collect=req)
                if plan:
                    break
            if plan:
                log(f"goal-search: resource plan of {len(plan)} moves (collect first: {req})")
                res = session.execute([{"action": a} for a in plan[:24]])
                if session.level > level0:
                    return done({"completed": True, "actions": session.actions_used - start, "reason": "resource plan reached the goal"})
                if res["game_over"]:
                    return done({"completed": False, "actions": session.actions_used - start, "reason": "game over"})
                if res["executed_count"]:
                    progressed = True; continue
        # 1. goal
        for t in targets(nav, set(goal_colors))[:3]:
            r = go(nav, t, final_any=True)
            if r is None or r.get("short"):
                continue
            log(f"goal-search: went to goal colour {t['color']} at ({t['row']},{t['col']}) -> level {'done' if session.level > level0 else 'not done'}")
            if session.level > level0:
                return done({"completed": True, "actions": session.actions_used - start, "reason": "reached goal"})
            if r["game_over"]:
                return done({"completed": False, "actions": session.actions_used - start, "reason": "game over"})
            visited.add((t["row"], t["col"])); progressed = True
            break
        if session.level != level0:
            break
        # 2. prerequisites: collect
        nav = nav_now()
        for t in targets(nav, set(collect_colors))[:4]:
            r = go(nav, t)
            if r is None or r.get("short"):
                continue
            log(f"goal-search: collected colour {t['color']} at ({t['row']},{t['col']})")
            visited.add((t["row"], t["col"])); progressed = True
            if session.level > level0:
                return done({"completed": True, "actions": session.actions_used - start, "reason": "collected -> level done"})
            if r["game_over"]:
                return done({"completed": False, "actions": session.actions_used - start, "reason": "game over"})
            break
        if not progressed:
            # every known route was too long for the gauge or nothing known is reachable: learn what the nearest unknown object is
            # (new colours on this level first — they are usually the new mechanic, e.g. refills); rules.py will classify the effect
            nav = nav_now(); known = set(goal_colors) | set(collect_colors) | set(refill_colors) | set(hazard_colors)
            unknown = [t for t in nav.targets(max_n=30) if t["color"] not in known and t.get("path_len") and (t["row"], t["col"]) not in visited]
            unknown.sort(key=lambda t: t["path_len"])
            for t in unknown[:2]:
                r = go(nav, t)
                if r and not r.get("short") and r["executed_count"]:
                    visited.add((t["row"], t["col"])); progressed = True
                    log(f"goal-search: probed unknown colour {t['color']} at ({t['row']},{t['col']})")
                    if session.level > level0:
                        return done({"completed": True, "actions": session.actions_used - start, "reason": "probe completed the level"})
                    if r["game_over"]:
                        return done({"completed": False, "actions": session.actions_used - start, "reason": "game over"})
                    g2 = nav_now().gauge()
                    if g and g2 and g2.get("size", 0) > g.get("size", 0) and t["color"] not in refill_colors:
                        refill_colors.append(t["color"]); log(f"goal-search: colour {t['color']} refills the gauge")
                    break
        if not progressed:
            stale_rounds += 1
            if stale_rounds >= 2:
                return done({"completed": False, "actions": session.actions_used - start, "reason": "no reachable goal or collectible"})
            visited.clear()   # allow a second pass over the same targets once (doors may have opened)
        else:
            stale_rounds = 0
    return done({"completed": session.level > level0, "actions": session.actions_used - start, "reason": "budget" if session.level == level0 else "level completed"})
