"""wml/induce.py — deterministic hypothesis induction from ActionSemantics + prior mechanisms (no LLM).

The spec generates candidates with the LLM; this module provides the prior-driven baseline the lab always tries first
(and the no-LLM fallback): movers under button actions become `agent`, consistent displacements become move rules,
cancelled moves become wall / floor rules, disappearances on overlap become collect rules, ui strips become shrink rules.
Every candidate is still verified on the whole log by evaluate()."""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..core.contracts import Rule, RuleModel
from ..core.types import Action, Scene, Transition
from ..memory.priors.mechanisms import (cancel_move_if_off_floor, cancel_move_if_outside, cancel_move_if_overlap, collect_on_overlap,
                                        move_on_click, move_role, noop_for, permute_on_click, push_role, recolor_on_click, remove_on_click, unknown_for)
from ..env.calibrate import object_under


def infer_roles_static(log: list[Transition], semantics: dict) -> dict:
    """Role hints from the log: colours/shapes of movers -> agent; objects that stopped a move -> wall; objects that vanished under the agent -> collectible."""
    hints: dict = {"agent": set(), "wall_colors": set(), "collectible_colors": set(), "floor_colors": set()}
    movers = set()
    for k, v in semantics.items():
        if not k.startswith("_") and isinstance(v, dict) and v.get("class") == "MOVE":
            movers |= set(v.get("movers", []))
    for t in log:
        for o in t.before.objects:
            if o.id in movers:
                hints["agent"].add((o.colors, o.shape_sig))
    hints["agent_colors"] = {cs for cs, _ in hints["agent"]}
    # indicators: objects in non-board panels (or outside every panel) that recoloured but never moved
    recolored: set[tuple] = set(); moved_keys: set[tuple] = set()
    for t in log:
        for oid, _, _ in t.diff.recolored:
            o = t.before.get(oid)
            if o is not None:
                recolored.add((o.region, tuple(o.bbox)))
        reshaped = {x[0] for x in t.diff.reshaped}
        for oid, _ in t.diff.moved:
            o = t.before.get(oid)
            if o is not None and oid not in reshaped:      # a shrink/grow is not a move
                moved_keys.add((o.region, tuple(o.bbox)))
    hints["indicator_boxes"] = {k for k in recolored if k not in moved_keys}
    # floor: colours of the pixels the agent actually moved onto (rendered scene without the agent)
    for t in log:
        moved = dict(t.diff.moved)
        agents = [o for o in t.before.objects if o.id in movers and o.id in moved]
        if not agents:
            continue
        floor = t.before.copy(objects=[o for o in t.before.objects if o.id not in {a.id for a in agents}]).render()
        h, w = floor.shape
        for a in agents:
            g = a.moved(*moved[a.id]); r0, c0, r1, c1 = g.bbox
            if 0 <= r0 and 0 <= c0 and r1 <= h and c1 <= w:
                hints["floor_colors"] |= {int(v) for v in floor[r0:r1, c0:c1][g.mask]}
    return hints


def _role_fn_factory(agent_keys: set, wall_colors: set, collectible_colors: set, agent_colors: set, pushable_colors: set = frozenset(),
                     indicator_boxes: set = frozenset()):
    ind_boxes = {b for _, b in indicator_boxes}
    agent_palette = set()
    for cs in agent_colors:
        agent_palette |= set(cs)

    def agent_similarity(o) -> float:
        """Colour-overlap similarity with the known agent appearances (an animated avatar changes pattern with facing)."""
        cs = set(o.colors)
        if not cs & agent_palette:
            return 0.0
        return len(cs & agent_palette) / len(cs | agent_palette)

    def role_fn(scene: Scene) -> dict[int, str]:
        roles = {}
        strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
        for o in scene.objects:
            if o.region in strips or (tuple(o.bbox) in ind_boxes and (o.colors, o.shape_sig) not in agent_keys):
                roles[o.id] = "indicator"
            elif (o.colors, o.shape_sig) in agent_keys or (o.colors in agent_colors and len(o.colors) > 1):
                roles[o.id] = "agent"
            elif o.color in collectible_colors:
                roles[o.id] = "collectible"
            elif o.color in pushable_colors:
                roles[o.id] = "pushable"
            elif o.color in wall_colors:
                roles[o.id] = "wall"
            else:
                roles[o.id] = "unknown"
        if "agent" not in roles.values():
            # appearance changed (facing/animation): the multi-colour object most similar to the known agent palette,
            # else the single-colour object of an agent colour
            cands = [(agent_similarity(o), len(o.colors), o.area, o) for o in scene.objects if o.region not in strips and roles.get(o.id) != "indicator"]
            cands = [c for c in cands if c[0] > 0]
            if cands:
                cands.sort(key=lambda c: (-c[0], -c[1], -c[2]))
                roles[cands[0][3].id] = "agent"
        return roles
    return role_fn


def induce_hypotheses(log: list[Transition], semantics: dict, available: list[Action]) -> list[tuple[str, RuleModel]]:
    """Return (name, model) candidates built from priors. Several variants so evaluate() can pick the best."""
    if not log:
        return []
    moves = {k: tuple(v["displacement"]) for k, v in semantics.items() if not k.startswith("_") and isinstance(v, dict) and v.get("class") == "MOVE" and v.get("displacement")}
    hints = infer_roles_static(log, semantics)
    agent_keys = set(hints["agent"]); agent_colors = set(hints["agent_colors"])
    if not moves or not agent_keys:
        return induce_click_hypotheses(log, semantics, available)
    dirs = {int(k.replace("ACTION", "")): d for k, d in moves.items() if k.startswith("ACTION")}
    if not dirs:
        return induce_click_hypotheses(log, semantics, available)
    buttons = sorted({a.id for a in available if a.type == "BUTTON"})
    noop_buttons = tuple(b for b in buttons if f"ACTION{b}" in semantics and semantics[f"ACTION{b}"].get("class") == "NOOP")
    other_buttons = tuple(b for b in buttons if b not in dirs and b not in noop_buttons)
    # candidate wall colours: colours of objects the agent overlapped when a move did not happen
    wall_votes: Counter = Counter(); collect_votes: Counter = Counter(); push_votes: Counter = Counter()
    for t in log:
        if t.action.type != "BUTTON" or t.action.id not in dirs:
            continue
        dr, dc = dirs[t.action.id]
        agents = [o for o in t.before.objects if (o.colors, o.shape_sig) in agent_keys or o.colors in agent_colors]
        moved_ids = {i for i, _ in t.diff.moved}
        for a in agents:
            ghost = a.moved(dr, dc)
            if a.id not in moved_ids:
                for o in t.before.objects:
                    if o.id != a.id and ghost.overlaps(o) and o.colors not in agent_colors:
                        wall_votes[o.color] += 1
            else:
                for o in t.diff.disappeared:
                    if ghost.overlaps(o):
                        collect_votes[o.color] += 1
                for i, d in t.diff.moved:
                    if i != a.id and d == (dr, dc):
                        ob = t.before.get(i)
                        if ob is not None and ob.colors not in agent_colors:
                            push_votes[ob.color] += 1
    wall_colors = {c for c, n in wall_votes.items() if n >= 1}
    collectible_colors = {c for c, n in collect_votes.items() if n >= 1}
    pushable_colors = {c for c, n in push_votes.items() if n >= 2}
    floor_colors = tuple(sorted(hints["floor_colors"]))
    step_sizes = {abs(dr) + abs(dc) for dr, dc in dirs.values()}
    unit_dirs = {k: (int(np.sign(dr)), int(np.sign(dc))) for k, (dr, dc) in dirs.items()}
    step = max(step_sizes) if step_sizes else 1
    # assumed directions for arrow buttons not yet observed to move (ARC-AGI-3 convention 1=up 2=down 3=left 4=right);
    # kept as a separate hypothesis so an information-gain experiment can test it
    from ..memory.priors.mechanisms import DIRS4
    assumed = {b: DIRS4[b] for b in buttons if b in DIRS4 and b not in unit_dirs and b not in noop_buttons}
    other_buttons = tuple(b for b in other_buttons if b not in assumed) if assumed else other_buttons

    def base_rules(with_walls: bool, with_floor: bool, with_collect: bool, with_push: bool, use_assumed: bool = False) -> list[Rule]:
        d = dict(unit_dirs); d.update(assumed if use_assumed else {})
        rules = [Rule("move_agent", lambda s, a, d=d: a.type == "BUTTON" and a.id in d, move_role("agent", d, step), source="induced")]
        if assumed and not use_assumed:
            rules.append(Rule("unassumed_dirs", lambda s, a: a.type == "BUTTON" and a.id in assumed, unknown_for(tuple(assumed)), source="induced"))
        if with_push and pushable_colors:
            rules.append(Rule("push", lambda s, a: a.type == "BUTTON" and a.id in dirs, push_role("agent", "pushable"), source="induced"))
        if with_walls:
            rules.append(Rule("wall_blocks", lambda s, a: a.type == "BUTTON" and a.id in dirs, cancel_move_if_overlap("agent", "wall"), source="induced"))
        if with_floor and floor_colors:
            rules.append(Rule("stay_on_floor", lambda s, a: a.type == "BUTTON" and a.id in dirs, cancel_move_if_off_floor("agent", floor_colors), source="induced"))
        else:
            rules.append(Rule("stay_inside", lambda s, a: a.type == "BUTTON" and a.id in dirs, cancel_move_if_outside("agent"), source="induced"))
        if with_collect and collectible_colors:
            rules.append(Rule("collect", lambda s, a: a.type == "BUTTON" and a.id in dirs, collect_on_overlap("agent", "collectible"), source="induced"))
        if noop_buttons:
            rules.append(Rule("noop_buttons", lambda s, a: a.type == "BUTTON" and a.id in noop_buttons, noop_for(noop_buttons), source="induced"))
        if other_buttons:
            rules.append(Rule("unknown_buttons", lambda s, a: a.type == "BUTTON" and a.id in other_buttons, unknown_for(other_buttons), source="induced"))
        return rules

    out = []
    variants = [("induced:move+walls+floor+collect+push", True, True, True, True), ("induced:move+floor+collect", False, True, True, False),
                ("induced:move+walls+collect", True, False, True, False), ("induced:move+floor", False, True, False, False), ("induced:move", False, False, False, False)]
    for name, w, f, c, p in variants:
        role_fn = _role_fn_factory(agent_keys, wall_colors if w else set(), collectible_colors if c else set(), agent_colors, pushable_colors if p else set(),
                                   hints["indicator_boxes"])
        m = RuleModel(base_rules(w, f, c, p), role_fn, default="unknown", name=name)
        out.append((name, m))
        if assumed:
            m2 = RuleModel(base_rules(w, f, c, p, use_assumed=True), role_fn, default="unknown", name=name + "+assumed_dirs")
            out.append((name + "+assumed_dirs", m2))
    return out


def induce_click_hypotheses(log: list[Transition], semantics: dict, available: list[Action]) -> list[tuple[str, RuleModel]]:
    """Click games: what happens to the clicked object, by (colour) class -> recolour cycle / removal / no-op."""
    clicks = [t for t in log if t.action.type == "CLICK"]
    if not clicks:
        return []
    effects: dict[int, Counter] = {}     # colour of clicked object -> Counter of effect keys
    cycles: dict[int, Counter] = {}

    def strip_ids(t: Transition) -> set[int]:
        strips = {r.id for r in t.before.regions if r.kind_hint == "ui_strip"} | {r.id for r in t.after.regions if r.kind_hint == "ui_strip"}
        return {o.id for o in t.before.objects + t.after.objects if o.region in strips}

    for t in clicks:
        oid = object_under(t.before, t.action.row, t.action.col)
        o = t.before.get(oid) if oid is not None else None
        color = o.color if o is not None else -1
        eff = effects.setdefault(color, Counter())
        rec = [x for x in t.diff.recolored if x[0] == oid]
        if oid is not None and rec:
            eff["recolor"] += 1; cycles.setdefault(color, Counter())[(rec[0][1], rec[0][2])] += 1
        elif oid is not None and any(x.id == oid for x in t.diff.disappeared):
            eff["remove"] += 1
        elif t.diff.is_noop or not (set(i for i, _ in t.diff.moved) | {x[0] for x in t.diff.recolored + t.diff.reshaped} | {o.id for o in t.diff.appeared + t.diff.disappeared}) - strip_ids(t):
            eff["noop"] += 1
        else:
            eff["other"] += 1
    # click on a trigger object moves other objects (arrows / controls): trigger (colour, shape) -> displacement of movers
    trig: dict[tuple, Counter] = {}; trig_movers: dict[tuple, Counter] = {}
    for t in clicks:
        oid = object_under(t.before, t.action.row, t.action.col)
        o = t.before.get(oid) if oid is not None else None
        if o is None or not t.diff.moved:
            continue
        key = (o.color, o.shape_sig)
        ui = strip_ids(t)
        disp = Counter(v for i, v in t.diff.moved if i != oid and i not in ui)
        if disp:
            (d, n) = disp.most_common(1)[0]
            trig.setdefault(key, Counter())[d] += 1
            for i, v in t.diff.moved:
                if v == d:
                    ob = t.before.get(i)
                    if ob is not None:
                        trig_movers.setdefault(key, Counter())[ob.color] += 1
    click_moves = {k: c.most_common(1)[0][0] for k, c in trig.items() if c.most_common(1)[0][1] >= 2}
    # learned permutations: per trigger class, the top-left -> top-left mapping of every moved object, if consistent
    votes: dict[tuple, dict] = {}
    last_level = max(t.level for t in clicks)
    for t in [x for x in clicks if x.level == last_level and not x.status_change]:   # level-specific positions; a level-up frame is the next board
        oid = object_under(t.before, t.action.row, t.action.col)
        o = t.before.get(oid) if oid is not None else None
        if o is None or not t.diff.moved:
            continue
        key = (o.color, o.shape_sig, tuple(o.bbox)); ui = strip_ids(t)     # a permutation belongs to ONE trigger object
        reshaped = {x[0] for x in t.diff.reshaped}
        m = votes.setdefault(key, {})
        for i, (dr, dc) in t.diff.moved:
            ob = t.before.get(i)
            if ob is None or i in ui or i == oid or i in reshaped:
                continue
            src = (ob.bbox[0], ob.bbox[1]); dst = (ob.bbox[0] + dr, ob.bbox[1] + dc)
            m.setdefault(src, Counter())[dst] += 1
    # majority vote per source position (tracking noise on small marks must not kill the whole permutation)
    perms: dict[tuple, dict] = {}
    for key, m in votes.items():
        mapping = {}; agree = total = 0
        for src, c in m.items():
            dst, n = c.most_common(1)[0]
            agree += n; total += sum(c.values())
            if n >= 1:
                mapping[src] = dst
        if len(mapping) >= 2 and agree >= 0.6 * total:
            perms[key] = mapping
    if not any(eff.get("recolor") or eff.get("remove") for eff in effects.values()) and not click_moves and not perms:
        return []
    recolor_colors = {c for c, eff in effects.items() if eff.get("recolor") and eff["recolor"] >= max(eff.get("other", 0), 1)}
    remove_colors = {c for c, eff in effects.items() if eff.get("remove") and eff["remove"] >= max(eff.get("other", 0), 1)}
    # colour cycles: chain a->b, b->c ...
    cycle: list[int] = []
    pairs = Counter()
    for c in recolor_colors:
        pairs.update(cycles.get(c, {}))
    nxt = {a: b for (a, b), _ in pairs.most_common()}
    if nxt:
        start = next(iter(nxt)); cur = start; cycle = [start]
        while cur in nxt and nxt[cur] not in cycle and len(cycle) < 16:
            cur = nxt[cur]; cycle.append(cur)
        if cur in nxt and nxt[cur] == start:
            pass
        else:
            cycle.append(nxt.get(cur, cur)) if nxt.get(cur) not in cycle and nxt.get(cur) is not None else None
    strips_role = "indicator"

    trigger_roles = {k: f"custom:trigger{i}" for i, k in enumerate(click_moves)}
    mover_colors = {k: {c for c, _ in trig_movers.get(k, Counter()).most_common(3)} for k in click_moves}
    perm_boxes: dict = {k[2]: f"custom:trigger_at_{k[2][0]}_{k[2][1]}" for k in perms}

    def role_fn(scene: Scene) -> dict[int, str]:
        roles = {}
        strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
        for o in scene.objects:
            if o.region in strips:
                roles[o.id] = strips_role
            elif tuple(o.bbox) in perm_boxes:
                roles[o.id] = perm_boxes[tuple(o.bbox)]
            elif (o.color, o.shape_sig) in trigger_roles:
                roles[o.id] = trigger_roles[(o.color, o.shape_sig)]
            elif any(o.color in mc for mc in mover_colors.values()):
                roles[o.id] = "custom:moved"
            elif o.color in recolor_colors or (cycle and o.color in cycle):
                roles[o.id] = "custom:toggle"
            elif o.color in remove_colors:
                roles[o.id] = "collectible"
            else:
                roles[o.id] = "unknown"
        return roles

    rules = []
    perm_rules = []
    for n, (k, m) in enumerate(perms.items()):
        role = perm_boxes[k[2]]
        perm_rules.append(Rule(f"click_perm_{n}", lambda s, a: a.type == "CLICK", permute_on_click(role, dict(m)), source="induced"))
    perm_classes = {(k[0], k[1]) for k in perms}
    for k, (dr, dc) in click_moves.items():
        if k in perm_classes:
            continue      # permutations explain these triggers better than a uniform shift
        rules.append(Rule(f"click_move_{trigger_roles[k]}", lambda s, a: a.type == "CLICK", move_on_click(trigger_roles[k], "custom:moved", dr, dc), source="induced"))
    rules = perm_rules + rules
    if cycle and len(cycle) >= 2:
        rules.append(Rule("click_recolor", lambda s, a: a.type == "CLICK", recolor_on_click("custom:toggle", tuple(cycle)), source="induced"))
    if remove_colors:
        rules.append(Rule("click_remove", lambda s, a: a.type == "CLICK", remove_on_click("collectible"), source="induced"))
    buttons = tuple(a.id for a in available if a.type == "BUTTON")
    if buttons:
        rules.append(Rule("unknown_buttons", lambda s, a: a.type == "BUTTON", unknown_for(buttons), source="induced"))
    if not rules:
        return []
    return [("induced:click", RuleModel(rules, role_fn, default="noop", name="induced:click"))]


def induced_code(name: str) -> str:
    """Pseudo-source recorded for induced models (they are rebuilt from semantics, not from code)."""
    return f"# induced model: {name}\n# (built by pbg.wml.induce.induce_hypotheses from action semantics + priors)\n"
