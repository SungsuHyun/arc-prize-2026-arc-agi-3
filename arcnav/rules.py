"""Symbolic rule induction over recorded transitions (object level, exact fit).

Each rule is a parametric hypothesis fitted to the transitions of the current level. A rule reports
`support` (transitions it explains) and `counter` (transitions it contradicts); rules with counter == 0
and support >= 2 are 'confirmed'. The harness shows confirmed rules and unexplained transitions to the
model, and planners may rely on confirmed rules.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .frame import masked_ascii
from .nav import NavHelper, extract_objects


@dataclass
class Rule:
    kind: str
    params: dict
    support: int = 0
    counter: int = 0
    examples: list = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.counter == 0 and self.support >= 2

    def text(self) -> str:
        p = self.params
        if self.kind == "move":
            return f"{p['action']} moves the avatar by (dx={p['dx']}, dy={p['dy']}) [{self.support} ok, {self.counter} blocked]"
        if self.kind == "wall":
            return f"the avatar cannot enter colour {p['color']} (blocked {self.support}x)"
        if self.kind == "gauge":
            return f"the gauge (colour {p['color']}) drops by {p['per_action']} per action"
        if self.kind == "refill":
            return f"touching colour {p['color']} refills the gauge (+{p['amount']}) and consumes it"
        if self.kind == "collect":
            return f"touching colour {p['color']} objects makes them disappear (collected, {self.support}x)"
        if self.kind == "hazard":
            return f"touching colour {p['color']} resets the avatar to its start position ({self.support}x)"
        if self.kind == "click":
            return f"clicking colour {p['color']} changes colour(s) {sorted(p['changes'])} ({self.support}x)" if p["changes"] else f"clicking colour {p['color']} does nothing ({self.support}x)"
        if self.kind == "mirror":
            how = {"mirror_x": "horizontally mirrored (-dx, dy)", "mirror_y": "vertically mirrored (dx, -dy)", "mirror_xy": "point-mirrored (-dx, -dy)"}[p["how"]]
            return f"a second body (colour {p['color']}, {p['size']}px) moves with every move, {how} — walls stop each body separately ({self.support}x)"
        if self.kind == "noop":
            return f"{p['action']} changed nothing ({self.support}x)"
        return f"{self.kind} {p}"


def _objs(frame):
    objs, bg = extract_objects(frame.grid)
    return objs, bg


def _count(frame, color):
    return sum(1 for row in frame.grid for v in row if v == color)


def induce(transitions: list, current_frame) -> tuple[list[Rule], list[str]]:
    """transitions: objects with .action (str or dict), .before_frame, .after_frame (same level).
    Returns (rules, unexplained) where unexplained lists transitions no rule accounts for."""
    rules: list[Rule] = []
    if not transitions:
        return rules, []
    nav = NavHelper(transitions, current_frame)
    moves = nav.moves
    # 1. movement + walls
    per_action_ok: dict[str, int] = Counter(); per_action_blocked: dict[str, int] = Counter()
    wall_hits: Counter = Counter()
    avatar_positions = []
    for t in transitions:
        a = t.action if isinstance(t.action, str) else "MOUSE"
        if a not in moves:
            continue
        nb = NavHelper([t], t.after_frame)
        av_b = NavHelper(transitions, t.before_frame).avatar(); av_a = NavHelper(transitions, t.after_frame).avatar()
        if not av_b or not av_a:
            continue
        moved = (av_a["row"] - av_b["row"], av_a["col"] - av_b["col"]) != (0, 0)
        if moved:
            per_action_ok[a] += 1
        else:
            per_action_blocked[a] += 1
            dx, dy = moves[a]; r, c = av_b["row"] + dy, av_b["col"] + dx
            if 0 <= r < 64 and 0 <= c < 64:
                wall_hits[t.before_frame.grid[r][c]] += 1
        avatar_positions.append((av_b["row"], av_b["col"], av_a["row"], av_a["col"], a, t))
    for a, (dx, dy) in moves.items():
        rules.append(Rule("move", {"action": a, "dx": dx, "dy": dy}, support=per_action_ok[a], counter=0))
    for color, n in wall_hits.items():
        rules.append(Rule("wall", {"color": color}, support=n))
    # 1b. mirrored / co-moving second body: another object that moves whenever the avatar moves, with a fixed transform
    if moves:
        pairs: Counter = Counter(); seen_moves = 0
        for t in transitions:
            a = t.action if isinstance(t.action, str) else "MOUSE"
            if a not in moves:
                continue
            po, _ = _objs(t.before_frame); co, _ = _objs(t.after_frame)
            from .nav import _moved
            mv = [(o, dx, dy) for o, dx, dy in _moved(po, co) if 4 <= o["size"] <= 400]   # ignore 1-3 px specks (ar25 false trigger)
            if len(mv) < 2:
                continue
            seen_moves += 1
            adx, ady = moves[a]
            for o, dx, dy in mv:
                if (dx, dy) == (adx, ady):
                    continue
                if (dx, dy) == (-adx, ady):
                    pairs[("mirror_x", o["color"], o["size"])] += 1
                elif (dx, dy) == (adx, -ady):
                    pairs[("mirror_y", o["color"], o["size"])] += 1
                elif (dx, dy) == (-adx, -ady):
                    pairs[("mirror_xy", o["color"], o["size"])] += 1
        for (kind, color, size), n in pairs.items():
            if n >= 2 and n * 2 >= seen_moves:   # the second body must move in at least half of the observed moves
                rules.append(Rule("mirror", {"how": kind, "color": color, "size": size}, support=n, counter=max(0, seen_moves - n)))
    # 2. gauge and refills
    g = nav.gauge()
    av0 = nav.avatar(); avatar_cols = set(av0["colors"]) if av0 else set()
    for (color, size), _ in nav.player_votes.most_common(5):
        avatar_cols.add(color)
    if g:
        rules.append(Rule("gauge", {"color": g["color"], "per_action": g["per_action"]}, support=len(transitions)))
        color = g["color"]
        refills: Counter = Counter(); amounts: dict = {}
        for t in transitions:
            b, af = _count(t.before_frame, color), _count(t.after_frame, color)
            if af > b + 2:
                # a small non-avatar object that vanished where the avatar arrived
                av_a = NavHelper(transitions, t.after_frame).avatar()
                after_objs = _objs(t.after_frame)[0]
                gone = [o for o in _objs(t.before_frame)[0] if o["size"] <= 40 and o["color"] not in avatar_cols and o["color"] != nav.background
                        and o["color"] not in nav.floor_colors
                        and not any(p["color"] == o["color"] and p["center"] == o["center"] for p in after_objs)
                        and (not av_a or (abs(o["center"][1] - av_a["row"]) <= 6 and abs(o["center"][0] - av_a["col"]) <= 6))]
                for o in gone[:1]:
                    refills[o["color"]] += 1; amounts[o["color"]] = af - b
        for c, n in refills.items():
            rules.append(Rule("refill", {"color": c, "amount": amounts[c]}, support=n))
    # 3. collectibles and hazards (movement games)
    if moves:
        vanish: Counter = Counter(); reset_colors: Counter = Counter()
        av = nav.avatar(); avatar_colors = set(av["colors"]) if av else set()
        for (color, size), _ in nav.player_votes.most_common(5):
            avatar_colors.add(color)
        start = avatar_positions[0][:2] if avatar_positions else None
        for rb, cb, ra, ca, a, t in avatar_positions:
            before_objs, _ = _objs(t.before_frame); after_objs, _ = _objs(t.after_frame)
            after_keys = {(o["color"], o["center"]) for o in after_objs}
            for o in before_objs:
                if o["size"] <= 60 and o["color"] not in avatar_colors and (o["color"], o["center"]) not in after_keys:
                    # object vanished; did the avatar arrive at it?
                    if abs(o["center"][1] - ra) <= 4 and abs(o["center"][0] - ca) <= 4:
                        vanish[o["color"]] += 1
            if start and (ra, ca) == start and (rb, cb) != start and abs(rb - ra) + abs(cb - ca) > 6:
                dx, dy = moves[a]; r, c = rb + dy, cb + dx
                if 0 <= r < 64 and 0 <= c < 64:
                    reset_colors[t.before_frame.grid[r][c]] += 1
        for color, n in vanish.items():
            if color != nav.background and color not in avatar_colors and color not in nav.floor_colors:
                rules.append(Rule("collect", {"color": color}, support=n))
        for color, n in reset_colors.items():
            rules.append(Rule("hazard", {"color": color}, support=n))
    # 4. click effects
    clicks: dict[int, Counter] = defaultdict(Counter); click_n: Counter = Counter()
    for t in transitions:
        if not isinstance(t.action, dict):
            continue
        r, c = t.action["row"], t.action["col"]
        color = t.before_frame.grid[r][c]
        click_n[color] += 1
        changed = {t.before_frame.grid[i][j] for i in range(64) for j in range(64)
                   if masked_ascii(t.before_frame).splitlines()[i][j] != masked_ascii(t.after_frame).splitlines()[i][j]}
        clicks[color].update(changed)
    for color, n in click_n.items():
        rules.append(Rule("click", {"color": color, "changes": set(clicks[color])}, support=n))
    # 5. no-ops
    noop: Counter = Counter()
    for t in transitions:
        a = t.action if isinstance(t.action, str) else "MOUSE"
        if a != "MOUSE" and masked_ascii(t.before_frame) == masked_ascii(t.after_frame):
            noop[a] += 1
    for a, n in noop.items():
        if a not in moves:
            rules.append(Rule("noop", {"action": a}, support=n))
    # unexplained: transitions with a masked change that no rule kind covers
    unexplained = []
    for i, t in enumerate(transitions):
        a = t.action if isinstance(t.action, str) else "MOUSE"
        if a == "MOUSE" or a in moves or a in noop:
            continue
        if masked_ascii(t.before_frame) != masked_ascii(t.after_frame):
            unexplained.append(f"transition {i}: {a} changed the board but matches no known rule")
    return rules, unexplained


def summary(rules: list[Rule], unexplained: list[str]) -> str:
    conf = [r for r in rules if r.confirmed]; weak = [r for r in rules if not r.confirmed and r.support >= 1]
    lines = ["CONFIRMED RULES (fit every recorded transition):"] + [f"  - {r.text()}" for r in conf] if conf else ["CONFIRMED RULES: none yet"]
    if weak:
        lines.append("TENTATIVE (seen once or with counter-examples): " + "; ".join(r.text() for r in weak[:6]))
    if unexplained:
        lines.append("UNEXPLAINED: " + "; ".join(unexplained[:4]))
    return "\n".join(lines)
