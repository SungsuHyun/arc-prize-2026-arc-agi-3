"""Win-condition inference from a completed level (harness side, no model).

Given the transitions of the winning attempt (same level, ending with the transition that entered the next
level), produce ranked goal hypotheses that planners can act on in later levels:
  reach:          the final move entered a cell of colour G (a goal tile / door)
  collect_reach:  all objects of colour C disappeared during the attempt, then the avatar reached G
  collect_all:    the last object of colour C disappeared on the final action (no separate goal tile)
  click_sequence: (click games) the ordered colours clicked in the winning attempt
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .nav import NavHelper, extract_objects


def _objs(frame):
    return extract_objects(frame.grid)[0]


def infer(attempt: list, level: int) -> list[dict]:
    """attempt: transitions of the winning attempt on `level` (last one enters level+1)."""
    if not attempt:
        return []
    same = [t for t in attempt if t.before_frame.level == level]
    if not same:
        return []
    first, last = same[0].before_frame, same[-1].before_frame   # last board seen on this level
    final = same[-1]
    hyps: list[dict] = []
    # colours that vanished completely during the attempt (collectibles), ignoring the avatar's own colours
    nav = NavHelper(same[:-1] or same, last)
    av = nav.avatar(); avatar_cols = set(av["colors"]) if av else set()
    for (c, _), _ in nav.player_votes.most_common(4):
        avatar_cols.add(c)
    bg = nav.background
    def counts(frame):
        cnt = Counter()
        for o in _objs(frame):
            if o["size"] <= 200 and o["color"] != bg and o["color"] not in avatar_cols and o["color"] not in nav.floor_colors:
                cnt[o["color"]] += 1
        return cnt
    c0, c1 = counts(first), counts(last)
    vanished = {c: n for c, n in c0.items() if n >= 1 and c1.get(c, 0) == 0}
    # what did the final move enter? (movement games)
    action = final.action if isinstance(final.action, str) else "MOUSE"
    entered = None
    if action in nav.moves and av:
        dx, dy = nav.moves[action]
        r, c = av["row"] + dy, av["col"] + dx
        if 0 <= r < 64 and 0 <= c < 64:
            col = last.grid[r][c]
            if col != bg and col not in nav.floor_colors and col not in avatar_cols:
                entered = col
    if entered is not None:
        if vanished:
            for c, n in sorted(vanished.items(), key=lambda x: -x[1]):
                hyps.append({"type": "collect_reach", "collect": c, "collect_count": n, "reach": entered,
                             "text": f"collect all colour-{c} objects ({n} on level {level}) and then step onto a colour-{entered} object"})
        hyps.append({"type": "reach", "reach": entered, "text": f"step onto a colour-{entered} object (the final move of level {level} entered one)"})
    elif vanished:
        for c, n in sorted(vanished.items(), key=lambda x: -x[1]):
            hyps.append({"type": "collect_all", "collect": c, "collect_count": n, "text": f"make every colour-{c} object disappear ({n} on level {level}); the level ended when the last one vanished"})
    # click games: the winning click order by colour
    clicks = [t.action for t in same if isinstance(t.action, dict)]
    if clicks and not nav.moves:
        seq = []
        for t in same:
            if isinstance(t.action, dict):
                r, c = t.action["row"], t.action["col"]; seq.append(t.before_frame.grid[r][c])
        hyps.append({"type": "click_sequence", "sequence": seq[-12:], "text": f"level {level} was won by clicking colours in this order: {seq[-12:]}"})
    # two bodies of the avatar colour became one on the final move (merge games)
    if not hyps and av:
        def n_bodies(frame):
            return sum(1 for o in _objs(frame) if o["color"] in avatar_cols and o["size"] >= 4)
        if n_bodies(first) >= 2 and n_bodies(final.after_frame) <= 1 and n_bodies(last) >= 2:
            hyps.append({"type": "merge", "colors": sorted(avatar_cols), "text": f"bring the two colour-{sorted(avatar_cols)} bodies onto the same cell (they merged on the final move of level {level})"})
    if not hyps:
        hyps.append({"type": "unknown", "text": f"level {level} ended after {action}; no clear goal pattern (avatar colours {sorted(avatar_cols)}, vanished {dict(vanished)})"})
    for h in hyps:
        h["level"] = level
    return hyps


def summary(hyps: list[dict]) -> str:
    if not hyps:
        return ""
    return "GOAL HYPOTHESES (inferred by the harness from how earlier levels were won; the same rule usually applies):\n" + \
        "\n".join(f"  {i + 1}. {h['text']}" for i, h in enumerate(hyps[:3]))
