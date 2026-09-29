"""hypothesis/explore.py — level-1 exploration before the first hypothesis (docs/029).

Level 1 weighs 1.8-4.8% of a game score and later levels count their actions from the level-up, so level 1 is where
actions are cheapest: act first, collect transitions (and the GAME_OVER condition), then hypothesise from the log.

    buttons   a random available button, never the one just pressed (repeating the last move shows nothing new)
    clicks    one click per object class (colour + shape) not tried yet, then classes that reacted, then any object
    stop      the cap, GAME_OVER, a level-up, or `stale` actions in a row with no new change pattern

Random clicks on the whole grid reveal almost nothing in click games (tn36 0/61, vc33 1/50 changed frames), hence the
object classes; movement games are rich under random moves (ls20 96/135)."""
from __future__ import annotations

import hashlib
import random
from typing import Callable, Optional

import numpy as np

from ..core.types import Action, Transition
from ..probe.clickmap import object_key


def _change_key(t: Transition) -> Optional[str]:
    """Signature of what an action changed (None = no change beyond a counter tick of <= 4 pixels)."""
    if t.before_frame is None or t.after_frame is None:
        return None
    d = np.asarray(t.before_frame.grid) != np.asarray(t.after_frame.grid)
    if int(d.sum()) <= 4:
        return None
    return t.action.label().split("(")[0] + ":" + hashlib.md5(d.tobytes()).hexdigest()[:8]


def explore_level(s, act: Callable[[Action, str], Optional[Transition]], *, cap: int, stale: int = 12, unreactive: int = 8, seed: int = 0,
                  emit: Callable[[str], None] = lambda m: None) -> dict:
    """Spend at most `cap` actions exploring the current level; returns counts for the event log / result JSON."""
    rng = random.Random(seed)
    level = s.level
    last_button: Optional[int] = None
    tried: dict[str, int] = {}          # object class -> clicks
    reacted: set[str] = set()           # object classes whose click changed the frame
    patterns: set[str] = set()
    used = since_new = changed = 0
    unreactive_clicks = 0               # object-class clicks in a row that changed nothing (a click game that ignores object clicks)
    stop = "cap"
    while used < cap:
        if s.finished() or s.timed_out():
            stop = "clock"; break
        if s.env.status().state == "GAME_OVER":
            stop = "game_over"; break
        if s.level != level:
            stop = "level_up"; break
        if since_new >= stale:
            stop = f"no new change in {stale} actions"; break
        if unreactive_clicks >= unreactive:
            stop = f"{unreactive} object clicks without any reaction"; break
        avail = s.available_actions()
        buttons = [a for a in avail if a.type == "BUTTON"]
        can_click = any(a.type == "CLICK" for a in avail)
        a: Optional[Action] = None
        klass = None
        if buttons and (not can_click or rng.random() < 0.5):
            pool = [b for b in buttons if b.id != last_button] or buttons
            a = rng.choice(pool)
        elif can_click:
            strips = {r.id for r in s.scene.regions if r.kind_hint == "ui_strip"}
            objs = [o for o in s.scene.objects if o.region not in strips]
            if objs:
                by_class: dict[str, list] = {}
                for o in objs:
                    by_class.setdefault(object_key(o), []).append(o)
                fresh = [k for k in by_class if k not in tried]
                live = [k for k in by_class if k in reacted]
                klass = rng.choice(fresh) if fresh else (rng.choice(live) if live else rng.choice(list(by_class)))
                o = rng.choice(by_class[klass])
                r, c = o.cells()[rng.randrange(len(o.cells()))] if o.area > 1 else o.center
                a = Action.click(int(r), int(c))
            else:
                a = Action.click(rng.randrange(64), rng.randrange(64))
        if a is None:
            stop = "no action"; break
        t = act(a, "explore")
        if t is None:
            stop = "refused"; break
        used += 1
        if a.type == "BUTTON":
            last_button = a.id
        if klass is not None:
            tried[klass] = tried.get(klass, 0) + 1
        k = _change_key(t)
        if klass is not None:
            unreactive_clicks = 0 if k is not None else unreactive_clicks + 1
        if k is not None:
            changed += 1
            if klass is not None:
                reacted.add(klass)
            if k not in patterns:
                patterns.add(k); since_new = 0
                continue
        since_new += 1
    if s.env.status().state == "GAME_OVER":
        stop = "game_over"
    out = {"actions": used, "changed": changed, "patterns": len(patterns), "classes_tried": len(tried), "classes_reacted": len(reacted), "stop": stop}
    emit(f"level-{level} exploration: {used} actions, {changed} changed the board, {len(patterns)} distinct change patterns, "
         f"{len(reacted)}/{len(tried)} clicked classes reacted; stop: {stop}")
    return out
