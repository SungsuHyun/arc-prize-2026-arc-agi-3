"""probe/semantics.py — ActionSemantics: deterministic classification of each action key from its Diff patterns
(spec §7). Classes: MOVE, TOGGLE, SPAWN, REMOVE, SELECT, NOOP, UNKNOWN."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from ..core.types import Scene, Transition

CLASSES = ("MOVE", "TRANSFORM", "TOGGLE", "SPAWN", "REMOVE", "SELECT", "NOOP", "UNKNOWN")


def _ui_ids(scene: Scene) -> set[int]:
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    return {o.id for o in scene.objects if o.region in strips}


def core_diff(t: Transition) -> dict:
    """Diff with ui-strip objects removed (counters change on every action and would hide the real effect)."""
    ui = _ui_ids(t.before) | _ui_ids(t.after)
    d = t.diff
    return {"moved": [(i, v) for i, v in d.moved if i not in ui], "recolored": [x for x in d.recolored if x[0] not in ui],
            "reshaped": [x for x in d.reshaped if x[0] not in ui], "appeared": [o for o in d.appeared if o.id not in ui],
            "disappeared": [o for o in d.disappeared if o.id not in ui],
            "counters": [i for i, _ in d.moved if i in ui] + [x[0] for x in d.recolored + d.reshaped if x[0] in ui]}


def classify_one(t: Transition) -> tuple[str, dict]:
    c = core_diff(t)
    if not (c["moved"] or c["recolored"] or c["reshaped"] or c["appeared"] or c["disappeared"]):
        return ("NOOP", {"counters": c["counters"]}) if t.diff.is_noop or c["counters"] or t.diff.pixel_changes == 0 else ("UNKNOWN", {"pixels": t.diff.pixel_changes})
    reshaped_ids = {x[0] for x in c["reshaped"]}
    if reshaped_ids and all(i in reshaped_ids for i, _ in c["moved"]) and not c["appeared"] and not c["disappeared"]:
        # every moved object also changed shape: a rotation / flip / re-drawing, not a translation
        return "TRANSFORM", {"targets": sorted(reshaped_ids), "counters": c["counters"]}
    if c["moved"] and not c["appeared"] and not c["disappeared"]:
        disp = Counter(v for _, v in c["moved"])
        (dr, dc), n = disp.most_common(1)[0]
        return "MOVE", {"displacement": (dr, dc), "movers": [i for i, v in c["moved"] if v == (dr, dc)], "others": len(c["moved"]) - n,
                        "recolored": [x[0] for x in c["recolored"]], "counters": c["counters"]}
    if c["recolored"] and not c["moved"] and not c["appeared"] and not c["disappeared"]:
        if t.action.type == "CLICK":
            return "SELECT", {"targets": [x[0] for x in c["recolored"]], "counters": c["counters"]}
        return "TOGGLE", {"targets": [(i, a, b) for i, a, b in c["recolored"]], "counters": c["counters"]}
    if c["appeared"] and not c["disappeared"] and not c["moved"]:
        return "SPAWN", {"ids": [o.id for o in c["appeared"]], "counters": c["counters"]}
    if c["disappeared"] and not c["appeared"] and not c["moved"]:
        return "REMOVE", {"ids": [o.id for o in c["disappeared"]], "counters": c["counters"]}
    if t.action.type == "CLICK" and (c["reshaped"] or c["recolored"]):
        return "SELECT", {"targets": [x[0] for x in c["reshaped"] + c["recolored"]], "counters": c["counters"]}
    if c["moved"] and (c["appeared"] or c["disappeared"]):
        disp = Counter(v for _, v in c["moved"])
        (dr, dc), n = disp.most_common(1)[0]
        return "MOVE", {"displacement": (dr, dc), "movers": [i for i, v in c["moved"] if v == (dr, dc)], "others": len(c["moved"]) - n,
                        "removed": [o.id for o in c["disappeared"]], "spawned": [o.id for o in c["appeared"]], "counters": c["counters"]}
    return "UNKNOWN", {"kinds": sorted(t.diff.kinds()), "counters": c["counters"]}


class ActionSemantics(dict):
    """{action_key: {"class", "effect", "consistency", ...}} plus "_counters" (ui objects that tick on actions)."""

    def moves(self) -> dict[str, tuple[int, int]]:
        return {k: tuple(v["displacement"]) for k, v in self.items() if not k.startswith("_") and v.get("class") == "MOVE" and v.get("displacement")}

    def mover_ids(self) -> set[int]:
        out: set[int] = set()
        for k, v in self.items():
            if not k.startswith("_") and v.get("class") == "MOVE":
                out |= set(v.get("movers", []))
        return out

    def noop_keys(self) -> set[str]:
        return {k for k, v in self.items() if not k.startswith("_") and v.get("class") == "NOOP"}

    def to_json(self) -> dict:
        return {k: v for k, v in self.items()}


def classify_actions(log: Iterable[Transition], prior: Optional[ActionSemantics] = None) -> ActionSemantics:
    obs: dict[str, list[tuple[str, dict, Transition]]] = {}
    for t in log:
        if t.action.type == "RESET":
            continue
        cls, info = classify_one(t)
        obs.setdefault(t.action.key, []).append((cls, info, t))
    sem = ActionSemantics(prior or {})
    counters: Counter = Counter()
    for key, items in obs.items():
        classes = Counter(c for c, _, _ in items)
        non_noop = [(c, i, t) for c, i, t in items if c != "NOOP"]
        for _, i, _ in items:
            counters.update(i.get("counters", []))
        if not non_noop:
            states = {t.before.frame_hash for _, _, t in items}
            cls = "NOOP" if len(items) >= 2 and len(states) >= 2 else "UNKNOWN"
            sem[key] = {"class": cls, "effect": None, "consistency": f"{len(items)}/{len(items)}", "n": len(items), "noop_count": len(items),
                        "note": None if cls == "NOOP" else "no change so far, but only observed from one state"}
            continue
        cls, n = Counter(c for c, _, _ in non_noop).most_common(1)[0]
        entry: dict = {"class": cls, "consistency": f"{n}/{len(items)}", "n": len(items), "noop_count": classes.get("NOOP", 0)}
        if cls == "MOVE":
            disp = Counter(tuple(i["displacement"]) for c, i, _ in non_noop if c == "MOVE")
            (dr, dc), k = disp.most_common(1)[0]
            movers = Counter()
            for c, i, _ in non_noop:
                if c == "MOVE" and tuple(i["displacement"]) == (dr, dc):
                    movers.update(i["movers"])
            entry.update(displacement=[dr, dc], movers=[m for m, _ in movers.most_common(4)], effect=f"obj {[m for m, _ in movers.most_common(2)]} by ({dr:+d},{dc:+d})",
                         displacement_consistency=f"{k}/{len(non_noop)}")
        elif cls == "TRANSFORM":
            tg = Counter(x for c, i, _ in non_noop if c == "TRANSFORM" for x in i["targets"])
            entry.update(effect=f"transforms objects {[t for t, _ in tg.most_common(3)]} (rotation/flip/redraw)", targets=[t for t, _ in tg.most_common(6)])
        elif cls == "TOGGLE":
            pairs = Counter((a, b) for c, i, _ in non_noop if c == "TOGGLE" for _, a, b in i["targets"])
            entry.update(effect="recolor " + ", ".join(f"{a}->{b}" for (a, b), _ in pairs.most_common(3)), targets=sorted({x[0] for c, i, _ in non_noop if c == "TOGGLE" for x in i["targets"]}))
        elif cls == "SELECT":
            responsive = sorted({x for c, i, _ in non_noop if c == "SELECT" for x in i["targets"]})
            entry.update(effect=f"targets {responsive[:6]} respond", responsive_targets=responsive)
        elif cls in ("SPAWN", "REMOVE"):
            entry.update(effect=f"{cls.lower()} {sorted({x for c, i, _ in non_noop if c == cls for x in i['ids']})[:6]}")
        else:
            entry.update(effect="mixed changes")
        if key == "CLICK":
            resp, unresp = [], []
            for c, i, t in items:
                tgt = (t.action.row, t.action.col)
                (resp if c != "NOOP" else unresp).append(tgt)
            entry["responsive_coords"] = resp[:20]; entry["unresponsive_coords"] = unresp[:20]
        sem[key] = entry
    sem["_counters"] = [i for i, _ in counters.most_common(6)]
    return sem
