"""tools/build_reaction_dataset.py — enumerate every level of every game (engine set_level) and record the initial map
plus the one-step reaction of each action. This is the raw material for distilling a level-start "objective / patterns"
model (teacher -> small model).

    .venv/bin/python -m pbg.tools.build_reaction_dataset --games r11l,vc33 --out experiments/pbg/datasets/reactions
    .venv/bin/python -m pbg.tools.build_reaction_dataset --all --out experiments/pbg/datasets/reactions

For each (game, level): set_level(L) gives the pristine initial state (no need to solve prior levels). We render the
initial 64x64 grid, segment it with our Perception, then for every candidate action -- the simple buttons the frame
offers, and one CLICK at the centre of every detected object -- we re-set the level, apply that single action, and record
the structured object diff (moved / appeared / disappeared / recoloured), whether the board changed at all, and whether
the action completed the level. One-step reactions only: stateful dynamics (selection, counters, sequence locks) need
action sequences and are out of scope for this first pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("ARC_API_KEY", "local-dev")
ROOT = Path(__file__).resolve().parents[2]
SIMPLE = [1, 2, 3, 4, 5, 7]     # ACTION ids that need no coordinates (ACTION6 is the click)


def _game_of(raw):
    for attr in ("_game", "game", "_env", "env"):
        g = getattr(raw, attr, None)
        if g is not None and hasattr(g, "set_level") and hasattr(g, "perform_action"):
            return g
    return None


def _render(game) -> np.ndarray:
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def _num_levels(game) -> int:
    for attr in ("_levels", "levels"):
        v = getattr(game, attr, None)
        if v is not None:
            try:
                return len(v)
            except TypeError:
                pass
    return 1


def _diff_summary(perc, before_grid, after_grid, level) -> dict:
    from ..core.types import Frame
    bg = np.asarray(before_grid); ag = np.asarray(after_grid)
    b = perc.parse(Frame(before_grid, level, 0), None)
    a = perc.parse(Frame(after_grid, level, 1), b)
    d = perc.diff(b, a, before_grid, after_grid)
    bo = {o.id: o for o in b.objects}; ao = {o.id: o for o in a.objects}
    # resize / reshape: bbox + area before -> after (vc33 bars grow, cd82 stamps) — not captured by "moved"
    reshaped = []
    for i, _, _ in d.reshaped[:12]:
        p, q = bo.get(i), ao.get(i)
        if p is not None and q is not None:
            reshaped.append({"id": int(i), "bbox_from": [int(x) for x in p.bbox], "bbox_to": [int(x) for x in q.bbox],
                             "area_from": int(p.area), "area_to": int(q.area)})
    mask = bg != ag
    box = None
    if mask.any():
        ys, xs = np.nonzero(mask)
        box = [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1]
    return {
        "changed": bool(mask.any()),
        "pixel_changes": int(mask.sum()),
        "changed_bbox": box,
        "moved": [[int(i), [int(dr), int(dc)]] for i, (dr, dc) in d.moved[:12]],
        "reshaped": reshaped,
        "recolored": [[int(i), int(o), int(n)] for i, o, n in d.recolored[:12]],
        "appeared": [{"color": int(o.color), "bbox": [int(x) for x in o.bbox]} for o in d.appeared[:12]],
        "disappeared": [{"color": int(o.color), "bbox": [int(x) for x in o.bbox]} for o in d.disappeared[:12]],
    }


def _objects(perc, grid, level) -> list[dict]:
    from ..core.types import Frame
    s = perc.parse(Frame(grid, level, 0), None)
    out = []
    for o in sorted(s.objects, key=lambda o: -o.area):
        out.append({"id": int(o.id), "color": int(o.color), "colors": [int(c) for c in o.colors],
                    "bbox": [int(x) for x in o.bbox], "area": int(o.area),
                    "center": [int((o.bbox[0] + o.bbox[2]) // 2), int((o.bbox[1] + o.bbox[3]) // 2)]})
    return out


def build_game(arc, game_id: str, *, max_click_objects: int = 40, log=print) -> dict:
    from pbg.env.wrapper import ArcadeEnv                       # noqa: F401  (ensures engine action enums import)
    from pbg.perception import Perception
    from arcengine import ActionInput, GameAction
    raw = arc.make(game_id)
    if raw is None:
        return {"game_id": game_id, "error": "could not create env"}
    game = _game_of(raw)
    if game is None:
        return {"game_id": game_id, "error": "no underlying game with set_level"}
    nlev = _num_levels(game)

    def run(action_input):
        """Apply an engine ActionInput (render-space coords, no scaling) from the current state; return (grid, state, completed)."""
        fr = game.perform_action(action_input, raw=True)
        grid = np.asarray(fr.frame[-1], dtype=np.int8) if fr.frame else None
        return grid, str(fr.state), int(getattr(fr, "levels_completed", 0) or 0)

    def obj_at(objs, row, col):
        for o in objs:
            r0, c0, r1, c1 = o["bbox"]
            if r0 <= row < r1 and c0 <= col < c1:
                return o["id"]
        return None

    levels = []
    for L in range(nlev):
        game.set_level(L)
        perc = Perception()
        init = _render(game)
        base_completed = int(getattr(game, "_score", 0) or 0)
        objs = _objects(perc, init, L)
        avail = [int(a.value if hasattr(a, "value") else a) for a in getattr(game, "_available_actions", []) or []]
        # authoritative reactive actions: the buttons + the EXACT clickable coordinates the engine offers (render space)
        try:
            valid = list(game._get_valid_actions())
        except Exception:
            valid = []
        seen_click = {(int(a.data["y"]), int(a.data["x"])) for a in valid if getattr(a, "data", None) and "x" in a.data}
        # click-anywhere games (ACTION6) react to arbitrary cells but expose only a few "valid" ones: also click the
        # centre of every object so each object's reaction is recorded (deduped against the engine's own click cells)
        extra = []
        if (not avail) or 6 in avail:
            for o in objs[:max_click_objects]:
                r, c = o["center"]
                if (int(r), int(c)) not in seen_click:
                    extra.append(ActionInput(id=GameAction.ACTION6, data={"x": int(c), "y": int(r)}))
        reactions = []
        for a in valid + extra:
            game.set_level(L)
            after, state, completed = run(a)
            if after is None:
                continue
            data = getattr(a, "data", None) or {}
            aid = a.id.value if hasattr(a.id, "value") else int(a.id)
            entry = {"action": f"ACTION{aid}"}
            if "x" in data:
                row, col = int(data["y"]), int(data["x"])
                entry.update({"action": "CLICK", "at": [row, col], "on_object": obj_at(objs, row, col)})
            entry.update(_diff_summary(perc, init, after, L))
            entry.update({"level_up": completed > base_completed, "state": state})
            reactions.append(entry)
        n_changed = sum(1 for x in reactions if x["changed"])
        levels.append({"level": L, "grid": init.tolist(), "n_objects": len(objs), "objects": objs,
                       "available_actions": avail, "n_reactions": len(reactions), "n_changed": n_changed,
                       "reactions": reactions})
        log(f"  {game_id} L{L}: {len(objs)} objects, {len(reactions)} actions swept, {n_changed} changed the board")
    return {"game_id": game_id, "n_levels": nlev, "levels": levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="experiments/pbg/datasets/reactions")
    args = ap.parse_args()
    from pbg.harness.online_runner import make_arcade
    arc = make_arcade()
    if args.all:
        games = sorted(p.name for p in (ROOT / "environment_files").iterdir() if p.is_dir())
    else:
        games = [g for g in args.games.split(",") if g]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for g in games:
        print(f"[{g}] building reaction dataset ...", flush=True)
        try:
            data = build_game(arc, g)
        except Exception as e:
            import traceback
            traceback.print_exc()
            data = {"game_id": g, "error": f"{type(e).__name__}: {e}"}
        (out / f"{g}.json").write_text(json.dumps(data, separators=(",", ":")))
        nlev = data.get("n_levels", 0)
        chg = sum(l.get("n_changed", 0) for l in data.get("levels", []))
        index.append({"game_id": g, "n_levels": nlev, "error": data.get("error"), "total_changed_reactions": chg})
        print(f"[{g}] -> {out / (g + '.json')}  ({nlev} levels)", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"index: {out / 'index.json'}")


if __name__ == "__main__":
    main()
