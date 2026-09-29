"""orchestrator/transfer.py — level-to-level and game-to-game transfer (spec §13).

* level novelty = new colours + new shape signatures + region-structure change (0/1)
* game transfer: structure vector of the first scene -> cosine similarity with games in semantic memory -> the top-2
  games' verified models become initial hypotheses with confidence 0.3 (discarded on their first failed verification)"""
from __future__ import annotations

import math
from typing import Optional

from ..core.contracts import Hypothesis
from ..core.types import Scene


def structure_vector(scene: Scene) -> list[float]:
    strips = [r for r in scene.regions if r.kind_hint == "ui_strip"]
    kinds = {"board": 0, "panel": 0, "ui_strip": 0, "unknown": 0}
    for r in scene.regions:
        kinds[r.kind_hint] = kinds.get(r.kind_hint, 0) + 1
    hist = [0.0] * 16
    for o in scene.objects:
        hist[o.color & 15] += 1
    tot = sum(hist) or 1.0
    return [len(scene.regions), kinds["board"], kinds["panel"], kinds["ui_strip"], len(scene.objects) / 10.0, float(bool(strips)),
            scene.grid_shape[0] / 64.0, scene.grid_shape[1] / 64.0] + [h / tot for h in hist]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b)); da = math.sqrt(sum(x * x for x in a)); db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def level_novelty(prev: Optional[Scene], cur: Scene) -> tuple[int, list[int]]:
    if prev is None:
        return 0, []
    strips = {r.id for r in cur.regions if r.kind_hint == "ui_strip"}
    pc = {o.color for o in prev.objects}; ps = {o.shape_sig for o in prev.objects}
    new_colors = {o.color for o in cur.objects if o.region not in strips} - pc
    new_shapes = {o.shape_sig for o in cur.objects if o.region not in strips} - ps
    region_change = int(sorted((r.kind_hint, r.bg_color) for r in prev.regions) != sorted((r.kind_hint, r.bg_color) for r in cur.regions))
    novel_ids = [o.id for o in cur.objects if (o.color in new_colors or o.shape_sig in new_shapes) and o.region not in strips]
    return len(new_colors) + len(new_shapes) + region_change, novel_ids


def transfer_hypotheses(memory, game_id: str, scene: Scene, sandbox, *, log=None, top_k: int = 2) -> list[Hypothesis]:
    log = log or (lambda *a, **k: None)
    out: list[Hypothesis] = []
    mem = memory.load(game_id)
    # 1. this game's own verified model / unpromoted hypotheses (retry)
    if mem.world_model_code:
        m = sandbox.load(mem.world_model_code)
        if m is not None:
            out.append(Hypothesis(m, 1.0, 1.0, [], mem.world_model_code, "memory:verified", False, "", "llm"))   # re-verified on this run's log before it counts as verified
    for h in mem.hypotheses[:3]:
        if h.get("code") and "def build_model" in h["code"]:
            m = sandbox.load(h["code"])
            if m is not None:
                out.append(Hypothesis(m, float(h.get("score", 0)), float(h.get("coverage", 0)), [], h["code"], f"memory:{h.get('name')}", False, "", h.get("origin", "llm")))
    # 2. similar games (cosine similarity of the structure vectors)
    vec = structure_vector(scene)
    sims = []
    for other in memory.semantic_games():
        if other == game_id:
            continue
        v = memory.structure_vector(other)
        if v:
            sims.append((cosine(vec, v), other))
    sims.sort(reverse=True)
    for sim, other in sims[:top_k]:
        code = memory.load(other).world_model_code
        if code:
            m = sandbox.load(code)
            if m is not None:
                out.append(Hypothesis(m, 0.3, 0.0, [], code, f"transfer:{other}:{sim:.2f}", False, "", "transferred"))
                log(f"transfer from {other} (sim {sim:.2f})")
    return out
