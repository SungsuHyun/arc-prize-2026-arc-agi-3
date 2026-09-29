"""docs/029: per game, level-1 weight in the game score, level-1 baseline, and what random play (uniform vs on-object clicks) reveals before GAME_OVER.

    .venv/bin/python scripts/scan_random_info.py 2>/dev/null | grep -v INFO
"""
import random, sys, os, hashlib
import numpy as np
sys.path.insert(0, os.getcwd()); os.environ.setdefault("ARC_API_KEY", "local-dev")
from arcengine import GameAction
from pbg.harness.online_runner import make_arcade
arc = make_arcade()
infos = {e.game_id.split("-")[0]: e for e in arc.get_environments()}

def grid(raw):
    return np.asarray(raw.frame[-1]) if raw.frame else None

def play(g, mode, cap=300, seed=0):
    env = arc.make(g); rng = random.Random(seed); raw = env.step(GameAction.RESET)
    g0 = grid(raw); bg = int(np.bincount(g0.ravel()).argmax())
    changed = 0; effects = set(); n = 0; end = "cap"
    for k in range(cap):
        av = [int(x) for x in raw.available_actions or []]; i = rng.choice(av); cur = grid(raw)
        if i == 6:
            if mode == "object":   # click a random non-background cell
                ys, xs = np.nonzero(cur != bg); j = rng.randrange(len(ys)); r_, c_ = int(ys[j]), int(xs[j])
            else:
                r_, c_ = rng.randrange(64), rng.randrange(64)
            ga = GameAction.ACTION6; ga.set_data({"x": c_, "y": r_}); raw = env.step(ga, data=ga.action_data.model_dump()); key = "click"
        else:
            raw = env.step(getattr(GameAction, f"ACTION{i}")); key = f"A{i}"
        n += 1; nxt = grid(raw)
        d = (nxt != cur)
        if d.sum() > 4:            # more than a counter tick
            changed += 1
            ys, xs = np.nonzero(d); effects.add((key, hashlib.md5(d.tobytes()).hexdigest()[:6]))
        if raw.levels_completed >= 1: end = "L1"; break
        if str(raw.state).endswith("GAME_OVER"): end = "over"; break
    return n, end, changed, len(effects)

print(f"{'game':5} {'lv':>2} {'L1w%':>5} {'L1base':>6} | {'uniform: n end chg eff':24} | object: n end chg eff")
for g in sorted(infos):
    e = infos[g]; nl = len(e.baseline_actions or []) or 1; w = 100 / (nl * (nl + 1) / 2)
    u = play(g, "uniform"); o = play(g, "object")
    print(f"{g:5} {nl:>2} {w:5.1f} {str((e.baseline_actions or ['?'])[0]):>6} | {u[0]:>4} {u[1]:>5} {u[2]:>4} {u[3]:>4}      | {o[0]:>4} {o[1]:>5} {o[2]:>4} {o[3]:>4}")
