"""docs/029: random play (seed 0, up to 300 actions) on every game of environment_files until GAME_OVER or level 1 cleared.

    .venv/bin/python scripts/scan_random_gameover.py 2>/dev/null | grep ":"
"""
import random, sys, os
sys.path.insert(0, os.getcwd()); os.environ.setdefault("ARC_API_KEY", "local-dev")
from arcengine import GameAction
from pbg.harness.online_runner import make_arcade
arc = make_arcade()
games = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
for g in games:
    env = arc.make(g); rng = random.Random(0); raw = env.step(GameAction.RESET); res = "no game over in 300"
    for k in range(300):
        av = [int(x) for x in raw.available_actions or []]; i = rng.choice(av)
        if i == 6:
            ga = GameAction.ACTION6; ga.set_data({"x": rng.randrange(64), "y": rng.randrange(64)}); raw = env.step(ga, data=ga.action_data.model_dump())
        else:
            raw = env.step(getattr(GameAction, f"ACTION{i}"))
        if raw.levels_completed >= 1: res = f"cleared L1 at {k+1}"; break
        if str(raw.state).endswith("GAME_OVER"): res = f"GAME_OVER at {k+1}"; break
    print(f"{g}: {res}  actions={sorted(int(x) for x in raw.available_actions or [])}")
