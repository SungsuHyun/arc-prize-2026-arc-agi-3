"""docs/029: vc33 level-1 score with (A) the known 8-action solution, (B) 150 random actions + level RESET first, (C) random + full game restart (RESET twice).

    .venv/bin/python scripts/probe_random_restart.py
"""
import json, random, sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault("ARC_API_KEY", "local-dev")
from arcengine import GameAction
from pbg.harness.online_runner import make_arcade

def act(env, a):
    if isinstance(a, dict):
        ga = GameAction.ACTION6; ga.set_data({"x": a["col"], "y": a["row"]}); return env.step(ga, data=ga.action_data.model_dump())
    return env.step(getattr(GameAction, a) if a.startswith("ACTION") or a == "RESET" else {"UP":GameAction.ACTION1,"DOWN":GameAction.ACTION2,"LEFT":GameAction.ACTION3,"RIGHT":GameAction.ACTION4,"SPACE":GameAction.ACTION5}[a])

def rand_action(raw, rng):
    av = [int(x) for x in raw.available_actions or []]
    i = rng.choice(av)
    return {"row": rng.randrange(64), "col": rng.randrange(64)} if i == 6 else f"ACTION{i}"

# vc33 level-1 solution from a recorded run
rows = [json.loads(l) for l in open("experiments/pbg/results/logs/20260929-084603-440659/vc33.actions.jsonl")]
sol = []
for r in rows:
    if r["kind"] == "step":
        sol.append(r["action"])
        if r["level_completed"]: break
print("vc33 L1 solution:", len(sol), "actions")

def scenario(name, n_random, full_restart):
    arc = make_arcade(); env = arc.make("vc33"); rng = random.Random(0)
    raw = env.step(GameAction.RESET)
    over = None
    for k in range(n_random):
        raw = act(env, rand_action(raw, rng))
        if raw.levels_completed >= 1: print("  (random cleared L1!)"); break
        if str(raw.state).endswith("GAME_OVER"): over = k + 1; break
    if n_random:
        raw = env.step(GameAction.RESET)                   # level reset (actions were taken)
        if full_restart: raw = env.step(GameAction.RESET)  # second RESET with no action -> whole game restart
    for a in sol:
        raw = act(env, a)
    sc = arc.get_scorecard().model_dump()
    e = [x for x in sc["environments"] if x["id"].startswith("vc33")][0]
    print(f"{name:32} random={n_random} game_over_at={over} levels={raw.levels_completed} env_score={e['score']:.2f} "
          f"runs={[ (round(r['score'],2), r['levels_completed'], r['actions'], r.get('level_actions')) for r in e['runs']]}")

scenario("A direct", 0, False)
scenario("B random, same play", 150, False)
scenario("C random, then full restart", 150, True)
