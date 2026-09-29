r"""docs/029: the vc33 A/B/C reset scenarios through the arc_agi REST app in normal vs COMPETITION mode (Flask test client, offline games).

    .venv/bin/python scripts/probe_competition_reset.py 2>&1 | grep -E "^\[|^    "
"""
import json, os, random, sys
sys.path.insert(0, os.getcwd())
import arc_agi
from arc_agi import OperationMode
from arc_agi.server import create_app
from arc_agi.scorecard import EnvironmentScorecard

rows = [json.loads(l) for l in open("experiments/pbg/results/logs/20260929-084603-440659/vc33.actions.jsonl")]
sol = []
for r in rows:
    if r["kind"] == "step":
        sol.append(r["action"])
        if r["level_completed"]: break

def run(name, n_random, n_extra_resets, competition):
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    app, api = create_app(arcade, competition_mode=competition)
    c = app.test_client(); H = {"X-API-Key": "k"}
    card = c.post("/api/scorecard/open", json={"competition_mode": True} if competition else {}, headers=H).get_json()["card_id"]
    gid = [g["game_id"] for g in c.get("/api/games", headers=H).get_json() if g["game_id"].startswith("vc33")][0]
    r = c.post("/api/cmd/RESET", json={"game_id": gid, "card_id": card}, headers=H).get_json(); guid = r["guid"]
    rng = random.Random(0); over = None
    def click(row, col):
        return c.post("/api/cmd/ACTION6", json={"game_id": gid, "card_id": card, "guid": guid, "x": col, "y": row}, headers=H).get_json()
    for k in range(n_random):
        r = click(rng.randrange(64), rng.randrange(64))
        if r["state"] == "GAME_OVER": over = k + 1; break
    notes = []
    for i in range(n_extra_resets):
        r = c.post("/api/cmd/RESET", json={"game_id": gid, "card_id": card, "guid": guid}, headers=H).get_json()
        notes.append(f"reset{i+1}: guid_same={r.get('guid')==guid} state={r.get('state')} lv={r.get('levels_completed')}")
    # a second env for the same game?
    r2 = c.post("/api/cmd/RESET", json={"game_id": gid, "card_id": card}, headers=H)
    notes.append(f"new-env RESET without guid -> http {r2.status_code} guid_same={r2.get_json().get('guid')==guid}")
    for a in sol:
        r = click(a["row"], a["col"])
    sc = api.arcade.scorecard_manager.get_scorecard(card, "k")
    es = EnvironmentScorecard.from_scorecard(sc, arcade.get_environments())
    e = [x for x in es.model_dump()["environments"] if x["id"].startswith("vc33")][0]
    print(f"{name:40} over={over} lv={r['levels_completed']} score={e['score']:.2f} runs={[(round(x['score'],2), x['levels_completed'], x['actions'], x['level_actions'][:1]) for x in e['runs']]}")
    for n in notes: print("   ", n)

for comp in (False, True):
    tag = "COMPETITION" if comp else "normal"
    run(f"[{tag}] A direct", 0, 0, comp)
    run(f"[{tag}] B random + 1 RESET", 150, 1, comp)
    run(f"[{tag}] C random + 2 RESETs (restart)", 150, 2, comp)
