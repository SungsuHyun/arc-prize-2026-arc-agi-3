#!/usr/bin/env python
"""Generalization check of the programmatic core (no model): for every public game, probe the keys, induce rules,
run the two-body autopilot when its trigger fires, and report what happened. Games not used during development
(everything except ls20, m0r0, vc33, sb26, tn36, r11l, cn04, s5i5) are the held-out set."""
import json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
from arcnav.runner import make_arcade
from arcnav.agent import GameSession
from arcnav.rules import induce
from arcnav.nav import NavHelper

DEV = {"ls20", "m0r0", "vc33", "sb26", "tn36", "r11l", "cn04", "s5i5"}
OUT = Path("/home/hss/code/kaggle/2026/arc-prize-2026-arc-agi-3/vendor/arcnav-runs/generalization.json")

arc = make_arcade(); games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
report = {}
for g in games:
    t0 = time.time(); row = {"heldout": g not in DEV}
    try:
        s = GameSession(arc.make(g), g, None, log_dir=Path("/home/hss/.claude/jobs/21ca0929/tmp/gen-check"), max_minutes=3, verbose=False); s.reset()
        row["valid"] = list(s.valid_actions)
        keys = [a for a in ("UP", "DOWN", "LEFT", "RIGHT") if a in s.valid_actions]
        if keys:
            s._run_tool(f"action({(keys * 2)!r})", who="gen")
        else:
            s._run_tool("nodes=[n for n in current_frame.segmentation['nodes'] if not n['hud']]; nodes.sort(key=lambda n:n['pixels'])\nfor n in nodes[:6]:\n    action({'action':'MOUSE','row':n['center'][0],'col':n['center'][1]})", who="gen")
        cur = [t for t in s.host_transitions if t.before_frame.level == t.after_frame.level == s.level]
        rules, unex = induce(cur, s.frame)
        row["rules"] = [r.text() for r in rules if r.confirmed][:8]
        row["tentative"] = len([r for r in rules if not r.confirmed])
        row["mirror"] = any(r.kind == "mirror" for r in rules)
        row["gauge"] = any(r.kind == "gauge" for r in rules)
        row["levels_by_autopilot"] = 0; row["autopilot"] = []
        for _ in range(3):
            lvl = s.level
            acted = s._maybe_autopilot()
            if not acted:
                break
            row["autopilot"].append(s.last_outcome[0][:80] if s.last_outcome else "?")
            if s.level > lvl:
                row["levels_by_autopilot"] += 1
                if any(a in s.valid_actions for a in keys):
                    s._run_tool(f"action({(keys * 2)!r})", who="gen")
            else:
                break
        row["actions"] = s.actions_used; row["level"] = s.level
        s.sandbox.close()
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:160]
    row["seconds"] = round(time.time() - t0, 1)
    report[g] = row
    print(g, {k: v for k, v in row.items() if k in ("heldout", "mirror", "gauge", "levels_by_autopilot", "level", "actions", "error", "tentative")}, flush=True)
OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
held = [g for g, r in report.items() if r["heldout"]]
print(f"\nheld-out games: {len(held)} | autopilot triggered on: {[g for g in held if report[g].get('autopilot')]} | "
      f"levels completed programmatically: {sum(report[g].get('levels_by_autopilot', 0) for g in held)} | errors: {[g for g in held if report[g].get('error')]}")
