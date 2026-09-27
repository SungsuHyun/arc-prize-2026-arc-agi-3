#!/usr/bin/env python
"""Model-free regression test for the rulebook predictor: replay the action sequence of a recorded run (per game) and
score every prediction against what the deterministic game did. Usage:
  python scripts/replay_eval.py --run 20260926-233226-3923089 --games tn36,lp85,r11l [--verbose]"""
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.CRITICAL)
from arcnav.runner import make_arcade
from rulebook.env import Game, action_label
from rulebook.predict import Evidence

ROOT = Path(__file__).resolve().parents[1]


def parse_actions(log_path: Path) -> list:
    acts = []
    for ln in log_path.read_text().splitlines():
        m = re.match(r"\[\s*\d+s a\s*(\d+) L(\d+)\] (OK |MISMATCH|obs): (MOUSE\((\d+),(\d+)\)|[A-Z0-9]+) ", ln)
        if m:
            acts.append({"action": "MOUSE", "row": int(m.group(5)), "col": int(m.group(6))} if m.group(5) else m.group(4))
    return acts


def evaluate(game_id: str, acts: list, arc, *, verbose=False, max_levels=3) -> dict:
    g = Game(arc.make(game_id), game_id); g.reset()
    per_level: dict = {}
    distrust: dict = {}
    for act in acts:
        lv = g.level
        if lv > max_levels:
            break
        ev = Evidence(g, distrust={c: n for (l, c), n in distrust.items() if l == lv})
        pred = ev.predict(act)
        before = g.frame
        res = g.step(act)
        if res.get("invalid"):
            continue
        v = ev.check(pred, g.transitions[-1], res)
        st = per_level.setdefault(lv, Counter())
        key = "level" if res["level_completed"] else "over" if res["game_over"] else "ok" if v.ok else "mismatch" if v.ok is False else "unknown"
        st[key] += 1
        if v.ok is False and isinstance(act, dict) and pred.kind == "board":
            k = (lv, before.grid[act["row"]][act["col"]]); distrust[k] = distrust.get(k, 0) + 1
        if verbose and key in ("mismatch", "unknown"):
            print(f"  L{lv} a{g.actions_used} {action_label(act)} [{key}] pred: {pred.text[:90]} | {v.text[:110]}")
        if res["game_over"]:
            g.reset()
    return {lv: dict(c) for lv, c in per_level.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--games", default="tn36,lp85,r11l")
    ap.add_argument("--verbose", action="store_true"); ap.add_argument("--levels", type=int, default=3)
    a = ap.parse_args()
    arc = make_arcade()
    log_dir = ROOT / "experiments" / "rulebook" / "results" / "logs" / a.run
    total = Counter()
    for gid in a.games.split(","):
        acts = parse_actions(log_dir / f"{gid}.log")
        t0 = time.time()
        r = evaluate(gid, acts, arc, verbose=a.verbose, max_levels=a.levels)
        line = " | ".join(f"L{lv}: " + " ".join(f"{k}={v}" for k, v in sorted(c.items())) for lv, c in sorted(r.items()))
        print(f"{gid:5} ({len(acts)} actions, {time.time() - t0:.0f}s): {line}")
        for c in r.values():
            total.update(c)
    print("TOTAL:", " ".join(f"{k}={v}" for k, v in sorted(total.items())))


if __name__ == "__main__":
    main()
