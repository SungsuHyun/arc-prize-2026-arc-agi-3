#!/usr/bin/env python
"""tools/collect_human_log.py — record a human play log (raw.jsonl with every frame) in the terminal (spec §5/§16).

    .venv/bin/python pbg/tools/collect_human_log.py ls20 [--out pbg/data/human_logs/<player>/ls20/raw.jsonl] [--player me]

Keys: w/a/s/d or arrows = ACTION1..4 (up/down/left/right), space = ACTION5, e = ACTION7, c r c = click at (row, col)
typed as `c 12 34`, R = reset, q = quit. The board is printed as hex digits after every action."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pbg.core.types import Action, grid_to_rows  # noqa: E402
from pbg.env.wrapper import ArcadeEnv  # noqa: E402
from pbg.harness.online_runner import make_arcade  # noqa: E402

KEYS = {"w": 1, "s": 2, "a": 3, "d": 4, " ": 5, "e": 7, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "7": 7}


def show(frame, status) -> None:
    rows = grid_to_rows(frame.grid)
    print("\n".join(f"{i:2d} {r}" for i, r in enumerate(rows)))
    print(f"level {status.level}/{status.levels_total} state {status.state} actions {status.actions_used}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game"); ap.add_argument("--player", default="human"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ROOT / "pbg" / "data" / "human_logs" / a.player / a.game
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.jsonl").exists():
        (out / "raw.jsonl").unlink()
    arc = make_arcade()
    env = ArcadeEnv(arc.make(a.game), a.game, log_dir=out)
    frame = env.reset(); show(frame, env.status())
    print("keys: w/a/s/d=1/3/2/4  space=5  e=7  'c r c'=click  R=reset  q=quit")
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line:
            continue
        if line == "q":
            break
        if line == "R":
            frame = env.reset(); show(frame, env.status()); continue
        if line.startswith("c "):
            _, r, c = line.split()[:3]
            act = Action.click(int(r), int(c))
        elif line[0] in KEYS:
            act = Action.button(KEYS[line[0]])
        else:
            print("?"); continue
        rt = env.step(act)
        if rt.error:
            print("error:", rt.error); continue
        show(rt.after, rt.status)
        if rt.status_change:
            print("***", rt.status_change)
        if rt.status.state in ("WIN", "GAME_OVER"):
            print("game", rt.status.state); break
    print("saved", out / "raw.jsonl")


if __name__ == "__main__":
    main()
