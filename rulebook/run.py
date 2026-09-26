"""Runner: play games with the rulebook agent (thread pool), write an experiments-style result JSON + per-game logs/rulebooks."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from arcnav.llm import ChatClient
from arcnav.runner import make_arcade

from .agent import RulebookAgent
from .env import Game
from .llm_io import Model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {"base_url": "http://127.0.0.1:1234/v1", "model": "local-qwen", "temperature": 0.6, "top_p": 0.95,
                  "think_tokens": 12000, "review_tokens": 6000, "decide_tokens": 1500, "max_minutes": 20, "level_actions": 200, "max_actions": 2000,
                  "reviews_per_level": 8, "max_levels": 10, "review_think": "level", "jobs": 2}


def _git() -> dict:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "rulebook"], cwd=ROOT, text=True).strip())
        return {"commit": rev, "dirty": dirty}
    except Exception:
        return {}


def play_game(arc, game_id: str, cfg: dict, log_dir: Path, *, lock: threading.Lock, deadline: Optional[float]) -> dict:
    with lock:
        env = arc.make(game_id)
    if env is None:
        return {"game_id": game_id, "error": "could not create env"}
    game = Game(env, game_id)
    agent = RulebookAgent(game, None, log_dir=log_dir, cfg=cfg, deadline=deadline)
    if not cfg.get("no_model"):
        client = ChatClient(cfg["base_url"], cfg["model"], temperature=cfg["temperature"], top_p=cfg["top_p"], max_tokens=cfg["decide_tokens"])
        agent.model = Model(client, think_tokens=cfg["think_tokens"], decide_tokens=cfg["decide_tokens"], review_tokens=cfg.get("review_tokens", 6000), log=agent.log)
    try:
        return agent.play()
    except Exception as e:
        logging.exception("game %s crashed", game_id)
        agent.log(f"CRASH {e!r}")
        return {"game_id": game_id, "error": f"{type(e).__name__}: {e}", "actions": game.actions_used, "levels_completed": game.level - 1, "levels_total": game.levels_total}


def run(game_ids: list[str], cfg: dict, *, out_dir: Path, tag: str = "", arc=None, deadline: Optional[float] = None) -> dict:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    log_dir = out_dir / "logs" / run_id
    arc = arc or make_arcade()
    lock = threading.Lock(); started = dt.datetime.now(dt.timezone.utc)
    with ThreadPoolExecutor(max_workers=int(cfg.get("jobs", 2))) as ex:
        games = list(ex.map(lambda g: play_game(arc, g, cfg, log_dir, lock=lock, deadline=deadline), game_ids))
    try:
        sc = arc.get_scorecard(); scd = sc.model_dump() if hasattr(sc, "model_dump") else {}
    except Exception as e:
        scd = {"error": repr(e)}
    by_id = {}
    for e in scd.get("environments", []) or []:
        if e.get("runs"):
            by_id[e["id"].split("-")[0]] = e["runs"][-1]
    for g in games:
        r = by_id.get(g["game_id"].split("-")[0])
        if r:
            g["score"] = r.get("score"); g["level_scores"] = r.get("level_scores"); g["level_baseline_actions"] = r.get("level_baseline_actions")
    scores = [g.get("score") or 0.0 for g in games]
    result = {"schema": 1, "experiment": "rulebook", "run_id": run_id, "started_at": started.isoformat(), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "git": _git(), "tag": tag, "config": {"description": "rulebook agent: hypothesis rulebook + deterministic predictor + model choice + review on mismatch", "games": game_ids, "params": cfg},
              "aggregate": {"score": sum(scores) / max(1, len(scores)), "levels_completed": sum(g.get("levels_completed", 0) for g in games),
                            "games_level2plus": sum(1 for g in games if g.get("levels_completed", 0) >= 2), "actions": sum(g.get("actions", 0) for g in games), "games_played": len(games)},
              "games": games, "scorecard": scd}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run-{run_id}.json"; out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n========= rulebook {tag} =========")
    for g in games:
        print(f"  {g['game_id']:8} levels={g.get('levels_completed', '?'):>2}/{g.get('levels_total', '?')} actions={g.get('actions', '?'):>5} score={g.get('score', 0) or 0:.3f} "
              f"mism={g.get('mismatches', '?')} rev={g.get('reviews', '?')} calls={g.get('model_calls', {})} stop={g.get('stop_reason', g.get('error'))}")
    print(f"Aggregate score: {result['aggregate']['score']:.4f} | level>=2: {result['aggregate']['games_level2plus']}/{len(games)}\nResult: {out}\nLogs: {log_dir}")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Play ARC-AGI-3 games with the rulebook agent")
    ap.add_argument("--games", default="ls20"); ap.add_argument("--config"); ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "rulebook" / "results"))
    ap.add_argument("--minutes", type=float); ap.add_argument("--jobs", type=int); ap.add_argument("--level-actions", type=int)
    ap.add_argument("--no-model", action="store_true", help="deterministic fallback policy only (smoke test)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = dict(DEFAULT_CONFIG)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    if a.minutes: cfg["max_minutes"] = a.minutes
    if a.jobs: cfg["jobs"] = a.jobs
    if a.level_actions: cfg["level_actions"] = a.level_actions
    cfg["no_model"] = a.no_model
    arc = make_arcade()
    games = sorted(e.game_id.split("-")[0] for e in arc.get_environments()) if a.games == "all" else [g.strip() for g in a.games.split(",") if g.strip()]
    run(games, cfg, out_dir=Path(a.out), tag=a.tag, arc=arc)


if __name__ == "__main__":
    main()
