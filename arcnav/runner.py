"""Play a list of games (thread pool) and write an experiments-style result JSON."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = {"base_url": "http://127.0.0.1:1234/v1", "model": "local-qwen", "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096,
                  "max_minutes": 20, "max_actions": 3000, "max_model_turns": 400, "tool_timeout": 30, "context_tokens": 32768, "jobs": 2}


def _git_info() -> dict:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "arcnav"], cwd=ROOT, text=True).strip())
        return {"commit": rev, "dirty": dirty}
    except Exception:
        return {}


def make_arcade(environments_dir: Optional[str] = None, *, competition: bool = False, base_url: Optional[str] = None):
    import arc_agi
    from arc_agi import OperationMode
    os.environ.setdefault("ARC_API_KEY", "local-dev")
    if competition:
        return arc_agi.Arcade(operation_mode=OperationMode.COMPETITION, arc_base_url=base_url or os.environ.get("ARC_BASE_URL", "http://gateway:8001/"))
    return arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=environments_dir or str(ROOT / "environment_files"))


def play_game(arc, game_id: str, cfg: dict, log_dir: Path, *, lock: threading.Lock) -> dict:
    from .agent import GameSession
    from .llm import ChatClient
    client = None if cfg.get("no_model") else ChatClient(cfg["base_url"], cfg["model"], temperature=cfg["temperature"], top_p=cfg["top_p"],
                                                          max_tokens=cfg["max_tokens"], extra_body=cfg.get("extra_body") or {})
    with lock:
        env = arc.make(game_id)
    if env is None:
        return {"game_id": game_id, "error": "could not create env"}
    sess = GameSession(env, game_id, client, log_dir=log_dir, max_minutes=cfg["max_minutes"], max_actions=cfg["max_actions"],
                       max_model_turns=cfg["max_model_turns"], tool_timeout=cfg["tool_timeout"], context_tokens=cfg["context_tokens"],
                       verbose=cfg.get("verbose", True), deadline=cfg.get("deadline"))
    try:
        return sess.play()
    except Exception as e:
        logging.exception("game %s crashed", game_id)
        sess.sandbox.close()
        return {"game_id": game_id, "error": f"{type(e).__name__}: {e}", "actions": sess.actions_used, "levels_completed": sess.level - 1}


def run(game_ids: list[str], cfg: dict, *, out_dir: Path, tag: str = "", environments_dir: Optional[str] = None, arc=None) -> dict:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    log_dir = out_dir / "logs" / run_id
    arc = arc or make_arcade(environments_dir)
    lock = threading.Lock()
    started = dt.datetime.now(dt.timezone.utc)
    with ThreadPoolExecutor(max_workers=int(cfg.get("jobs", 2))) as ex:
        games = list(ex.map(lambda g: play_game(arc, g, cfg, log_dir, lock=lock), game_ids))
    try:
        sc = arc.get_scorecard(); scd = sc.model_dump() if hasattr(sc, "model_dump") else {}
    except Exception as e:  # the competition gateway scores on its own side
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
    result = {"schema": 1, "experiment": "arcnav", "run_id": run_id, "started_at": started.isoformat(),
              "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "git": _git_info(), "tag": tag,
              "config": {"description": "arcnav: tool-using LLM harness with nav helper and solver synthesis", "games": game_ids, "params": cfg},
              "aggregate": {"score": sum(scores) / max(1, len(scores)), "levels_completed": sum(g.get("levels_completed", 0) for g in games),
                            "actions": sum(g.get("actions", 0) for g in games), "games_played": len(games)},
              "games": games, "scorecard": scd}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run-{run_id}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n========= arcnav {tag} =========")
    for g in games:
        print(f"  {g['game_id']:8} levels={g.get('levels_completed', '?'):>3}/{g.get('levels_total', '?')} actions={g.get('actions', '?'):>5} "
              f"score={g.get('score', 0) or 0:.3f} turns={g.get('model_turns', '?')} solver={g.get('solver_stored', '?')} stop={g.get('stop_reason', g.get('error'))}")
    print(f"Aggregate score: {result['aggregate']['score']:.4f}\nResult saved: {out}")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Play ARC-AGI-3 games with the arcnav agent")
    ap.add_argument("--games", default="ls20", help="comma-separated game ids or 'all'")
    ap.add_argument("--config", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "arcnav" / "results"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--minutes", type=float); ap.add_argument("--jobs", type=int); ap.add_argument("--max-actions", type=int)
    ap.add_argument("--no-model", action="store_true", help="only the environment/sandbox path (smoke test)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = dict(DEFAULT_CONFIG)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    if a.minutes: cfg["max_minutes"] = a.minutes
    if a.jobs: cfg["jobs"] = a.jobs
    if a.max_actions: cfg["max_actions"] = a.max_actions
    cfg["no_model"] = a.no_model; cfg["verbose"] = not a.quiet
    arc = make_arcade()
    if a.games == "all":
        games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
    else:
        games = [g.strip() for g in a.games.split(",") if g.strip()]
    run(games, cfg, out_dir=Path(a.out), tag=a.tag, arc=arc)


if __name__ == "__main__":
    main()
