"""harness/online_runner.py — play games with the full orchestrator against the arc_agi engine (offline or gateway),
writing an experiments-style result JSON (same schema as rulebook/arcnav runs) + per-game event logs.

    .venv/bin/python -m pbg.harness.online_runner --games ls20,vc33 --minutes 10 --jobs 2 [--no-llm] [--budget 2000] [--tag x]"""
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

from ..core.budget import load_budget_config
from ..env.wrapper import ArcadeEnv
from ..llm.gateway import LLMGateway, load_llm_config
from ..memory.store import Memory
from ..orchestrator.loop import Orchestrator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments" / "pbg" / "results"
DEFAULT_MEMORY = ROOT / "experiments" / "pbg" / "memory"


def _git() -> dict:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "pbg"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
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


def play_game(arc, game_id: str, cfg: dict, log_dir: Path, memory: Memory, llm, lock: threading.Lock) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{game_id}.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        if cfg.get("verbose"):
            print(f"[{game_id}] {msg}", flush=True)

    with lock:
        raw_env = arc.make(game_id)
    if raw_env is None:
        return {"game_id": game_id, "error": "could not create env"}
    env = ArcadeEnv(raw_env, game_id, log_dir=memory.episodic_dir(game_id), budget_total=cfg.get("budget"), replay_path=log_dir / f"{game_id}.actions.jsonl")
    if cfg.get("policy") == "hypothesis":
        from ..hypothesis.policy import HypothesisPolicy
        if llm is None:
            return {"game_id": game_id, "error": "the hypothesis policy needs an LLM"}
        orch = HypothesisPolicy(memory=memory, llm=llm, log=log, events_dir=log_dir, max_seconds=cfg["max_minutes"] * 60, budget_cfg=cfg.get("budget_cfg"),
                                explore=int(cfg.get("explore") or 0))
    else:
        orch = Orchestrator(memory=memory, llm=llm, budget_cfg=cfg.get("budget_cfg"), log=log, events_dir=log_dir, use_llm=not cfg.get("no_llm"),
                            max_seconds=cfg["max_minutes"] * 60, planner_kwargs=cfg.get("planner", {}), max_levels=int(cfg.get("max_levels", 20)))
    try:
        r = orch.play(env, game_id, budget_total=cfg.get("budget"))
        d = r.to_json(); d["llm"] = llm.stats() if llm else {}
        return d
    except Exception as e:
        logging.exception("game %s crashed", game_id)
        log(f"CRASH {e!r}")
        st = env.status()
        return {"game_id": game_id, "error": f"{type(e).__name__}: {e}", "actions": st.actions_used, "levels_completed": st.level - 1, "levels_total": st.levels_total}


def write_run_meta(log_dir: Path, **meta) -> None:
    from scripts.eval_viewer.ids import game_hash
    """logs/<run>/run.json, written before the first action: lets the eval viewer list a run that is still in progress
    (the summary run-<id>.json only exists once every game has finished)."""
    started = meta.pop("started")
    log_dir.mkdir(parents=True, exist_ok=True)
    meta["hashes"] = {g: game_hash(meta["run_id"], g) for g in meta.get("game_ids") or []}
    meta["games"] = meta.pop("game_ids", [])
    (log_dir / "run.json").write_text(json.dumps({**meta, "started_at": started.isoformat(), "pid": os.getpid(), "git": _git()}, indent=1, default=str))


def run(game_ids: list[str], cfg: dict, *, out_dir: Path = DEFAULT_OUT, memory_root: Path = DEFAULT_MEMORY, tag: str = "", arc=None) -> dict:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    log_dir = out_dir / "logs" / run_id
    arc = arc or make_arcade()
    memory = Memory(memory_root)
    llm = None
    if not cfg.get("no_llm"):
        llm_cfg = load_llm_config()
        if cfg.get("llm"):
            llm_cfg.update(cfg["llm"])
        llm = LLMGateway(llm_cfg, cache_dir=memory_root / "llm_cache", max_calls=int(load_budget_config().get("llm_calls_max", 60)) * len(game_ids))
    lock = threading.Lock(); started = dt.datetime.now(dt.timezone.utc)
    write_run_meta(log_dir, experiment="pbg", run_id=run_id, tag=tag, started=started, game_ids=game_ids, params={k: v for k, v in cfg.items() if k != "llm"})
    import faulthandler
    faulthandler.enable()
    hard_limit = cfg["max_minutes"] * 60 + 300          # a game that is still computing this long after its deadline is reported as hung
    games = []
    ex = ThreadPoolExecutor(max_workers=int(cfg.get("jobs", 2)))
    futures = [ex.submit(play_game, arc, g, cfg, log_dir, memory, llm, lock) for g in game_ids]
    for g, fut in zip(game_ids, futures):
        try:
            games.append(fut.result(timeout=hard_limit))
        except Exception as e:      # TimeoutError -> hung game; the thread is abandoned (daemon-like) and the run still writes results
            faulthandler.dump_traceback(all_threads=True)
            games.append({"game_id": g, "error": f"hung: {type(e).__name__}", "actions": 0, "levels_completed": 0, "stop_reason": "hung"})
    ex.shutdown(wait=False, cancel_futures=True)
    from scripts.eval_viewer.ids import game_hash
    for g in games:
        g["hash"] = game_hash(run_id, g["game_id"])     # stable per game execution: the Replay viewer's lookup key
    try:
        sc = arc.get_scorecard(); scd = sc.model_dump() if hasattr(sc, "model_dump") else {}
    except Exception as e:
        scd = {"error": repr(e)}
    by_id = {}
    for e in scd.get("environments", []) or []:
        if e.get("runs"):
            by_id[e["id"].split("-")[0]] = e["runs"][-1]
    runs_by_id = {e["id"].split("-")[0]: e.get("runs") or [] for e in scd.get("environments", []) or []}
    for g in games:
        r = by_id.get(g["game_id"].split("-")[0])
        if r:
            g["score"] = r.get("score"); g["level_scores"] = r.get("level_scores"); g["level_baseline_actions"] = r.get("level_baseline_actions")
            runs = runs_by_id.get(g["game_id"].split("-")[0], [])
            best = max(runs, key=lambda x: x.get("score") or 0) if runs else r
            g["scorecard_runs"] = len(runs); g["score_best_run"] = best.get("score"); g["levels_best_run"] = best.get("levels_completed")
            if len(runs) > 1:
                g["note"] = f"{len(runs)} scorecard runs (a RESET on a fresh level restarts the game); last run scored, best run {best.get('score')}"
    scores = [g.get("score") or 0.0 for g in games]
    result = {"schema": 1, "experiment": "pbg", "run_id": run_id, "started_at": started.isoformat(), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "git": _git(), "tag": tag, "config": {"description": "pbg: perception -> probe -> world-model lab -> goal inference -> planner (spec docs/027)", "games": game_ids, "params": {k: v for k, v in cfg.items() if k != "llm"}},
              "aggregate": {"score": sum(scores) / max(1, len(scores)), "levels_completed": sum(g.get("levels_completed", 0) for g in games),
                            "games_level2plus": sum(1 for g in games if (g.get("levels_completed") or 0) >= 2), "actions": sum(g.get("actions", 0) for g in games), "games_played": len(games),
                            "llm_calls": sum(g.get("llm_calls", 0) for g in games)},
              "games": games, "scorecard": scd, "llm": llm.stats() if llm else {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run-{run_id}.json"; out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n========= pbg {tag} =========")
    for g in games:
        print(f"  {g['game_id']:8} levels={g.get('levels_completed', '?'):>2}/{g.get('levels_total', '?')} actions={g.get('actions', '?'):>5} score={g.get('score', 0) or 0:.3f} "
              f"llm={g.get('llm_calls', '?')} runs={g.get('scorecard_runs', 1)} best={g.get('score_best_run', 0) or 0:.2f} stop={g.get('stop_reason', g.get('error'))} per-level={g.get('level_actions')}")
    print(f"Aggregate score: {result['aggregate']['score']:.4f} | level>=2: {result['aggregate']['games_level2plus']}/{len(games)}\nResult: {out}\nLogs: {log_dir}")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Play ARC-AGI-3 games with the pbg system")
    ap.add_argument("--games", default="ls20"); ap.add_argument("--tag", default="")
    ap.add_argument("--policy", default="orchestrator", choices=["orchestrator", "hypothesis"], help="hypothesis: look/hypothesise/test/verify/revise/plan policy (needs an LLM)")
    ap.add_argument("--out", default=str(DEFAULT_OUT)); ap.add_argument("--memory", default=str(DEFAULT_MEMORY))
    ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--budget", type=int)
    ap.add_argument("--no-llm", action="store_true", help="priors-only induction, no model calls")
    ap.add_argument("--explore", type=int, default=0, help="hypothesis policy: explore level 1 for at most N actions before the first hypothesis (docs/029)")
    ap.add_argument("--fresh", action="store_true", help="wipe this run's games from memory first")
    ap.add_argument("--quiet", action="store_true"); ap.add_argument("--max-levels", type=int, default=20)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = {"max_minutes": a.minutes, "jobs": a.jobs, "budget": a.budget, "no_llm": a.no_llm, "verbose": not a.quiet, "max_levels": a.max_levels, "policy": a.policy, "explore": a.explore}
    arc = make_arcade()
    games = sorted(e.game_id.split("-")[0] for e in arc.get_environments()) if a.games == "all" else [g.strip() for g in a.games.split(",") if g.strip()]
    if a.fresh:
        m = Memory(Path(a.memory))
        for g in games:
            m.wipe_game(g)
    run(games, cfg, out_dir=Path(a.out), memory_root=Path(a.memory), tag=a.tag, arc=arc)


if __name__ == "__main__":
    main()
