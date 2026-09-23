"""Run one experiment version and record a structured result JSON.

Each experiment lives in experiments/<name>/ with:
    agent.py       - snapshot of the agent (defines MyAgent)
    config.json    - games, max_steps, free-form params
    notes.md       - hypothesis / observations (human-written)
    results/       - one JSON per run, appended over time (dashboard input)

Usage:
    .venv/bin/python scripts/run_experiment.py v001-random-baseline
    .venv/bin/python scripts/run_experiment.py v001 --game ls20 --max-steps 100
    (a unique name prefix is enough)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

EXPERIMENTS = ROOT / "experiments"


def resolve_experiment(name: str) -> Path:
    """Accept a full name or unique prefix, e.g. 'v001'."""
    exact = EXPERIMENTS / name
    if exact.is_dir():
        return exact
    matches = [d for d in EXPERIMENTS.iterdir()
               if d.is_dir() and not d.name.startswith("_")
               and d.name.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"No experiment matching {name!r}. See experiments/.")
    raise SystemExit(f"Ambiguous {name!r}: {sorted(m.name for m in matches)}")


def load_agent_class(agent_file: Path):
    spec = importlib.util.spec_from_file_location("experiment_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit(f"{agent_file} must define a class named `MyAgent`")
    return module.MyAgent


def play_one(args: tuple) -> dict:
    """Play a single game in a fresh process (used by --jobs). Returns the
    flat game record plus that game's scorecard entry."""
    agent_file, game_id, max_steps, seed, exp_name = args
    import random
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    os.environ.setdefault("ARC_API_KEY", "local-dev")
    try:
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make(game_id)
    except Exception:
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
        env = arc.make(game_id)
    AgentCls = load_agent_class(Path(agent_file))
    AgentCls.MAX_ACTIONS = max_steps
    if seed is not None:
        random.seed(seed)
        random.seed = lambda *a, **k: None   # type: ignore[assignment]
    agent = AgentCls(card_id="local-exp", game_id=game_id,
                     agent_name=f"{exp_name}.{game_id}", ROOT_URL="http://localhost",
                     record=False, arc_env=env, tags=["experiment", exp_name])
    agent.main()
    final = agent.frames[-1]
    record = {"game_id": game_id, "state": str(final.state),
              "levels_completed": final.levels_completed, "actions": agent.action_counter}
    extra = getattr(agent, "metrics", None)
    if isinstance(extra, dict):
        record.update({k: v for k, v in extra.items() if k not in record})
    sc = arc.get_scorecard()
    env_score = sc.find_environment(game_id) if sc is not None else None
    if env_score is not None:
        record["score"] = env_score.score
        record["win_levels"] = env_score.level_count
        record["_env_scorecard"] = env_score.model_dump(mode="json")
    return record


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", help="experiment folder name (or unique prefix)")
    p.add_argument("--game", default=None,
                   help="comma-separated short ids; overrides config.json games")
    p.add_argument("--max-steps", type=int, default=None,
                   help="overrides config.json max_steps")
    p.add_argument("--tag", default=None, help="optional label stored in the result")
    p.add_argument("--jobs", type=int, default=1,
                   help="play this many games in parallel processes (score is the mean "
                        "of per-game scores, identical to the scorecard formula)")
    p.add_argument("--seed", type=int, default=None,
                   help="override the agent's own random seed (variance checks); "
                        "recorded in config.seed_override")
    args = p.parse_args()

    exp_dir = resolve_experiment(args.experiment)
    config = json.loads((exp_dir / "config.json").read_text())
    games_cfg = args.game or config.get("games") or None
    if isinstance(games_cfg, list):
        games_cfg = ",".join(games_cfg)
    max_steps = args.max_steps or config.get("max_steps", 200)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # NORMAL은 시작 시 three.arcprize.org에 접근(익명 키/게임 목록)하는데,
    # 네트워크 불안정 시 예외가 전파됨. 게임이 캐시돼 있으면 OFFLINE로 폴백.
    os.environ.setdefault("ARC_API_KEY", "local-dev")
    try:
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        all_envs = arc.get_environments()
    except Exception as e:
        print(f"NORMAL mode failed ({type(e).__name__}) — falling back to OFFLINE cache")
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
        all_envs = arc.get_environments()
    if games_cfg:
        wanted = {g.strip().split("-")[0] for g in games_cfg.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]

    AgentCls = load_agent_class(exp_dir / "agent.py")
    # max_steps fully overrides the agent's own MAX_ACTIONS (raise or lower);
    # the agent's is_done() can still stop earlier.
    AgentCls.MAX_ACTIONS = max_steps
    if args.seed is not None:
        # Agents call random.seed(<fixed>) in __init__; neutralise that so the
        # run follows our seed instead (same code, different trajectory).
        import random
        random.seed(args.seed)
        random.seed = lambda *a, **k: None   # type: ignore[assignment]

    started = datetime.now(timezone.utc)
    games = []
    if args.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        work = [(str(exp_dir / "agent.py"), g, max_steps, args.seed, exp_dir.name) for g in game_ids]
        print(f"=== {len(game_ids)} games, {args.jobs} parallel workers ===", flush=True)
        with ctx.Pool(args.jobs) as pool:
            for rec in pool.imap_unordered(play_one, work):
                games.append(rec)
                print(f"  -> {rec['game_id']:6} state={rec['state']} levels={rec['levels_completed']} "
                      f"actions={rec['actions']} score={rec.get('score')}", flush=True)
        games.sort(key=lambda r: game_ids.index(r["game_id"]))
        env_cards = [g.pop("_env_scorecard", None) for g in games]
        scores = [g.get("score", 0.0) for g in games]
        score = sum(scores) / len(scores) if scores else None
        scorecard = {"parallel": True, "environments": [c for c in env_cards if c]}
        run_id = started.strftime("%Y%m%d-%H%M%S") + (f"-s{args.seed}" if args.seed is not None else "") + f"-{os.getpid() % 1000:03d}"
        result = {
            "schema": 1, "experiment": exp_dir.name, "run_id": run_id,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "git": git_info(), "tag": args.tag,
            "config": {**config, "games": game_ids, "max_steps": max_steps,
                       "seed_override": args.seed, "jobs": args.jobs},
            "aggregate": {"score": score,
                          "levels_completed": sum(g.get("levels_completed", 0) for g in games),
                          "actions": sum(g.get("actions", 0) for g in games),
                          "games_played": len(games)},
            "games": games, "scorecard": scorecard,
        }
        results_dir = exp_dir / "results"; results_dir.mkdir(exist_ok=True)
        out = results_dir / f"run-{run_id}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n========= {exp_dir.name} =========")
        for g in games:
            print(f"  {g['game_id']:8} levels={g.get('levels_completed', '?'):>3} "
                  f"actions={g.get('actions', '?'):>5}  score={g.get('score', '?')}")
        print(f"\nAggregate score: {score}")
        print(f"Result saved: {out.relative_to(ROOT)}")
        return
    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===", flush=True)
        env = arc.make(game_id)
        if env is None:
            games.append({"game_id": game_id, "error": "could not create env"})
            continue
        agent = AgentCls(
            card_id="local-exp",
            game_id=game_id,
            agent_name=f"{exp_dir.name}.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["experiment", exp_dir.name],
        )
        agent.main()
        final = agent.frames[-1]
        record = {
            "game_id": game_id,
            "state": str(final.state),
            "levels_completed": final.levels_completed,
            "actions": agent.action_counter,
        }
        # agents may expose extra metrics (e.g. unique_states) via .metrics
        extra = getattr(agent, "metrics", None)
        if isinstance(extra, dict):
            record.update({k: v for k, v in extra.items() if k not in record})
        games.append(record)
        print(f"  -> state={final.state} levels={final.levels_completed} "
              f"actions={agent.action_counter}")

    sc = arc.get_scorecard()
    scorecard = sc.model_dump(mode="json") if sc is not None else None
    score = getattr(sc, "score", None) if sc is not None else None

    # Merge per-game score from the scorecard into the flat games list.
    if sc is not None:
        for g in games:
            env_score = sc.find_environment(g["game_id"])
            if env_score is not None:
                g["score"] = env_score.score
                g["win_levels"] = env_score.level_count

    run_id = started.strftime("%Y%m%d-%H%M%S") + (f"-s{args.seed}" if args.seed is not None else "") + f"-{os.getpid() % 1000:03d}"
    result = {
        "schema": 1,
        "experiment": exp_dir.name,
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "tag": args.tag,
        "config": {**config, "games": game_ids, "max_steps": max_steps,
                   "seed_override": args.seed},
        "aggregate": {
            "score": score,
            "levels_completed": sum(g.get("levels_completed", 0) for g in games),
            "actions": sum(g.get("actions", 0) for g in games),
            "games_played": len(games),
        },
        "games": games,
        "scorecard": scorecard,
    }

    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"run-{run_id}.json"
    out.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n========= {exp_dir.name} =========")
    for g in games:
        print(f"  {g['game_id']:8} levels={g.get('levels_completed', '?'):>3} "
              f"actions={g.get('actions', '?'):>5}  score={g.get('score', '?')}")
    print(f"\nAggregate score: {score}")
    print(f"Result saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
