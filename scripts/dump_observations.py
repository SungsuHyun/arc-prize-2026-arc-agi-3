"""Print an experiment agent's encoded observations for offline inspection.

Plays one game with the experiment's own agent and prints the observation
samples it collected (agents expose `obs_samples: list[(action_counter, text)]`;
v003+). Used to eyeball encoding correctness and measure size before wiring
an LLM to it.

Usage:
    .venv/bin/python scripts/dump_observations.py v003 --game ls20 --max-steps 120
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode

sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import load_agent_class, resolve_experiment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment")
    p.add_argument("--game", default="ls20")
    p.add_argument("--max-steps", type=int, default=120)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    exp_dir = resolve_experiment(args.experiment)
    AgentCls = load_agent_class(exp_dir / "agent.py")
    AgentCls.MAX_ACTIONS = args.max_steps

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    agent = AgentCls(card_id="obs-dump", game_id=args.game,
                     agent_name=f"{exp_dir.name}.dump", ROOT_URL="http://localhost",
                     record=False, arc_env=env, tags=["obs-dump"])
    agent.main()

    samples = getattr(agent, "obs_samples", [])
    if not samples:
        raise SystemExit(f"{exp_dir.name} agent has no obs_samples "
                         "(needs the v003+ encoder).")
    for counter, text in samples:
        est_tokens = len(text) // 3
        print(f"{'='*70}\n--- observation @ action {counter} "
              f"({len(text)} chars, ~{est_tokens} tokens) ---")
        print(text)
    m = getattr(agent, "metrics", {})
    print(f"\n{'='*70}\nmetrics: {m}")


if __name__ == "__main__":
    main()
