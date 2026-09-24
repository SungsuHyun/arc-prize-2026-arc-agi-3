"""Persistent-solver policy: when the model has stored a `solve()`, the host
runs it turn after turn without the model, and stops on clear failure signals."""
from __future__ import annotations

import json
from typing import Optional

MAX_ACTIONS_PER_TURN = 12
NOOP_TURN_LIMIT = 2          # consecutive solver turns without a board change
NO_PROGRESS_ACTIONS = 80     # solver actions without a level advance
CYCLE_WINDOW = 6             # identical board seen this many times -> cycling
DEMAND_EVERY_TURNS: Optional[int] = None   # periodic proposal demand disabled (hurt free play in v012e/f)

RUN_SNIPPET = f"""
exec(compile(__solver_code, "solver.py", "exec"), globals())
__acts = list(solve() or [])
if not __acts:
    print("SOLVER_EMPTY")
else:
    __r = action(__acts[:{MAX_ACTIONS_PER_TURN}])
    print("SOLVER_RAN", len(__acts), __r.get("executed_count"), __r.get("board_changed"), __r.get("level_completed"), __r.get("game_over"))
"""


def run_snippet(code: str) -> str:
    import json as _j
    return f"__solver_code = {_j.dumps(code)}\n" + RUN_SNIPPET


def new_solver(code: str, report: dict) -> dict:
    return {"code": code, "report": report, "status": "active", "turns": 0, "actions_run": 0, "noop_turns": 0,
            "no_progress_actions": 0, "reason": None, "failure_shown": 0, "board_seen": {}}


def status_lines(solver: Optional[dict], *, model_turns: int, level_just_completed: bool) -> list[str]:
    if not solver:
        base = "Solver: none stored yet."
        if level_just_completed:
            return [base + " You just completed a level, so you know the rules: in THIS turn call propose_solver(code) with a solve() that "
                    "reproduces what worked (see the templates in the system prompt) and let the harness play the new level."]
        if DEMAND_EVERY_TURNS is not None and model_turns >= DEMAND_EVERY_TURNS:
            return [base + f" {model_turns} thinking turns spent: encode your current plan in propose_solver(code) now; the harness reports failures."]
        return [base + " When the rules are clear, call propose_solver(code) so the harness can play on without you."]
    st = solver.get("status")
    if st == "failed":
        shown = solver.get("failure_shown", 0); solver["failure_shown"] = shown + 1
        head = (f"Solver: your stored solver FAILED ({solver.get('reason')}) after {solver.get('actions_run', 0)} actions. "
                "Act manually for a few turns to learn what was wrong; when you know, repair the code and call propose_solver(code) again.")
        return [head, f"Failed solver code:\n{solver.get('code', '')[:1500]}"] if shown == 0 else [head]
    if st == "rejected":
        return [f"Solver: your last proposal was rejected: {solver.get('reason')}. Report: {json.dumps(solver.get('report', {}))[:600]}"]
    return []
