"""Per-game loop: environment stepping, model turns, python tool, solver auto-run."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from arcengine import GameAction, GameState

from . import solver as solver_policy
from .frame import Frame, masked_ascii, summarize_diff
from .llm import ChatClient, ContextLengthError
from .nav import NavHelper
from .prompts import PROPOSE_TOOL, PYTHON_TOOL, system_prompt, turn_header
from .sandbox import Sandbox

MODEL_TO_ENGINE = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5", "MOUSE": "ACTION6"}
ENGINE_TO_MODEL = {v: k for k, v in MODEL_TO_ENGINE.items()}
MAX_ACTIONS_PER_CALL = 24   # blind 50-action batches walked straight into game over on s5i5
MAX_ACTIONS_PER_TOOL_RUN = 30   # also caps loops of single-action calls inside one python tool run
PROPOSAL_ATTEMPTS = 0   # >0: after a level completion, action() is blocked and propose_solver is forced via tool_choice for this many turns.
                        # Set E (forced, 3 turns): 0.75, level>=2 0/4 — forced solvers burned 50–80 actions per level; disabled by default.


class _T:  # transition view for the host-side NavHelper
    def __init__(self, action, before_frame, after_frame):
        self.action, self.before_frame, self.after_frame = action, before_frame, after_frame


class GameSession:
    def __init__(self, env, game_id: str, client: Optional[ChatClient], *, log_dir: Path, max_minutes: float = 20.0,
                 max_actions: int = 3000, max_model_turns: int = 400, keep_full_turns: int = 3, tool_timeout: int = 30,
                 context_tokens: int = 32768, verbose: bool = True, deadline: Optional[float] = None):
        self.env, self.game_id, self.client = env, game_id, client
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_minutes, self.max_actions, self.max_model_turns = max_minutes, max_actions, max_model_turns
        self.keep_full_turns, self.tool_timeout, self.context_tokens, self.verbose = keep_full_turns, tool_timeout, context_tokens, verbose
        self.deadline = deadline   # absolute epoch seconds (global run cap), optional
        self.transitions: list[dict] = []          # payloads for the sandbox
        self.host_transitions: list[_T] = []       # Frame views for the host NavHelper
        self.frame: Optional[Frame] = None
        self.level, self.levels_total, self.state = 1, 0, "NOT_PLAYED"
        self.valid_actions: list[str] = []
        self.actions_used = self.level_actions = self.resets = self.step_no = 0
        self.level_action_log: list[int] = []
        self.solver: Optional[dict] = None
        self.model_turns = self.solver_turns = 0
        self.level_just_completed = False
        self.last_outcome: list[str] = []
        self.notes = ""
        self._run_actions = 0
        self.game_overs_this_level = 0
        self.proposal_required = 0   # >0: action() blocked until propose_solver is called (set after a level completion)
        self.messages: list[dict] = []
        self.t0 = time.time()
        self.transcript = open(self.log_dir / f"{game_id}.log", "a")
        self.events = open(self.log_dir / f"{game_id}.jsonl", "a")
        self.sandbox = Sandbox(action_handler=self._on_action, solver_handler=self._on_solver, state_provider=self._state, timeout=tool_timeout)

    # ------------------------------------------------------------ environment
    def _apply(self, raw) -> None:
        self.state = str(raw.state).split(".")[-1]
        self.levels_total = int(raw.win_levels or 0)
        self.level = int(raw.levels_completed) + 1
        self.valid_actions = [ENGINE_TO_MODEL.get(f"ACTION{a}", f"ACTION{a}") for a in (raw.available_actions or [])]
        grids = [g.tolist() if hasattr(g, "tolist") else g for g in raw.frame]
        if grids:
            self.frame = Frame(grids[-1], step=self.step_no, level=self.level)

    def reset(self) -> None:
        raw = self.env.step(GameAction.RESET)
        self.resets += 1; self._apply(raw)
        self.level_actions = 0

    def _step_engine(self, act: dict):
        name = act["action"]
        ga = GameAction.from_name(MODEL_TO_ENGINE[name]) if hasattr(GameAction, "from_name") else GameAction[MODEL_TO_ENGINE[name]]
        data = {"x": int(act["col"]), "y": int(act["row"])} if name == "MOUSE" else {}
        ga.set_data(data) if data else None
        return self.env.step(ga, data=ga.action_data.model_dump())

    def execute(self, actions: list[dict]) -> dict:
        """Run actions until one completes a level or ends the game. Records transitions."""
        results, executed, changed_any, level_completed, game_over, stopped = [], 0, False, False, False, None
        if len(actions) > MAX_ACTIONS_PER_CALL:
            actions = actions[:MAX_ACTIONS_PER_CALL]; stopped = f"only the first {MAX_ACTIONS_PER_CALL} actions of a call are executed; look at the board, then continue"
        for act in actions:
            if act["action"] not in self.valid_actions and self.valid_actions:
                stopped = f"{act['action']} is not a valid action now ({self.valid_actions})"; break
            if self.actions_used >= self.max_actions:
                stopped = "action budget exhausted"; break
            before = self.frame; prev_level = self.level
            raw = self._step_engine(act)
            self.step_no += 1; self.actions_used += 1; self.level_actions += 1; executed += 1
            self._apply(raw)
            after = self.frame
            label = act["action"] if act["action"] != "MOUSE" else {"action": "MOUSE", "row": act["row"], "col": act["col"]}
            self.transitions.append({"action": label, "before": before.to_payload(), "after": after.to_payload()})
            self.host_transitions.append(_T(label if isinstance(label, str) else "MOUSE", before, after))
            ch = before.ascii != after.ascii; changed_any |= ch
            results.append({"action": label, "changed": ch})
            if self.level > prev_level or self.state == "WIN":
                level_completed = True; self.level_action_log.append(self.level_actions); self.level_actions = 0; self.game_overs_this_level = 0
                self._log(f"*** level {prev_level} completed after {self.level_action_log[-1]} actions (total {self.actions_used})")
                stopped = "level completed" if self.state != "WIN" else "game won"; break
            if self.state == "GAME_OVER":
                game_over = True; self.game_overs_this_level += 1; self._log("*** game over -> reset"); self.reset(); stopped = "game over (reset done, level restarted)"; break
        return {"executed_count": executed, "board_changed": changed_any, "level_completed": level_completed, "game_over": game_over,
                "stopped_reason": stopped, "results": results[-12:]}

    # ------------------------------------------------------------ sandbox hooks
    def _state(self) -> dict:
        return {"frame": self.frame.to_payload() if self.frame else None, "valid_actions": self.valid_actions, "level": self.level,
                "levels_total": self.levels_total, "transitions": self.transitions[-400:], "notes": self.notes}

    def _on_action(self, actions: list[dict]) -> tuple[dict, dict]:
        if self.proposal_required > 0:
            return {"executed_count": 0, "board_changed": False, "level_completed": False, "game_over": False,
                    "stopped_reason": "blocked: a level was just completed, so call propose_solver(code) first (inspection is fine, actions are not)",
                    "results": []}, self._state()
        left = MAX_ACTIONS_PER_TOOL_RUN - self._run_actions
        if left <= 0:
            return {"executed_count": 0, "board_changed": False, "level_completed": False, "game_over": False,
                    "stopped_reason": f"this tool run already executed {MAX_ACTIONS_PER_TOOL_RUN} actions: finish the call, read the board and the outcome, then continue in the next turn",
                    "results": []}, self._state()
        res = self.execute(actions[:left])
        self._run_actions += res["executed_count"]
        if res["level_completed"]:
            self.level_just_completed = True
            if self.solver and self.solver["status"] == "active":
                self.solver["no_progress_actions"] = 0
            elif self.state != "WIN":
                self.proposal_required = PROPOSAL_ATTEMPTS
        return res, self._state()

    def _on_solver(self, code: str, report: dict) -> dict:
        if report.get("ok"):
            self.solver = solver_policy.new_solver(code, report)
            self._log("solver stored:\n" + code)
            return {**report, "stored": True, "note": "Solver stored; the harness will run solve() automatically from the next turn."}
        self.solver = {"status": "rejected", "reason": report.get("reason"), "report": report, "code": code}
        self._log(f"solver rejected: {report.get('reason')}")
        return {**report, "stored": False}

    # ------------------------------------------------------------ model turns
    def _log(self, text: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s a={self.actions_used} L{self.level}] {text}"
        self.transcript.write(line + "\n"); self.transcript.flush()
        if self.verbose:
            print(f"{self.game_id} {line[:220]}", flush=True)

    def _event(self, **kw) -> None:
        self.events.write(json.dumps({"t": round(time.time() - self.t0, 1), **kw}, default=str) + "\n"); self.events.flush()

    def _nav_lines(self) -> list[str]:
        try:
            cur = [t for t in self.host_transitions if t.before_frame.level == t.after_frame.level == self.level]
            nav = NavHelper(cur, self.frame)
            if nav.moves:
                lines = ["Navigation helper summary:", nav.summary()]
                ready = []
                for t in [t for t in nav.targets() if t.get("path_len")][:3]:
                    path = nav.path_to(t["row"], t["col"])
                    if path:
                        ready.append(f"  nav.path_to({t['row']}, {t['col']}) -> {path}  # colour {t.get('color')}, {'visited' if t.get('visited') else 'unvisited'}")
                fr = nav.frontier()
                if fr and (path := nav.path_to(fr[0], fr[1])):
                    ready.append(f"  nav.path_to({fr[0]}, {fr[1]}) -> {path}  # nearest unexplored block")
                if ready:
                    lines.append("Ready-made routes (pass one straight to action(...)):\n" + "\n".join(ready))
                lines += self._gauge_lines(nav, cur)
                return lines
            g = nav.gauge()
            if g:
                return [f"Action gauge detected: colour {g.get('color')}, {g.get('size')} cells, {g.get('per_action')} per action, ~{g.get('actions_left')} actions left. "
                        "This strip is a counter of your remaining actions (a HUD), NOT a goal to fill or empty: ignore it when choosing targets and finish the level before it runs out."]
        except Exception as e:  # never let the helper break a turn
            return [f"Navigation helper unavailable ({type(e).__name__})."]
        if any(a in self.valid_actions for a in ("UP", "DOWN", "LEFT", "RIGHT")):
            return ["Navigation helper: no movement learned yet. Press each of UP/DOWN/LEFT/RIGHT once, then read nav.summary()."]
        return ["Navigation helper: this level has no movement keys (click game). Use current_frame.segmentation to enumerate objects; nodes with hud=True are edge strips (counters), not targets."]

    def _seconds_left(self) -> float:
        left = self.max_minutes * 60 - (time.time() - self.t0)
        if self.deadline is not None:
            left = min(left, self.deadline - time.time())
        return left

    def _gauge_lines(self, nav, cur) -> list[str]:
        g = nav.gauge()
        if not g:
            return []
        out = [f"Gauge budget: ~{g.get('actions_left')} actions left before the level restarts (colour {g.get('color')}, {g.get('per_action')} per action). "
               "Any route longer than that fails: pick a target reachable within the budget or find what refills the gauge first."]
        # refill clues: transitions of this level where the gauge object grew
        color = g.get("color"); grew = []
        for t in cur[-60:]:
            try:
                b = sum(1 for row in t.before_frame.grid for v in row if v == color)
                a = sum(1 for row in t.after_frame.grid for v in row if v == color)
            except Exception:
                continue
            if a > b:
                grew.append(f"{t.action} (+{a - b})")
        if grew:
            out.append("Gauge REFILLED after: " + ", ".join(grew[-4:]) + " — whatever the avatar touched then refills it; plan to collect such objects on the way.")
        return out

    def _budget_line(self) -> str:
        left = self._seconds_left()
        return f"Time left: {max(0, left) / 60:.1f} min. Action budget left: {self.max_actions - self.actions_used}."

    def _user_message(self) -> str:
        parts = [turn_header(level=self.level, levels_total=self.levels_total, actions_used=self.actions_used, level_actions=self.level_actions,
                             valid_actions=self.valid_actions, budget_line=self._budget_line())]
        if self.last_outcome:
            parts += self.last_outcome
        parts += self._nav_lines()
        parts.append("Your notes (the sandbox variable `notes`; keep it current instead of re-deriving the rules each turn):\n" +
                     (self.notes or "(empty — write what each key does, the goal hypothesis and the next plan)"))
        if self.proposal_required > 0:
            parts.append(f"REQUIRED: you just completed a level, so you know the rules. Call propose_solver(code) with a solve() that reproduces "
                         f"the winning strategy from `transitions` of the previous level (use nav.path_to / segmentation, not fixed coordinates). "
                         f"action() stays blocked until you do ({self.proposal_required} turn(s) left before the requirement is waived); "
                         "inspecting the board is allowed.")
        parts += solver_policy.status_lines(self.solver, model_turns=self.model_turns, level_just_completed=self.level_just_completed)
        self.level_just_completed = False
        if self.solver and self.solver.get("status") in ("failed", "rejected"):
            self.solver["status"] = "shown"
        parts.append("Current board:\n" + self.frame.ascii)
        return "\n".join(parts)

    def _trim_context(self) -> None:
        """Older turns lose their board text and long tool output; on overflow drop the oldest turns."""
        user_idx = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        for i in user_idx[:-self.keep_full_turns]:
            c = self.messages[i]["content"]
            if "Current board:" in c:
                self.messages[i]["content"] = c.split("Current board:")[0] + "[board omitted]"
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        for i in tool_idx[:-self.keep_full_turns]:
            c = self.messages[i]["content"]
            if len(c) > 800:
                self.messages[i]["content"] = c[:600] + "\n...[older tool output trimmed]..."
        # token estimate calibrated on the last real prompt_tokens (hex boards tokenize at ~1 token per character)
        budget = (self.context_tokens - self.client.max_tokens) * 0.85 if self.client else self.context_tokens * 0.8
        est = self._estimate_tokens()
        while est > budget and len(self.messages) > 4:
            self._drop_oldest_turn(); est = self._estimate_tokens()

    def _estimate_tokens(self) -> float:
        chars = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) for m in self.messages)
        return chars * getattr(self, "_tok_per_char", 0.45) + 300

    def _drop_oldest_turn(self) -> None:
        # messages[0] is the system prompt; a turn is user, assistant(, tool)
        k = 1
        while k < len(self.messages) and self.messages[k]["role"] != "user":
            k += 1
        end = k + 1
        while end < len(self.messages) and self.messages[end]["role"] != "user":
            end += 1
        if end - k >= len(self.messages) - 1:  # never drop the only turn
            return
        del self.messages[k:end]

    def _run_tool(self, code: str, *, who: str) -> str:
        self._run_actions = 0
        res = self.sandbox.run(code, timeout=self.tool_timeout)
        if res.get("notes"):
            self.notes = res["notes"]
        out = res["stdout"]
        if res["error"]:
            out += ("\n" if out else "") + "Error: " + res["error"]
        self._event(kind="tool", who=who, code=code, stdout=res["stdout"][:4000], error=res["error"], actions=res["actions_executed"])
        return out or "(no output)"

    def _outcome_lines(self, before_n: int) -> list[str]:
        new = self.host_transitions[before_n:]
        extra = []
        if self.game_overs_this_level:
            extra.append(f"GAME OVER count on this level: {self.game_overs_this_level} (each one reset the level). The action limit or the gauge ran out "
                         "before the goal: stop repeating the same sweep, re-read what changed the board, and try a different hypothesis.")
        if not new:
            return ["Last turn executed no actions."] + extra
        acts = [t.action if isinstance(t.action, str) else "MOUSE" for t in new]
        diff = summarize_diff(new[0].before_frame, new[-1].after_frame)
        return [f"Last turn executed {len(new)} action(s): {', '.join(acts[:12])}{'...' if len(acts) > 12 else ''}. "
                f"Board change over the turn: {json.dumps(diff, default=str)[:700]}"] + extra

    def model_turn(self) -> bool:
        """One model call + tool execution. Returns False when the model produced nothing usable."""
        self.messages.append({"role": "user", "content": self._user_message()})
        self._trim_context()
        tools = [PYTHON_TOOL, PROPOSE_TOOL]
        # after a level completion: one free inspection turn, then the proposal call is forced via tool_choice
        choice = {"type": "function", "function": {"name": "propose_solver"}} if 0 < self.proposal_required < PROPOSAL_ATTEMPTS else "auto"
        r = None
        for attempt in range(6):
            try:
                r = self.client.chat(self.messages, tools=tools, tool_choice=choice); break
            except ContextLengthError:
                self._tok_per_char = min(1.2, getattr(self, "_tok_per_char", 0.45) * 1.3)   # we under-estimated: be more aggressive
                n_before = len(self.messages)
                for _ in range(2 + attempt):
                    self._drop_oldest_turn()
                if len(self.messages) == n_before:   # nothing left to drop: shrink the board text of the current turn
                    self.messages[-1]["content"] = self.messages[-1]["content"].split("Current board:")[0] + "[board omitted: context full]"
                self._log(f"context overflow: dropped turns ({n_before} -> {len(self.messages)} messages), retry {attempt + 1}")
        if r is None:
            raise RuntimeError("context overflow could not be resolved")
        # calibrate the estimate with the real prompt size
        pt = int(r.usage.get("prompt_tokens") or 0); chars = sum(len(m.get("content") or "") for m in self.messages)
        if pt and chars:
            self._tok_per_char = max(0.25, min(1.2, pt / chars))
        self.model_turns += 1
        msg = r.message
        self._event(kind="model", turn=self.model_turns, content=msg.get("content", "")[:2000], reasoning=r.reasoning[:1500],
                    tool_calls=msg.get("tool_calls"), latency=round(r.latency, 1), usage=r.usage)
        self.messages.append({"role": "assistant", "content": msg.get("content", ""), **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})})
        self._log(f"model turn {self.model_turns} ({r.latency:.0f}s, {r.usage.get('completion_tokens', '?')} tok): {(msg.get('content') or r.reasoning)[:200]!r}")
        calls = msg.get("tool_calls") or []
        if not calls:
            if r.finish_reason == "length":
                self.last_outcome = ["Your last reply was cut off by the output limit while you were still reasoning, so nothing happened. "
                                     "Reason briefly (a few sentences) and act with one python tool call; let the code do the arithmetic."]
            else:
                self.last_outcome = ["Your last reply contained no python tool call; nothing happened. Reply with exactly one python tool call."]
            return False
        before_n = len(self.host_transitions)
        for call in calls[:1]:
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                code = args.get("code") or ""
            except Exception:
                code = ""
            if call["function"].get("name") == "propose_solver" and code:
                code = f"print(propose_solver({json.dumps(code)}))"   # routed through the sandbox's verifier
            out = self._run_tool(code, who="model") if code else "Error: tool call had no `code` argument"
            if self.proposal_required > 0:
                if "propose_solver(" in code:
                    self.proposal_required = 0
                else:
                    self.proposal_required -= 1
                    if self.proposal_required == 0:
                        out += "\n(The solver requirement is now waived; you may act directly.)"
            self._log(f"tool ({len(self.host_transitions) - before_n} actions): {out[:300]!r}")
            self.messages.append({"role": "tool", "tool_call_id": call.get("id", "call_0"), "content": out[:6000]})
        for call in calls[1:]:  # extra calls are acknowledged, not executed
            self.messages.append({"role": "tool", "tool_call_id": call.get("id", "call_x"), "content": "Skipped: only one python call per reply is executed."})
        self.last_outcome = self._outcome_lines(before_n)
        return True

    def solver_turn(self) -> None:
        s = self.solver
        before_n = len(self.host_transitions); prev_level = self.level; prev_go = self.game_overs_this_level + self.resets
        budget = solver_policy.MAX_ACTIONS_PER_TURN
        try:  # never let a solver turn spend the last actions of a gauge blindly
            cur = [t for t in self.host_transitions if t.before_frame.level == t.after_frame.level == self.level]
            g = NavHelper(cur, self.frame).gauge() if self.frame else None
            if g and g.get("actions_left") is not None:
                budget = min(budget, int(g["actions_left"]) - 2)
        except Exception:
            pass
        if budget <= 0:
            s["status"], s["reason"] = "failed", "the action gauge is nearly exhausted; the solver is paused so you can decide what to do with the last actions"
            self._log("solver paused: gauge nearly exhausted"); self.last_outcome = [f"The stored solver was paused: {s['reason']}."]
            return
        out = self._run_tool(solver_policy.run_snippet(s["code"], budget), who="solver")
        s["turns"] += 1; self.solver_turns += 1
        n = len(self.host_transitions) - before_n; s["actions_run"] += n
        # progress = change outside HUD strips (a ticking gauge alone is not progress)
        changed = any(masked_ascii(t.before_frame) != masked_ascii(t.after_frame) for t in self.host_transitions[before_n:])
        self._log(f"solver turn {s['turns']}: {n} actions, changed={changed}: {out[:160]!r}")
        if "Error:" in out:
            s["status"], s["reason"] = "failed", "solve() raised: " + out.split("Error:", 1)[1].strip()[:300]
        elif "SOLVER_EMPTY" in out:
            s["status"], s["reason"] = "failed", "solve() returned [] (it could not decide)"
        elif n == 0:
            s["status"], s["reason"] = "failed", "solve() produced no executable action"
        elif self.game_overs_this_level + self.resets > prev_go and self.level <= prev_level:
            s["status"], s["reason"] = "failed", "its actions ran into a GAME OVER (gauge/action limit exhausted before the goal)"
        if self.level > prev_level:
            s["noop_turns"] = 0; s["no_progress_actions"] = 0; s["board_seen"] = {}
            return
        s["noop_turns"] = 0 if changed else s["noop_turns"] + 1
        s["no_progress_actions"] += n
        key = masked_ascii(self.frame) if self.frame else ""
        s["board_seen"][key] = s["board_seen"].get(key, 0) + 1
        if s["status"] == "active":
            if s["noop_turns"] >= solver_policy.NOOP_TURN_LIMIT:
                s["status"], s["reason"] = "failed", f"{s['noop_turns']} consecutive solver turns without any board change"
            elif s["no_progress_actions"] >= solver_policy.NO_PROGRESS_ACTIONS:
                s["status"], s["reason"] = "failed", f"{s['no_progress_actions']} solver actions without completing the level"
            elif s["board_seen"][key] >= solver_policy.CYCLE_WINDOW:
                s["status"], s["reason"] = "failed", "the solver keeps returning to the same board (cycle)"
        if s["status"] == "failed":
            self._log(f"solver failed: {s['reason']}")
            self.last_outcome = [f"The stored solver ran {s['actions_run']} actions in total and was stopped: {s['reason']}."]

    # ------------------------------------------------------------ main loop
    def _out_of_budget(self) -> Optional[str]:
        if self.state == "WIN":
            return "won"
        if self._seconds_left() <= 0:
            return "time"
        if self.actions_used >= self.max_actions:
            return "actions"
        if self.model_turns >= self.max_model_turns:
            return "turns"
        return None

    def play(self) -> dict:
        self.reset()
        self.messages = [{"role": "system", "content": system_prompt()}]
        self._log(f"start: {self.levels_total} levels, valid={self.valid_actions}")
        idle = 0
        while not (why := self._out_of_budget()):
            if self.solver and self.solver.get("status") == "active":
                self.solver_turn(); continue
            if self.client is None:
                why = "no model"; break
            ok = self.model_turn()
            idle = 0 if ok else idle + 1
            if idle >= 6:
                why = "model produced no tool calls"; break
        summary = {"game_id": self.game_id, "state": self.state, "levels_completed": self.level - 1 if self.state != "WIN" else self.levels_total,
                   "levels_total": self.levels_total, "actions": self.actions_used, "resets": self.resets, "model_turns": self.model_turns,
                   "solver_turns": self.solver_turns, "solver_stored": bool(self.solver and self.solver.get("code")),
                   "level_actions": self.level_action_log, "elapsed_s": round(time.time() - self.t0, 1), "stop_reason": why,
                   "prompt_tokens": getattr(self.client, "prompt_tokens", 0), "completion_tokens": getattr(self.client, "completion_tokens", 0)}
        self._log(f"end: {json.dumps(summary)}")
        self.sandbox.close(); self.transcript.close(); self.events.close()
        return summary
