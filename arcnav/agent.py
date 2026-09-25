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

MODEL_TO_ENGINE = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5", "MOUSE": "ACTION6", "ACTION7": "ACTION7"}
ENGINE_TO_MODEL = {v: k for k, v in MODEL_TO_ENGINE.items()}
MAX_ACTIONS_PER_CALL = 24   # blind 50-action batches walked straight into game over on s5i5
PROBE_SWEEP_AFTER_TURNS = 8   # iter4: after this many model turns on a level without completing it, the harness probes every untried action once
MAX_ACTIONS_PER_TOOL_RUN = 30   # also caps loops of single-action calls inside one python tool run
PROPOSAL_ATTEMPTS = 0   # >0: after a level completion, action() is blocked and propose_solver is forced via tool_choice for this many turns.
                        # Set E (forced, 3 turns): 0.75, level>=2 0/4 — forced solvers burned 50–80 actions per level; disabled by default.


class _T:  # transition view for the host-side NavHelper
    def __init__(self, action, before_frame, after_frame):
        self.action, self.before_frame, self.after_frame = action, before_frame, after_frame


class GameSession:
    def __init__(self, env, game_id: str, client: Optional[ChatClient], *, log_dir: Path, max_minutes: float = 20.0,
                 max_actions: int = 3000, max_model_turns: int = 400, keep_full_turns: int = 3, tool_timeout: int = 30,
                 context_tokens: int = 32768, verbose: bool = True, deadline: Optional[float] = None, think_first_turns: int = 0,
                 tool_choice_required: bool = False, oracle_rules: str = ""):
        self.env, self.game_id, self.client = env, game_id, client
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_minutes, self.max_actions, self.max_model_turns = max_minutes, max_actions, max_model_turns
        self.keep_full_turns, self.tool_timeout, self.context_tokens, self.verbose = keep_full_turns, tool_timeout, context_tokens, verbose
        self.deadline = deadline   # absolute epoch seconds (global run cap), optional
        self.oracle_rules = oracle_rules   # D1 diagnostic: ground-truth rules injected into every turn (never in submissions)
        self.tool_choice_required = tool_choice_required   # force a tool call every turn (models with flaky tool formatting)
        self.think_first_turns = think_first_turns   # iter3: chain-of-thought ON for the first N model turns of every level
        self.level_turn_start = 0
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
        self.checklist: dict = {"goal": "", "roles": {}, "plan": "", "tried": []}   # model-owned fields; harness adds facts
        self.checklist_level = 1
        self._run_actions = 0
        self.game_overs_this_level = 0
        self.rejections_in_row = 0
        self.probe_sweeps: dict[int, str] = {}   # iter4: level -> transition table from the harness probe sweep
        self.level_recaps: list[str] = []   # harness-written summaries of how each completed level was won
        self.recent_codes: list[str] = []   # repetition guard: identical tool code is not executed twice in a row
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
            if act["action"] == "MOUSE" and ("row" not in act or "col" not in act):
                stopped = "MOUSE needs row and col"; break
            if act["action"] == "MOUSE" and not (0 <= int(act["row"]) <= 63 and 0 <= int(act["col"]) <= 63):
                stopped = f"MOUSE coordinates out of range (row={act['row']}, col={act['col']}; valid 0..63)"; break
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
                self.level_turn_start = self.model_turns
                try:
                    self.level_recaps.append(self._level_recap(prev_level))
                except Exception as e:  # never break play on a recap error
                    self._log(f"recap failed: {e!r}")
                self._log(f"*** level {prev_level} completed after {self.level_action_log[-1]} actions (total {self.actions_used})")
                stopped = "level completed" if self.state != "WIN" else "game won"; break
            if self.state == "GAME_OVER":
                game_over = True; self.game_overs_this_level += 1; self._log("*** game over -> reset"); self.reset(); stopped = "game over (reset done, level restarted)"
                self.level_turn_start = self.model_turns   # re-open the thinking window: the level restarts, re-plan with reasoning
                break
        return {"executed_count": executed, "board_changed": changed_any, "level_completed": level_completed, "game_over": game_over,
                "stopped_reason": stopped, "results": results[-12:]}

    # ------------------------------------------------------------ sandbox hooks
    def _state(self) -> dict:
        return {"frame": self.frame.to_payload() if self.frame else None, "valid_actions": self.valid_actions, "level": self.level,
                "levels_total": self.levels_total, "transitions": self.transitions[-400:], "notes": self.notes, "level_recaps": self.level_recaps,
                "checklist": self.checklist}

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
            self._log(f"solver stored ({self.solver['status']}):\n" + code)
            note = ("Solver stored and VERIFIED; the harness will run solve() automatically from the next turn." if self.solver["status"] == "active" else
                    "Solver stored as a DRAFT (no predict() or accuracy < 0.8): the harness will show its suggestion each turn but will not run it by itself.")
            return {**report, "stored": True, "note": note}
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
                g = nav.gauge(); left = g.get("actions_left") if g else None
                for t in [t for t in nav.targets() if t.get("path_len")][:4]:
                    path = nav.path_to(t["row"], t["col"])
                    if path:
                        tag = f"colour {t.get('color')}, {'visited' if t.get('visited') else 'unvisited'}"
                        if left is not None and len(path) >= left:
                            tag += f", TOO LONG for the gauge ({len(path)} >= {left} left)"
                        ready.append(f"  nav.path_to({t['row']}, {t['col']}) -> {path}  # {tag}")
                fr = nav.frontier()
                if fr and (path := nav.path_to(fr[0], fr[1])):
                    ready.append(f"  nav.path_to({fr[0]}, {fr[1]}) -> {path}  # nearest unexplored block")
                if ready:
                    lines.append("Ready-made routes (pass one straight to action(...)):\n" + "\n".join(ready))
                lines += self._gauge_lines(nav, cur)
                lines += self._budget_lines(nav, cur)
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

    def _level_recap(self, level: int) -> str:
        """What won the level: the last attempt's action sequence, objects that vanished (collected), gauge refills."""
        cur = [t for t in self.host_transitions if t.before_frame.level == level]
        # last attempt = transitions after the last reset on this level (a reset restores the initial board)
        first = cur[0].before_frame.ascii if cur else None
        start = 0
        for i, t in enumerate(cur):
            if i > 0 and t.before_frame.ascii == first:
                start = i
        att = cur[start:]
        acts = [t.action if isinstance(t.action, str) else f"MOUSE({t.action['row']},{t.action['col']})" for t in att]
        seq, run = [], None
        for a in acts:  # compress runs: UP x3
            if run and run[0] == a:
                run[1] += 1
            else:
                run = [a, 1]; seq.append(run)
        seq_txt = ", ".join(f"{a} x{n}" if n > 1 else a for a, n in seq)[:400]
        before_objs = {n["hash"]: n for n in att[0].before_frame.segmentation["nodes"]} if att else {}
        after_objs = {n["hash"]: n for n in att[-1].before_frame.segmentation["nodes"]} if att else {}
        gone = [f"colour {n['color']} ({n['pixels']}px at {n['center']})" for h, n in before_objs.items()
                if h not in after_objs and not n["hud"] and n["pixels"] <= 300][:6]
        refills = []
        nav = NavHelper(att, att[-1].after_frame) if att else None
        g = nav.gauge() if nav else None
        if g:
            color = g["color"]
            for t in att[:-1]:   # the last transition enters the next level (its full gauge is not a refill)
                b = sum(v == color for row in t.before_frame.grid for v in row); a = sum(v == color for row in t.after_frame.grid for v in row)
                if a > b:
                    refills.append(f"after {t.action if isinstance(t.action, str) else 'MOUSE'} (+{a - b})")
        parts = [f"Level {level} was completed in {len(att)} actions (attempt {sum(1 for i, t in enumerate(cur) if i > 0 and t.before_frame.ascii == first) + 1}).",
                 f"Winning sequence: {seq_txt}."]
        if gone:
            parts.append("Objects that disappeared during the win (probably collected/used): " + "; ".join(gone) + ".")
        if refills:
            parts.append("Gauge refilled " + ", ".join(refills[:4]) + ".")
        parts.append("The next level normally keeps the same rules with a new layout: find the analogous objects and repeat the strategy with nav.path_to.")
        return " ".join(parts)

    def _probe_sweep(self) -> str:
        """Try every untried action type once (each movement key, SPACE, clicks on up to 6 untried objects) and
        tabulate what each did. Runs at most once per level and only with enough gauge left."""
        cur = [t for t in self.host_transitions if t.after_frame.level == self.level]
        used = {t.action if isinstance(t.action, str) else "MOUSE" for t in cur}
        clicked = {(t.action["row"] // 4, t.action["col"] // 4) for t in cur if isinstance(t.action, dict)}
        plan = [{"action": a} for a in ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "ACTION7") if a in self.valid_actions and a not in used]
        if "MOUSE" in self.valid_actions and self.frame is not None:
            nodes = [n for n in self.frame.segmentation["nodes"] if not n["hud"] and (n["center"][0] // 4, n["center"][1] // 4) not in clicked]
            nodes.sort(key=lambda n: n["pixels"])
            seen_colors = set()
            for n in nodes:  # one click per colour, smallest objects first
                if n["color"] in seen_colors:
                    continue
                seen_colors.add(n["color"]); plan.append({"action": "MOUSE", "row": n["center"][0], "col": n["center"][1], "colour": n["color"]})
                if len(plan) >= 10:
                    break
        try:
            g = NavHelper([t for t in cur if t.before_frame.level == self.level], self.frame).gauge() if self.frame else None
            if g and g.get("actions_left") is not None:
                plan = plan[:max(0, int(g["actions_left"]) - 4)]
        except Exception:
            pass
        if not plan:
            return ""
        rows = []
        for act in plan:
            before = self.frame
            res = self.execute([{k: v for k, v in act.items() if k != "colour"}])
            if not res["executed_count"]:
                break
            after = self.frame
            diff = summarize_diff(before, after)
            label = act["action"] if act["action"] != "MOUSE" else f"MOUSE({act['row']},{act['col']}) on colour {act.get('colour')}"
            what = "no change" if diff["changed_cells"] == 0 else (f"{diff['changed_cells']} cells changed" +
                    (f"; moved: {[(m['color'], m['from'], m['to']) for m in diff.get('moved', [])][:3]}" if diff.get("moved") else "") +
                    (f"; appeared: {[(m['color'], m['center']) for m in diff.get('appeared', [])][:3]}" if diff.get("appeared") else "") +
                    (f"; disappeared: {[(m['color'], m['center']) for m in diff.get('disappeared', [])][:3]}" if diff.get("disappeared") else ""))
            rows.append(f"  {label}: {what}")
            if res["level_completed"] or res["game_over"]:
                rows.append(f"  -> {'LEVEL COMPLETED' if res['level_completed'] else 'GAME OVER (level restarted)'}")
                break
        self._log(f"probe sweep on level {self.level}: {len(rows)} probes")
        return "Harness probe sweep (each untried action once, so you can see what every action does):\n" + "\n".join(rows)

    def _host_probe(self) -> str:
        """After repeated identical calls: execute ONE untried action so the model gets new information."""
        cur = [t for t in self.host_transitions if t.after_frame.level == self.level]
        used = {t.action if isinstance(t.action, str) else "MOUSE" for t in cur}
        clicked = {(t.action["row"] // 4, t.action["col"] // 4) for t in cur if isinstance(t.action, dict)}
        probe = None
        for a in ("SPACE", "ACTION7", "UP", "DOWN", "LEFT", "RIGHT"):
            if a in self.valid_actions and a not in used:
                probe = {"action": a}; break
        if probe is None and "MOUSE" in self.valid_actions and self.frame is not None:
            nodes = [n for n in self.frame.segmentation["nodes"] if not n["hud"] and (n["center"][0] // 4, n["center"][1] // 4) not in clicked]
            nodes.sort(key=lambda n: n["pixels"])
            if nodes:
                r, c = nodes[0]["center"]; probe = {"action": "MOUSE", "row": r, "col": c}
        if probe is None:
            return "Harness probe: nothing untried is left on this level; change the ORDER or the target of your actions."
        before = self.frame
        res = self.execute([probe])
        diff = summarize_diff(before, self.frame) if self.frame else {}
        label = probe["action"] if probe["action"] != "MOUSE" else f"MOUSE(row={probe['row']}, col={probe['col']})"
        self._log(f"host probe: {label} -> changed={res['board_changed']}")
        return (f"Harness probe (because you repeated yourself): executed {label} -> board_changed={res['board_changed']}, "
                f"level_completed={res['level_completed']}, game_over={res['game_over']}; change: {json.dumps(diff, default=str)[:400]}. Build on this.")

    def _checklist_lines(self) -> list[str]:
        """The turn starts from this state instead of from zero: harness facts + the model's own fields, with the next gap named."""
        if self.checklist_level != self.level:   # new level: plan and tried restart; goal/roles carry over (same rules)
            self.checklist["plan"] = ""; self.checklist["tried"] = []; self.checklist_level = self.level
        cur = [t for t in self.host_transitions if t.before_frame.level == t.after_frame.level == self.level]
        used = {t.action if isinstance(t.action, str) else "MOUSE" for t in cur}
        nav = None
        try:
            nav = NavHelper(cur, self.frame) if self.frame else None
        except Exception:
            pass
        items = []
        keys = [a for a in self.valid_actions if a != "MOUSE"]
        untried = [a for a in keys if a not in used]
        moves = nav.moves if nav else {}
        ctrl = ", ".join(f"{k}={moves[k]}" for k in moves) if moves else "no movement learned"
        items.append((not untried and (moves or not keys), "controls", ctrl + (f"; untried keys: {untried}" if untried else "") +
                      ("; MOUSE available" if "MOUSE" in self.valid_actions else "")))
        av = nav.avatar() if nav else None
        if keys:
            items.append((bool(av), "avatar", f"colours {av['colors']} at ({av['row']},{av['col']})" if av else "not identified (press each movement key once)"))
        g = nav.gauge() if nav else None
        lim = (f"gauge colour {g['color']}, ~{g.get('actions_left')} actions left" if g else "no gauge detected yet") + f"; game overs on this level: {self.game_overs_this_level}"
        items.append((True, "limits", lim))
        if self.frame is not None:
            seg = [n for n in self.frame.segmentation["nodes"] if not n["hud"]]
            counts: dict[int, int] = {}
            for n in seg:
                counts[n["color"]] = counts.get(n["color"], 0) + 1
            roles = self.checklist.get("roles") or {}
            desc = ", ".join(f"colour {c} x{k}" + (f" = {roles.get(str(c)) or roles.get(c)}" if (roles.get(str(c)) or roles.get(c)) else " (role ?)") for c, k in sorted(counts.items(), key=lambda x: -x[1])[:8])
            known = sum(1 for c in counts if roles.get(str(c)) or roles.get(c))
            items.append((known >= min(2, len(counts)), "objects/roles", desc + "   <- fill checklist['roles'][colour] = 'role'"))
        goal = self.checklist.get("goal") or ""
        items.append((bool(goal), "goal", goal or "(none)   <- fill checklist['goal'] = 'hypothesis' after a probe"))
        tried = self.checklist.get("tried") or []
        items.append((True, "tried", "; ".join(tried[-5:]) if tried else "(nothing recorded)   <- checklist['tried'].append('what + outcome')"))
        plan = self.checklist.get("plan") or ""
        items.append((bool(plan), "plan", plan or "(none)   <- fill checklist['plan'] = 'next concrete step'"))
        done = sum(1 for ok, _, _ in items if ok); first_gap = next((name for ok, name, _ in items if not ok), None)
        head = f"CHECKLIST (level {self.level}) — {done}/{len(items)} settled. " + (f"Resolve first: {first_gap.upper()}." if first_gap else "All settled: execute the plan.")
        return [head] + [f"[{'x' if ok else ' '}] {name}: {text}" for ok, name, text in items]

    def _new_kinds_lines(self) -> list[str]:
        """Object kinds (colour, size class) present on this level that never appeared on earlier levels: probe these first."""
        if self.level <= 1 or self.frame is None:
            return []
        def kinds(frame):
            out = set()
            for n in frame.segmentation["nodes"]:
                if n["hud"]:
                    continue
                size = "tiny" if n["pixels"] <= 4 else "small" if n["pixels"] <= 30 else "medium" if n["pixels"] <= 200 else "large"
                out.add((n["color"], size))
            return out
        seen = set()
        for t in self.host_transitions:
            if t.before_frame.level < self.level:
                seen |= kinds(t.before_frame)
        now = kinds(self.frame)
        new = sorted(now - seen)
        if not new:
            return []
        ex = {}
        for n in self.frame.segmentation["nodes"]:
            size = "tiny" if n["pixels"] <= 4 else "small" if n["pixels"] <= 30 else "medium" if n["pixels"] <= 200 else "large"
            if (n["color"], size) in new and (n["color"], size) not in ex:
                ex[(n["color"], size)] = n["center"]
        items = ", ".join(f"colour {c} ({sz}, e.g. at {ex.get((c, sz))})" for c, sz in new[:6])
        return [f"NEW ON THIS LEVEL (not seen on earlier levels): {items}. A new kind usually carries the new rule of this level: "
                "touch/click each one once early and record what it did."]

    def _budget_lines(self, nav, cur) -> list[str]:
        """Arithmetic the model tends to skip: moves left on the gauge vs. route lengths, and whether refills are known."""
        g = nav.gauge() if nav else None
        if not g or g.get("actions_left") is None:
            return []
        per = abs(float(g.get("per_action") or 1)) or 1
        full = int(round(g["size"] / per)) if g.get("size") else None
        lines = [f"BUDGET CHECK: ~{g['actions_left']} moves left on this gauge" + (f" (a full gauge is ~{full} moves at {per:g} per move)" if full else "") + "."]
        try:
            reach = [t for t in nav.targets() if t.get("path_len")]
            if reach:
                far = max(t["path_len"] for t in reach); near = min(t["path_len"] for t in reach)
                lines.append(f"Known targets are {near}-{far} moves away. Anything beyond {g['actions_left']} moves is unreachable before the gauge runs out "
                             "unless you pass a refill item on the way (objects that increased the gauge when touched, see gauge lines).")
        except Exception:
            pass
        return lines

    def _micro_diff_lines(self, before_n: int) -> list[str]:
        """When the last actions changed only a few cells (a sprite rotated/recoloured), zoom into that region."""
        new = self.host_transitions[before_n:]
        if not new:
            return []
        b, a = new[0].before_frame, new[-1].after_frame
        mb, ma = masked_ascii(b).splitlines(), masked_ascii(a).splitlines()
        cells = [(r, c) for r in range(len(mb)) for c in range(len(mb[r])) if mb[r][c] != ma[r][c]]
        if not cells or len(cells) > 40:
            return []
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1 = max(0, min(rs) - 2), min(63, max(rs) + 2); c0, c1 = max(0, min(cs) - 2), min(63, max(cs) + 2)
        if (r1 - r0) > 14 or (c1 - c0) > 14:
            return []
        crop = lambda rows: "\n".join("    " + rows[r][c0:c1 + 1] for r in range(r0, r1 + 1))
        return [f"SMALL CHANGE ZOOM (rows {r0}-{r1}, cols {c0}-{c1}; {len(cells)} cells changed, HUD excluded) — before / after:",
                crop(b.ascii.splitlines()), "    ->", crop(a.ascii.splitlines())]

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
        if self.oracle_rules:
            parts.append("KNOWN RULES OF THIS GAME (given, trust them fully):\n" + self.oracle_rules)
        try:
            parts += self._checklist_lines()
            parts += self._new_kinds_lines()
        except Exception as e:
            parts.append(f"(checklist unavailable: {type(e).__name__})")
        if self.last_outcome:
            parts += self.last_outcome
        parts += self._nav_lines()
        if (PROBE_SWEEP_AFTER_TURNS and self.level not in self.probe_sweeps and self.model_turns - self.level_turn_start >= PROBE_SWEEP_AFTER_TURNS
                and not (self.solver and self.solver.get("status") == "active")):
            lvl = self.level
            text = self._probe_sweep()
            self.probe_sweeps[lvl] = text if self.level == lvl else ""   # the sweep itself completed the level: nothing to show on the new level
            if self.level != lvl:
                self.probe_sweeps.setdefault(self.level, "")
                self.level_turn_start = self.model_turns
        if self.probe_sweeps.get(self.level):
            parts.append(self.probe_sweeps[self.level])
        if self.level_recaps:
            parts.append("Recap of previous levels (written by the harness):\n" + "\n".join(f"- {r}" for r in self.level_recaps[-2:]))
        parts.append("Your notes (the sandbox variable `notes`; keep it current instead of re-deriving the rules each turn):\n" +
                     (self.notes or "(empty — write what each key does, the goal hypothesis and the next plan)"))
        if self.proposal_required > 0:
            parts.append(f"REQUIRED: you just completed a level, so you know the rules. Call propose_solver(code) with a solve() that reproduces "
                         f"the winning strategy from `transitions` of the previous level (use nav.path_to / segmentation, not fixed coordinates). "
                         f"action() stays blocked until you do ({self.proposal_required} turn(s) left before the requirement is waived); "
                         "inspecting the board is allowed.")
        if self.solver and self.solver.get("status") == "draft":
            try:
                res = self.sandbox.run(f"__solver_code = {json.dumps(self.solver['code'])}\nexec(compile(__solver_code, 'solver.py', 'exec'), globals())\nprint(list(solve() or [])[:12])", timeout=15)
                self.solver["suggestion"] = (res["stdout"].strip() or res.get("error") or "?")[:300]
            except Exception as e:
                self.solver["suggestion"] = f"(dry run failed: {e!r})"
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
        if isinstance(res.get("checklist"), dict) and res["checklist"]:
            ck = res["checklist"]
            self.checklist = {"goal": str(ck.get("goal") or "")[:300], "roles": ck.get("roles") if isinstance(ck.get("roles"), dict) else {},
                              "plan": str(ck.get("plan") or "")[:300], "tried": [str(t)[:120] for t in (ck.get("tried") or [])][-8:]}
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
        choice = {"type": "function", "function": {"name": "propose_solver"}} if 0 < self.proposal_required < PROPOSAL_ATTEMPTS else ("required" if self.tool_choice_required else "auto")
        override = None
        if self.think_first_turns and (self.model_turns - self.level_turn_start) < self.think_first_turns:
            override = {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": max(self.client.max_tokens, 8192), "temperature": 0.6, "top_p": 0.95}
        r = None
        for attempt in range(6):
            try:
                r = self.client.chat(self.messages, tools=tools, tool_choice=choice, override=override); break
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
            # content-as-code fallback: some models write the python call in the message body instead of a tool call
            body = (msg.get("content") or "").strip()
            if body.startswith("```"):
                body = body.strip("`"); body = body.split("\n", 1)[1] if "\n" in body else body
                body = body.rsplit("```", 1)[0] if "```" in body else body
            # JSON fallback: the tool arguments emitted as a bare JSON object in the body (seen with gpt-oss + tool_choice=required)
            if body.startswith("{") and body.endswith("}"):
                try:
                    obj = json.loads(body)
                    if isinstance(obj, dict) and isinstance(obj.get("code"), str) and obj["code"].strip():
                        calls = [{"id": "fallback_json", "type": "function", "function": {"name": "python", "arguments": json.dumps(obj)}}]
                        self.messages[-1]["tool_calls"] = calls; self.messages[-1]["content"] = ""
                        self._log("json-content fallback: executing the body's `code` field as the python tool")
                except Exception:
                    pass
            looks_like_code = not calls and any(k in body for k in ("action(", "propose_solver(", "print(", "nav.", "checklist[")) and not body.startswith("{")
            if looks_like_code and len(body) < 6000:
                try:
                    import ast as _ast; _ast.parse(body)
                    calls = [{"id": "fallback_0", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": body})}}]
                    self.messages[-1]["tool_calls"] = calls; self.messages[-1]["content"] = ""
                    self._log("content-as-code fallback: executing the message body as python")
                except SyntaxError:
                    pass
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
                args, code = {}, ""
            for key in ("goal", "plan"):   # checklist fields carried by the tool call itself
                if isinstance(args.get(key), str) and args[key].strip():
                    self.checklist[key] = args[key].strip()[:300]
            if isinstance(args.get("roles"), str) and args["roles"].strip():
                for part in args["roles"].replace(";", ",").split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if k.strip().isdigit() and v.strip():
                            self.checklist.setdefault("roles", {})[k.strip()] = v.strip()[:40]
            if call["function"].get("name") == "propose_solver" and code:
                code = f"print(propose_solver({json.dumps(code)}))"   # routed through the sandbox's verifier
            norm = " ".join(code.split())
            if code and norm in self.recent_codes[-2:]:
                used = {t.action if isinstance(t.action, str) else "MOUSE" for t in self.host_transitions if t.after_frame.level == self.level}
                unused = [a for a in self.valid_actions if a not in used]
                out = ("Rejected: this code is identical to one of your last two calls, so it was NOT executed (repeating it cannot give new information). "
                       "Do something different: " + (f"actions not yet tried on this level: {unused}. " if unused else "") +
                       "Try a different key, a different object, SPACE/MOUSE on things you have not touched, or update `notes` with a new hypothesis first.")
                self._log("repetition guard: identical tool code rejected")
                self.rejections_in_row += 1
                if self.rejections_in_row >= 2:
                    out += "\n" + self._host_probe()
                    self.rejections_in_row = 0
            else:
                self.rejections_in_row = 0
                out = self._run_tool(code, who="model") if code else "Error: tool call had no `code` argument"
            if code:
                self.recent_codes = (self.recent_codes + [norm])[-4:]
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
        try:
            self.last_outcome += self._micro_diff_lines(before_n)
        except Exception:
            pass
        # harness-written 'tried' entry: what this turn did and what happened
        new = self.host_transitions[before_n:]
        acts = [t.action if isinstance(t.action, str) else "MOUSE" for t in new]
        if acts:
            compact = []
            for a in acts:
                if compact and compact[-1][0] == a:
                    compact[-1][1] += 1
                else:
                    compact.append([a, 1])
            what = " ".join(f"{a}x{n}" if n > 1 else a for a, n in compact)[:80]
            changed = sum(1 for t in new if masked_ascii(t.before_frame) != masked_ascii(t.after_frame))
            outcome = ("LEVEL DONE" if any(t.before_frame.level != t.after_frame.level for t in new) else
                       f"{changed}/{len(new)} moves changed the board")
            self.checklist.setdefault("tried", []).append(f"T{self.model_turns}: {what} -> {outcome}")
            self.checklist["tried"] = self.checklist["tried"][-8:]
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
