"""The rulebook loop.

  level 1 start ─ INIT: the model (thinking on) writes the rulebook from the first board.
  every decision ─ the program lists candidate actions and PREDICTS each deterministically from the record;
                   the model (thinking off) picks one; the program executes it one action at a time, checking
                   every action against its prediction.
  mismatch / level end / game over ─ the program re-induces its facts, scores the rulebook, then the model
                   (thinking on) REVIEWS the rulebook (refute / add / edit) and the plan.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from arcnav.frame import masked_ascii

from .book import Rulebook
from .candidates import Candidate, build, resolve, state_key
from .env import Game, action_label
from .llm_io import Model
from .predict import Evidence
from . import goals2
from .entities import build_scene
from .plans import make_plans
from .repl import Repl

DEFAULTS = {"max_minutes": 20.0, "level_actions": 200, "max_actions": 2000, "reviews_per_level": 8, "max_levels": 10,
            "review_think": "level",   # thinking during reviews: "level" = only after a level completion, "always", "never"
            "init_think": True,        # thinking for the initial rulebook (off on Kaggle: the FP8 27B thinks past the token budget)
            "mode": "choose"}          # "choose": the model picks a candidate label; "coder": the model writes python against the library


class RulebookAgent:
    def __init__(self, game: Game, model: Optional[Model], *, log_dir: Path, cfg: Optional[dict] = None, deadline: Optional[float] = None):
        self.g, self.model = game, model
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.book = Rulebook()
        self.tried: set = set()
        self.outcomes: list[str] = []
        self.mismatches = self.reviews = self.observations = 0
        self.level_stats: list[dict] = []
        self.deadline = deadline
        self.t0 = time.time()
        self._log_f = open(self.log_dir / f"{game.game_id}.log", "a")
        game.record_path = self.log_dir / f"{game.game_id}.actions.jsonl"
        self.last_choice: list[str] = []
        self.last_cell: Optional[tuple] = None   # cell of the last executed click: the fallback never clicks it again right away (toggle undo)
        self.last_verdict = None; self.last_essential: list = []; self.pending_mismatch = None
        self.custom_rules: list = []      # (name, fn, accuracy, n) model-written transition rules verified by replay
        self.idle_turns = 0
        from collections import Counter as _C
        self.metrics = _C()               # counters for the evaluation report (prediction kinds, verdicts, plans, fallbacks ...)
        self.attempt_start_actions = 0    # actions_used when the current attempt (level or reset) began
        self.plan_ignored = 0             # consecutive decisions in which the model ignored an offered plan
        self.pending_then: list = []      # follow-up labels from the last decision's "then" list
        self.plan_fail: dict = {}         # (level, plan label) -> executions that ended in a mismatch / game over
        self.refused: dict = {}           # (level, label) -> times the model asked for a label that is not a candidate
        self.game_over_at: dict = {}      # level -> [actions into the attempt at which a game over happened]
        from collections import Counter as _C
        self.metrics = _C()               # counters for the evaluation report (prediction kinds, verdicts, plans, fallbacks ...)
        self.attempt_start_actions = 0    # actions_used when the current attempt (level or reset) began
        self.plan_ignored = 0             # consecutive decisions in which the model ignored an offered plan
        self.pending_then: list = []      # follow-up labels from the last decision's "then" list
        self.plan_fail: dict = {}         # (level, plan label) -> executions that ended in a mismatch / game over
        self.refused: dict = {}           # (level, label) -> times the model asked for a label that is not a candidate
        self.game_over_at: dict = {}      # level -> [actions into the attempt at which a game over happened]
        self.repl = None
        self.distrust: dict = {}          # (level, click colour) -> failed board predictions; the predictor stops predicting after 2
        self.preds: dict = {}             # rulebook entry id -> goals2.Predicate (win predicates learned from completed levels)
        self.level_start_frame = None     # first board of the current attempt (goal inference compares against it)

    # ── logging ───────────────────────────────────────────────────────────
    def log(self, text: str) -> None:
        self._log_f.write(f"[{time.time() - self.t0:6.0f}s a{self.g.actions_used:4d} L{self.g.level}] {text}\n"); self._log_f.flush()

    def save_book(self) -> None:
        self.book.save(self.log_dir / f"{self.g.game_id}.rulebook.json")

    def _out_of_time(self) -> bool:
        if self.deadline and time.time() > self.deadline:
            return True
        return time.time() - self.t0 > self.cfg["max_minutes"] * 60 or self.g.actions_used >= self.cfg["max_actions"]

    # ── evidence ─────────────────────────────────────────────────────────
    def evidence(self, full: bool = False) -> Evidence:
        lv = self.g.level
        return Evidence(self.g, distrust={key: n for (l, key), n in self.distrust.items() if l == lv}, custom_rules=self.custom_rules)

    def sync(self, ev: Evidence) -> list[str]:
        changes = self.book.sync(ev.facts(), level=self.g.level)
        if changes:
            self.log("rulebook <- evidence: " + " | ".join(changes)); self.save_book()
        return changes

    # ── win predicates ───────────────────────────────────────────────────
    def live_preds(self) -> list:
        # win entries of known kinds written by the model (or by the old goal inference) get predicate objects too
        for e in self.book.section("win"):
            if e.id not in self.preds and e.status != "refuted":
                p = goals2.from_entry(e)
                if p is not None:
                    self.preds[e.id] = p
        return [(eid, p) for eid, p in self.preds.items() if (e := self.book.get(eid)) is not None and e.status != "refuted"]

    def goal_state(self, ev: Evidence) -> dict:
        """Evaluate every live win predicate on the current board: progress lines for the model, and whether a submit is due."""
        sc = ev.scene; lines = []; held = []; missing = []; submit = None; has_submit = False
        for eid, p in self.live_preds():
            if isinstance(p, goals2.Pressed):
                has_submit = True; b = p.button(sc)
                if b is not None:
                    submit = b
                lines.append(f"{eid} [submit] {p.evaluate(sc, ev)[1]}"); continue
            ok, txt, _ = p.evaluate(sc, ev)
            (held if ok else missing).append(eid)
            lines.append(f"{eid} [{p.kind}] {'HOLDS' if ok else 'not yet'}: {txt}")
        conds = [eid for eid, p in self.live_preds() if not isinstance(p, goals2.Pressed)]
        return {"lines": lines, "held": held, "missing": missing, "all_hold": bool(conds) and not missing, "submit": submit if has_submit else None,
                "has_submit": has_submit, "conds": conds}

    def refute_preds(self, ids: list, note: str) -> list:
        out = []
        for eid in ids:
            r = self.book.refute(eid, note)
            if r:
                out.append(r)
        if out:
            self.log("win predicates refuted: " + " | ".join(out)); self.save_book()
        return out

    # ── model steps ──────────────────────────────────────────────────────
    def init_book(self) -> None:
        if self.model is None:
            self.book.plan = "no model: try untested actions, then act on harness-induced rules"; return
        obj = self.model.init_rulebook(self.g, think=bool(self.cfg.get("init_think", True)))
        if not obj:
            self.log("INIT failed; starting with an empty rulebook"); return
        for sec in ("env", "rules", "win"):
            for e in (obj.get(sec) or [])[:10]:
                if isinstance(e, dict) and e.get("text"):
                    self.book.add(sec, str(e["text"]), kind=str(e.get("kind") or "other"), params=e.get("params") if isinstance(e.get("params"), dict) else {}, level=self.g.level)
                elif isinstance(e, str):
                    self.book.add(sec, e, level=self.g.level)
        self.book.plan = str(obj.get("plan", ""))[:600]
        self.log("INIT rulebook:\n" + self.book.render()); self.save_book()

    def review(self, event: str, changes: list[str], *, before=None, bbox=None, level_event: bool = False, extra: str = "") -> None:
        if self.model is None:
            return
        self.reviews += 1
        mode = self.cfg.get("review_think", "level")
        think = mode == "always" or (mode == "level" and level_event)
        obj = self.model.review(self.g, self.book, event, changes, self.outcomes, before=before, bbox=bbox, think=think, extra=extra)
        if not obj:
            return
        done = self.book.apply_edits(obj.get("edits") or [], level=self.g.level)
        self.apply_roles(obj.get("roles"))
        if obj.get("procedure"):
            proc = str(obj["procedure"])[:500]
            old = next((e for e in self.book.section("win") if e.text.startswith("PROCEDURE:")), None)
            if old is not None:
                old.text = "PROCEDURE: " + proc; old.level = self.g.level
            else:
                self.book.add("win", "PROCEDURE: " + proc, level=self.g.level)
        if obj.get("plan"):
            self.book.plan = str(obj["plan"])[:600]
        self.log(f"REVIEW ({event[:60]}): {done}\n" + self.book.render()); self.save_book()

    def apply_roles(self, roles) -> None:
        if not isinstance(roles, dict):
            return
        for key, role in list(roles.items())[:12]:
            key, role = str(key)[:40], str(role)[:60]
            text = f"role of {key}: {role}"
            old = next((e for e in self.book.section("env") if e.text.startswith(f"role of {key}:")), None)
            if old is not None:
                old.text = text; old.level = self.g.level
            else:
                self.book.add("env", text, level=self.g.level)
        self.save_book()

    def _same_cell(self, c: Candidate) -> bool:
        a = c.actions[0] if c.actions else None
        return isinstance(a, dict) and self.last_cell is not None and (a["row"] // 3, a["col"] // 3) == self.last_cell

    def choose(self, cands: list[Candidate], budget_text: str, ev: Evidence, goal: Optional[dict] = None) -> Candidate:
        cands = [c for c in cands if c.kind != "info"] or cands
        fallback = next((c for c in cands if not self._same_cell(c)), cands[0])   # untested first, then by priority; never undo the last toggle
        if self.model is None or not cands:
            return fallback
        extra = ("WIN CONDITION PROGRESS (program-evaluated on the current board):\n  " + "\n  ".join(goal["lines"])) if goal and goal.get("lines") else ""
        bad = [lab for (lv, lab), n in self.refused.items() if lv == self.g.level and n >= 2]
        if bad:
            extra = (f"NOTE: there is NO candidate named {', '.join(repr(b) for b in bad[:3])} in this game (asked {sum(self.refused[(self.g.level, b)] for b in bad[:3])}x, refused). "
                     "Choose an exact label from CANDIDATES.\n") + extra
        obj = self.model.decide(self.g, self.book, cands, self.outcomes, budget_text, extra=extra)
        if not obj:
            return fallback
        done = self.book.apply_edits(obj.get("edits") or [], level=self.g.level)
        self.apply_roles(obj.get("roles"))
        if done:
            self.log(f"DECIDE edits: {done}"); self.save_book()
        label = str(obj.get("choice", "")).strip()
        c = resolve(label, cands, self.g, ev)
        if c is not None and c.label != label:
            self.log(f"DECIDE: label {label!r} resolved to {c.label}")
        if c is None:
            self.metrics["fallback"] += 1
            self.refused[(self.g.level, label)] = self.refused.get((self.g.level, label), 0) + 1
            self.log(f"DECIDE: label {label!r} not a candidate -> fallback {fallback.label}")
            self.outcomes.append(f"(program refused {label}: not a candidate — hidden objects did nothing twice on this level; ran {fallback.label} instead)")
            return fallback
        # wandering guards: a candidate already tried in this world state that is predicted to change nothing (or only a counter)
        # is a wasted action; so is the same label three times in a row
        self.last_choice.append(c.label)
        repeat = len(self.last_choice) >= 3 and len(set(self.last_choice[-3:])) == 1
        recent = self.last_choice[-3:-1].count(c.label) >= 1
        no_effect = c.pred_kind in ("noop", "hud", "blocked")
        rl = getattr(self, "recent_labels", [])
        monotone = len(rl) >= 12 and rl.count(c.label) >= 8 and c.kind != "submit"
        if monotone:
            alt = next((x for x in cands if not x.tested and x.label != c.label and x.pred_kind not in ("noop", "hud", "blocked") and not self._same_cell(x)), None)
            if alt:
                self.log(f"DECIDE: {c.label} chosen {rl.count(c.label)} of the last 12 decisions without finishing the level -> {alt.label}")
                self.outcomes.append(f"(program vetoed {c.label}: {rl.count(c.label)}/12 recent decisions, no level progress; ran {alt.label} instead)")
                self.metrics["veto_monotone"] += 1
                return alt
        # veto only wasted repeats: a repeat that keeps changing the world (pressing a conveyor button again) is legitimate
        if (c.tested and (no_effect or repeat)) or (recent and no_effect):
            alt = next((x for x in cands if not x.tested and x.label != c.label and x.pred_kind not in ("noop", "hud", "blocked") and not self._same_cell(x)), None) or \
                next((x for x in cands if not x.tested and x.label != c.label and not self._same_cell(x)), None)
            if alt:
                self.log(f"DECIDE: {c.label} already tried here and predicted '{c.pred_kind}'{' (3x in a row)' if repeat else ''} -> {alt.label}")
                self.outcomes.append(f"(program vetoed {c.label}: already tried here, no world change expected; ran {alt.label} instead)")
                return alt
        self.log(f"DECIDE: {c.label} (expect: {str(obj.get('expect', ''))[:120]})")
        self.metrics["chosen_" + ("plan" if c.kind == "plan" else c.kind)] += 1
        then = obj.get("then")
        self.pending_then = [str(x) for x in then][:3] if isinstance(then, list) else []
        return c

    # ── main loop ────────────────────────────────────────────────────────
    def play(self) -> dict:
        g = self.g
        g.reset()
        self.log(f"start: {g.levels_total} levels, valid {g.valid_actions}")
        self.init_book()
        stop = "done"
        while g.state != "WIN" and g.level <= self.cfg["max_levels"]:
            r = self.play_level()
            self.level_stats.append(r)
            self.log(f"level {r['level']}: {'COMPLETED' if r['completed'] else 'not completed'} in {r['actions']} actions ({r['reason']})")
            if not r["completed"]:
                stop = r["reason"]; break
        res = {"game_id": g.game_id, "levels_completed": g.level - 1 if g.state != "WIN" else g.levels_total, "levels_total": g.levels_total,
               "actions": g.actions_used, "level_action_log": g.level_action_log, "stop_reason": stop, "seconds": round(time.time() - self.t0, 1),
               "mismatches": self.mismatches, "reviews": self.reviews, "observations": self.observations, "rulebook": self.book.stats(),
               "rulebook_version": self.book.version, "levels": self.level_stats,
               "model_calls": self.model.calls if self.model else {}, "model_seconds": round(self.model.seconds, 1) if self.model else 0,
               "metrics": dict(self.metrics)}
        self.log("result: " + json.dumps(res)); self.save_book(); self._log_f.close()
        return res

    def play_level(self) -> dict:
        g = self.g; level0 = g.level; a0 = g.actions_used; reviews0 = self.reviews; decisions = 0
        self.level_start_frame = g.frame; self.attempt_start_actions = g.actions_used
        while g.level == level0 and g.state != "WIN":
            if self._out_of_time():
                return self._lv(level0, a0, decisions, "time/actions budget")
            if g.actions_used - a0 >= self.cfg["level_actions"]:
                return self._lv(level0, a0, decisions, "level action cap")
            ev = self.evidence(); self.sync(ev)
            goal = self.goal_state(ev)
            cands = build(g, ev, self.tried, goal=goal)
            plans = make_plans(g, ev, self.live_preds())
            kept = []
            for c in plans:
                nf = self.plan_fail.get((g.level, c.label), 0)
                if c.kind == "plan" and nf >= 3:   # three failures on this level: the plan is wrong for this board, stop offering it
                    c = Candidate(c.label, "info", [], f"withdrawn after {nf} failed runs on this level: {c.prediction[:120]}", priority=999)
                elif c.kind == "plan" and nf >= 2:
                    c.priority = 40; c.prediction = f"[failed {nf}x on this level] " + c.prediction
                kept.append(c)
            plans = kept
            self.metrics["plans_offered"] += sum(1 for c in plans if c.kind == "plan")
            for c in plans:
                if c.kind == "plan" and c.actions:
                    k = ev.predict(c.actions[0]).kind
                    if k in ("noop", "hud", "blocked"):
                        c.pred_kind = k; c.prediction = f"[first step predicted {k}] " + c.prediction
                self.log(f"{'plan candidate' if c.kind == 'plan' else 'plan info'}: {c.label} -> {c.prediction[:160]}")
            cands = [c for c in plans if c.kind == "plan"] + cands + [c for c in plans if c.kind != "plan"]
            if not [c for c in cands if c.kind != "info"]:
                return self._lv(level0, a0, decisions, "no candidates")
            budget = f"actions used on this level {g.actions_used - a0} (cap {self.cfg['level_actions']}), total {g.actions_used}"
            lim = self.attempt_limit()
            if lim:
                budget += f"; THIS ATTEMPT ENDS IN A GAME OVER AFTER ~{lim} ACTIONS (seen {len(self.game_over_at.get(g.level, []))}x): {lim - (g.actions_used - self.attempt_start_actions)} left"
            if goal["lines"]:
                self.log("goal state: " + " | ".join(goal["lines"]))
            if self.cfg.get("mode") == "coder" and self.model is not None:
                decisions += 1
                res = self.code_turn(ev, cands, plans, goal, budget, level0, reviews0, a0)
                if res == "won":
                    return self._lv(level0, a0, decisions, "completed")
                continue
            cand = self.choose(cands, budget, ev, goal); decisions += 1
            sk = state_key(g, ev)
            live_plans = [c for c in plans if c.kind == "plan" and self.plan_fail.get((g.level, c.label), 0) < 2 and c.pred_kind not in ("noop", "hud", "blocked")
                          and (sk, c.label) not in self.tried]
            if live_plans and cand.kind != "plan":
                self.plan_ignored += 1
                if self.plan_ignored >= 3:   # the model keeps ignoring a program-made plan for a live win condition: the program runs it
                    cand = live_plans[0]; self.plan_ignored = 0; self.metrics["plan_autorun"] += 1
                    self.log(f"DECIDE: plan ignored 3x -> program runs {cand.label}")
                    self.outcomes.append(f"(program ran {cand.label} because the plan was ignored three times)")
            elif cand.kind == "plan":
                self.plan_ignored = 0
            self.tried.add((state_key(g, ev), cand.label))
            progress0 = self._progress_score(ev) if cand.kind == "plan" else None
            res = self.execute(cand, level0, reviews0)
            # batched follow-ups the model listed in "then": run while nothing surprising happens (labels re-resolved on the fresh board)
            for lab in list(getattr(self, "pending_then", []) or []):
                if res != "ok" or g.level != level0 or self._out_of_time():
                    break
                ev2 = self.evidence(); cands2 = build(g, ev2, self.tried, goal=self.goal_state(ev2))
                c2 = resolve(lab, cands2, g, ev2)
                if c2 is None or c2.kind in ("info", "submit") or c2.pred_kind in ("noop", "hud", "blocked") or (state_key(g, ev2), c2.label) in self.tried:
                    self.log(f"THEN: {lab!r} skipped"); continue
                self.log(f"THEN: {c2.label}"); self.metrics["then_run"] += 1
                self.tried.add((state_key(g, ev2), c2.label))
                res = self.execute(c2, level0, reviews0)
            self.pending_then = []
            if cand.kind == "plan" and res not in ("won", "level"):
                progress1 = self._progress_score(self.evidence())
                if res in ("mismatch", "over") or progress1 <= progress0:   # a plan that ran without moving any win condition forward failed
                    self.plan_fail[(g.level, cand.label)] = self.plan_fail.get((g.level, cand.label), 0) + 1
                    self.log(f"plan {cand.label}: no progress ({progress0:.2f} -> {progress1:.2f}), failures {self.plan_fail[(g.level, cand.label)]}")
            # monotony guard: one label for most of the last 12 decisions without progress -> next decision must differ
            self.recent_labels = (getattr(self, "recent_labels", []) + [cand.label])[-12:]
            if res == "won":
                return self._lv(level0, a0, decisions, "completed")
        return self._lv(level0, a0, decisions, "completed" if g.level > level0 else "left")

    def code_turn(self, ev, cands, plans, goal, budget: str, level0: int, reviews0: int, a0: int) -> str:
        """Coder mode: one model call -> python code -> run against the library with verified actions."""
        g = self.g
        if self.repl is None:
            self.repl = Repl(self)
        nudge = ""
        if g.level_action_log and g.actions_used - a0 > 2 * max(g.level_action_log[-1], 15):
            nudge = (f"You have spent {g.actions_used - a0} actions on this level (the previous level took {g.level_action_log[-1]}). "
                     "Stop exploring: replay the procedure that won the previous level adapted to this board, or test one new hypothesis with a verified rule.")
        ns = self.repl.namespace(ev, cands, plans, goal, level0, reviews0)
        code, text = self.model.code_turn(g, self.book, cands, plans, goal, self.outcomes, budget, self.repl.notes, self.repl_outputs if hasattr(self, "repl_outputs") else [], nudge)
        if not hasattr(self, "repl_outputs"):
            self.repl_outputs = []
        if not code:
            fb = next((c for c in cands if c.kind not in ("info",) and not self._same_cell(c)), None)
            self.log(f"CODE: no code returned ({text[:100]!r}) -> fallback {fb.label if fb else 'none'}")
            if fb is None:
                return "ok"
            self.tried.add((state_key(g, ev), fb.label))
            return self.execute(fb, level0, reviews0)
        self.log("CODE:\n" + code[:1500])
        self.pending_mismatch = None
        out = self.repl.run(code, ns)
        self.log("CODE OUTPUT:\n" + out[:1500])
        # runaway guard: turns that execute nothing cost model time but no game time; after two in a row the program acts
        self.idle_turns = (self.idle_turns + 1) if self.repl.turn_actions == 0 else 0
        if self.idle_turns >= 2:
            fb = next((c for c in cands if c.kind != "info" and not self._same_cell(c)), None)
            if fb is not None:
                self.log(f"CODE: {self.idle_turns} calls without an action -> program runs {fb.label}")
                self.repl_outputs.append(f"--- program note: your last {self.idle_turns} calls executed no action, so the program ran {fb.label} itself ---")
                self.idle_turns = 0
                self.tried.add((state_key(g, ev), fb.label))
                return self.execute(fb, level0, reviews0)
        self.repl_outputs = (self.repl_outputs + [f"--- call (actions {self.repl.turn_actions}) ---\n{code[:700]}\n>>> output:\n{out[:900]}"])[-3:]
        if self.pending_mismatch is not None and self.reviews - reviews0 < self.cfg["reviews_per_level"]:
            act, label, ptxt, vtxt, before, bbox, changes = self.pending_mismatch
            self.review(f"MISMATCH on {action_label(act)} (inside model code). Predicted: {ptxt}. Observed: {vtxt}. Which rulebook entry was wrong, and what rule explains the observation?",
                        changes, before=before, bbox=bbox)
            self.pending_mismatch = None
        if self.repl.turn_status in ("won", "level", "over"):
            return self.repl.turn_status
        return "ok"

    def _progress_score(self, ev) -> float:
        """Sum of the live win predicates' progress numbers (0..1 each) on the current board."""
        sc = ev.scene; tot = 0.0
        for eid, p_ in self.live_preds():
            try:
                tot += float(p_.evaluate(sc, ev)[2])
            except Exception:
                pass
        return tot

    def attempt_limit(self) -> Optional[int]:
        """If game overs on this level happened at (nearly) the same action count, that count is the attempt's action limit."""
        xs = self.game_over_at.get(self.g.level, [])
        if len(xs) >= 2 and max(xs) - min(xs) <= 3:
            return round(sum(xs) / len(xs))
        return None

    def _lv(self, level0, a0, decisions, reason) -> dict:
        return {"level": level0, "completed": self.g.level > level0 or self.g.state == "WIN", "actions": self.g.actions_used - a0, "decisions": decisions, "reason": reason}

    def execute(self, cand: Candidate, level0: int, reviews0: int) -> str:
        """Run the candidate's actions one by one, each checked against its prediction. Returns 'won' | 'level' | 'over' | 'mismatch' | 'ok'."""
        for i, act in enumerate(cand.actions):
            st = self._step(act, cand.label, level0, reviews0, kind=cand.kind)
            if st in ("won", "level", "over", "mismatch", "invalid"):
                return "ok" if st == "invalid" else st
            if cand.kind == "key-run" and not self.g.transitions[-1].before_frame.ascii != self.g.transitions[-1].after_frame.ascii:
                break
        return "ok"

    def _step(self, act, label: str, level0: int, reviews0: int, *, kind: str = "", review_mismatch: bool = True) -> str:
        """One verified action: predict, execute, judge, update the book and win predicates, review on surprises.
        Returns 'won' | 'level' | 'over' | 'mismatch' | 'ok' | 'invalid'."""
        g = self.g
        ev = self.evidence()
        pred = ev.predict(act)
        before = g.frame
        res = g.step(act)
        if res.get("invalid"):
            self.outcomes.append(f"{label}: {action_label(act)} invalid now"); return "invalid"
        t = g.transitions[-1]
        if isinstance(act, dict):   # only a click that changed the clicked cell itself (a toggle) is protected from an immediate undo
            self.last_cell = (act["row"] // 3, act["col"] // 3) if before.grid[act["row"]][act["col"]] != g.frame.grid[act["row"]][act["col"]] else None
        else:
            self.last_cell = None
        v = ev.check(pred, t, res)
        self.last_verdict = (pred, v, res)
        self.metrics[f"pred_{pred.kind}"] += 1; self.metrics["verdict_ok" if v.ok else "verdict_mismatch" if v.ok is False else "verdict_obs"] += 1
        tag = "OK " if v.ok else "MISMATCH" if v.ok is False else "obs"
        self.log(f"{tag}: {action_label(act)} [{label}] predicted: {pred.text} | actual: {v.text}")
        self.outcomes.append(f"a{g.actions_used} {action_label(act)} ({label}): predicted '{pred.text[:70]}' -> {v.text[:110]}")
        self.outcomes = self.outcomes[-14:]
        if res["level_completed"] or res["won"]:
            ev2 = self.evidence(); changes = self.sync(ev2)
            wins = self.book.sync(ev2.win_facts(level0), level=level0)
            attempt = [x for x in g.level_transitions(level0) if x.attempt == t.attempt]
            try:
                post = ev.synth_board(pred, act)
                new_preds = goals2.infer(self.level_start_frame or attempt[0].before_frame, before, post, t.action, ev)
            except Exception as e:
                self.log(f"goal inference failed: {e!r}"); new_preds = []
            pfacts = [p_.fact() for p_ in new_preds]
            wins += self.book.sync(pfacts, level=level0)
            for p_ in new_preds:
                e = self.book.find(p_.kind, p_.fact()["params"], exact=True)
                if e is not None:
                    self.preds[e.id] = p_
            if wins:
                self.log("win facts: " + " | ".join(wins)); self.save_book()
            try:
                essential = goals2.essential_sequence(attempt)
            except Exception as e:
                essential = [f"(sequence unavailable: {e!r})"]
            self.log("essential winning sequence: " + " ; ".join(essential))
            self.last_essential = essential
            seq = [action_label(x.action) for x in attempt]
            if g.state != "WIN":
                extra = ("ESSENTIAL WINNING SEQUENCE (program-filtered: actions that changed nothing in the playfield removed; object ids refer to the old board):\n  "
                         + "\n  ".join(essential[-24:]) +
                         "\nWIN PREDICATES THE PROGRAM FOUND TRUE AT COMPLETION (each is verified/refuted automatically on this level):\n  "
                         + ("\n  ".join(f"{e.id}: {e.text}" for e in self.book.section("win") if e.id in self.preds) or "(none)"))
                self.review(f"LEVEL {level0} COMPLETED after {g.level_action_log[-1]} actions on the last attempt ({len(seq)} actions, {len(essential)} essential). "
                            f"The last action was {action_label(act)} ({label}). Level {g.level} starts now (new board below). Write the PROCEDURE that won, in terms of "
                            f"object roles (which objects to click in which order and why), so that it can be repeated on the new board; state which win condition proved "
                            f"true; and write the plan for the new level.", changes + wins, level_event=True, extra=extra)
            return "won" if g.state == "WIN" else "level"
        if res["game_over"]:
            n_att = g.actions_used - self.attempt_start_actions
            self.game_over_at.setdefault(g.level, []).append(n_att)
            g.reset(); self.level_start_frame = g.frame; self.attempt_start_actions = g.actions_used
            self.metrics["game_over"] += 1
            ev2 = self.evidence(); changes = self.sync(ev2)
            lim = self.attempt_limit()
            if lim:
                changes = changes + [f"the attempt ends in a GAME OVER after about {lim} actions on this level (seen {len(self.game_over_at[g.level])}x): the level must be finished within that budget"]
            self.review(f"GAME OVER after {action_label(act)} ({label}): the level was reset. Predicted: {pred.text}. Observed: {v.text}. "
                        "Add the rule that explains the game over (hazard / limit) and adjust the plan.", changes, before=before, bbox=v.diff.get("bbox"))
            return "over"
        # win predicates: a submit that did not end the level refutes the conditions that held and the button; a condition that
        # holds in a game without a submit button, while the level goes on, is refuted too
        if self.preds:
            gs = self.goal_state(self.evidence())
            if kind == "submit":
                self.refute_preds(gs["held"] + [eid for eid, p_ in self.live_preds() if isinstance(p_, goals2.Pressed)],
                                  "the submit button was pressed while these held, but the level did not end")
            elif not gs["has_submit"] and gs["held"]:
                self.refute_preds(gs["held"], "held on the board but the level did not end (and there is no submit button)")
        if v.ok is False:
            self.mismatches += 1
            if isinstance(act, dict) and pred.kind in ("board", "objects", "cursor"):
                key = (g.level, ev.click_key(act["row"], act["col"])[0]); self.distrust[key] = self.distrust.get(key, 0) + 1
            ev2 = self.evidence(); changes = self.sync(ev2)
            if review_mismatch and self.reviews - reviews0 < self.cfg["reviews_per_level"]:
                self.review(f"MISMATCH on {action_label(act)} ({label}). Predicted: {pred.text}. Observed: {v.text}. "
                            "Which rulebook entry was wrong, and what rule explains the observation?", changes, before=before, bbox=v.diff.get("bbox"))
            elif not review_mismatch:
                self.pending_mismatch = (act, label, pred.text, v.text, before, v.diff.get("bbox"), changes)
            else:
                self.log("review cap reached for this level; programmatic update only")
            return "mismatch"
        if v.ok is None:
            self.observations += 1
            if kind in ("click", "key", "interact", "code"):
                ev2 = self.evidence(); self.sync(ev2)   # the harness scores the book; the model sees the changes on its next decision
        return "ok"
