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

DEFAULTS = {"max_minutes": 20.0, "level_actions": 200, "max_actions": 2000, "reviews_per_level": 8, "max_levels": 10,
            "review_think": "level"}   # thinking during reviews: "level" = only after a level completion, "always", "never"


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
        self.last_choice: list[str] = []
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
        return Evidence(self.g, distrust={key: n for (l, key), n in self.distrust.items() if l == lv})

    def sync(self, ev: Evidence) -> list[str]:
        changes = self.book.sync(ev.facts(), level=self.g.level)
        if changes:
            self.log("rulebook <- evidence: " + " | ".join(changes)); self.save_book()
        return changes

    # ── win predicates ───────────────────────────────────────────────────
    def live_preds(self) -> list:
        return [(eid, p) for eid, p in self.preds.items() if (e := self.book.get(eid)) is not None and e.status != "refuted"]

    def goal_state(self, ev: Evidence) -> dict:
        """Evaluate every live win predicate on the current board: progress lines for the model, and whether a submit is due."""
        sc = ev.scene; lines = []; held = []; missing = []; submit = None; has_submit = False
        for eid, p in self.live_preds():
            if isinstance(p, goals2.Pressed):
                has_submit = True; b = p.button(sc)
                if b is not None:
                    submit = b
                lines.append(f"{eid} [submit] {p.evaluate(sc)[1]}"); continue
            ok, txt, _ = p.evaluate(sc)
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
        obj = self.model.init_rulebook(self.g)
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

    def choose(self, cands: list[Candidate], budget_text: str, ev: Evidence, goal: Optional[dict] = None) -> Candidate:
        cands = [c for c in cands if c.kind != "info"] or cands
        fallback = cands[0]   # untested first, then by priority (deterministic policy)
        if self.model is None or not cands:
            return fallback
        extra = ("WIN CONDITION PROGRESS (program-evaluated on the current board):\n  " + "\n  ".join(goal["lines"])) if goal and goal.get("lines") else ""
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
            self.log(f"DECIDE: label {label!r} not a candidate -> fallback {fallback.label}"); return fallback
        # wandering guards: a candidate already tried in this world state that is predicted to change nothing (or only a counter)
        # is a wasted action; so is the same label three times in a row
        self.last_choice.append(c.label)
        repeat = len(self.last_choice) >= 3 and len(set(self.last_choice[-3:])) == 1
        if c.tested and (c.pred_kind in ("noop", "hud") or repeat):
            alt = next((x for x in cands if not x.tested and x.pred_kind not in ("noop", "hud")), None) or next((x for x in cands if not x.tested), None)
            if alt:
                self.log(f"DECIDE: {c.label} already tried here and predicted '{c.pred_kind}'{' (3x in a row)' if repeat else ''} -> {alt.label}")
                self.outcomes.append(f"(program vetoed {c.label}: already tried here, no world change expected; ran {alt.label} instead)")
                return alt
        self.log(f"DECIDE: {c.label} (expect: {str(obj.get('expect', ''))[:120]})")
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
               "model_calls": self.model.calls if self.model else {}, "model_seconds": round(self.model.seconds, 1) if self.model else 0}
        self.log("result: " + json.dumps(res)); self.save_book(); self._log_f.close()
        return res

    def play_level(self) -> dict:
        g = self.g; level0 = g.level; a0 = g.actions_used; reviews0 = self.reviews; decisions = 0
        self.level_start_frame = g.frame
        while g.level == level0 and g.state != "WIN":
            if self._out_of_time():
                return self._lv(level0, a0, decisions, "time/actions budget")
            if g.actions_used - a0 >= self.cfg["level_actions"]:
                return self._lv(level0, a0, decisions, "level action cap")
            ev = self.evidence(); self.sync(ev)
            goal = self.goal_state(ev)
            cands = build(g, ev, self.tried, goal=goal)
            plans = make_plans(g, ev, self.live_preds())
            for c in plans:
                self.log(f"plan candidate: {c.label} -> {c.prediction[:160]}")
            cands = [c for c in plans if c.kind == "plan"] + cands + [c for c in plans if c.kind != "plan"]
            if not [c for c in cands if c.kind != "info"]:
                return self._lv(level0, a0, decisions, "no candidates")
            budget = f"actions used on this level {g.actions_used - a0} (cap {self.cfg['level_actions']}), total {g.actions_used}"
            if goal["lines"]:
                self.log("goal state: " + " | ".join(goal["lines"]))
            cand = self.choose(cands, budget, ev, goal); decisions += 1
            self.tried.add((state_key(g, ev), cand.label))
            res = self.execute(cand, level0, reviews0)
            if res == "won":
                return self._lv(level0, a0, decisions, "completed")
        return self._lv(level0, a0, decisions, "completed" if g.level > level0 else "left")

    def _lv(self, level0, a0, decisions, reason) -> dict:
        return {"level": level0, "completed": self.g.level > level0 or self.g.state == "WIN", "actions": self.g.actions_used - a0, "decisions": decisions, "reason": reason}

    def execute(self, cand: Candidate, level0: int, reviews0: int) -> str:
        """Run the candidate's actions one by one, each checked against its prediction. Returns 'won' | 'level' | 'over' | 'mismatch' | 'ok'."""
        g = self.g
        for i, act in enumerate(cand.actions):
            ev = self.evidence()
            pred = ev.predict(act)
            before = g.frame
            res = g.step(act)
            if res.get("invalid"):
                self.outcomes.append(f"{cand.label}: {action_label(act)} invalid now"); return "ok"
            t = g.transitions[-1]
            v = ev.check(pred, t, res)
            tag = "OK " if v.ok else "MISMATCH" if v.ok is False else "obs"
            line = f"{action_label(act)} [{cand.label}] predicted: {pred.text} | actual: {v.text}"
            self.log(f"{tag}: {line}")
            self.outcomes.append(f"a{g.actions_used} {action_label(act)} ({cand.label}): predicted '{pred.text[:70]}' -> {v.text[:110]}")
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
                pfacts = [p.fact() for p in new_preds]
                wins += self.book.sync(pfacts, level=level0)
                for p in new_preds:
                    e = self.book.find(p.kind, p.fact()["params"], exact=True)
                    if e is not None:
                        self.preds[e.id] = p
                if wins:
                    self.log("win facts: " + " | ".join(wins)); self.save_book()
                try:
                    essential = goals2.essential_sequence(attempt)
                except Exception as e:
                    essential = [f"(sequence unavailable: {e!r})"]
                self.log("essential winning sequence: " + " ; ".join(essential))
                seq = [action_label(x.action) for x in attempt]
                if g.state != "WIN":
                    extra = ("ESSENTIAL WINNING SEQUENCE (program-filtered: actions that changed nothing in the playfield removed; object ids refer to the old board):\n  "
                             + "\n  ".join(essential[-24:]) +
                             "\nWIN PREDICATES THE PROGRAM FOUND TRUE AT COMPLETION (each is verified/refuted automatically on this level):\n  "
                             + ("\n  ".join(f"{e.id}: {e.text}" for e in self.book.section("win") if e.id in self.preds) or "(none)"))
                    self.review(f"LEVEL {level0} COMPLETED after {g.level_action_log[-1]} actions on the last attempt ({len(seq)} actions, {len(essential)} essential). "
                                f"The last action was {action_label(act)} ({cand.label}). Level {g.level} starts now (new board below). Write the PROCEDURE that won, in terms of "
                                f"object roles (which objects to click in which order and why), so that it can be repeated on the new board; state which win condition proved "
                                f"true; and write the plan for the new level.", changes + wins, level_event=True, extra=extra)
                return "won" if g.state == "WIN" else "level"
            if res["game_over"]:
                g.reset(); self.level_start_frame = g.frame
                ev2 = self.evidence(); changes = self.sync(ev2)
                self.review(f"GAME OVER after {action_label(act)} ({cand.label}): the level was reset. Predicted: {pred.text}. Observed: {v.text}. "
                            "Add the rule that explains the game over (hazard / limit) and adjust the plan.", changes, before=before, bbox=v.diff.get("bbox"))
                return "over"
            # win predicates: a submit that did not end the level refutes the conditions that held and the button; a condition that
            # holds in a game without a submit button, while the level goes on, is refuted too
            if self.preds:
                gs = self.goal_state(self.evidence())
                if cand.kind == "submit":
                    self.refute_preds(gs["held"] + [eid for eid, p in self.live_preds() if isinstance(p, goals2.Pressed)],
                                      "the submit button was pressed while these held, but the level did not end")
                elif not gs["has_submit"] and gs["held"]:
                    self.refute_preds(gs["held"], "held on the board but the level did not end (and there is no submit button)")
            if v.ok is False:
                self.mismatches += 1
                if isinstance(act, dict) and pred.kind in ("board", "objects", "cursor"):
                    key = (g.level, ev.click_key(act["row"], act["col"])[0]); self.distrust[key] = self.distrust.get(key, 0) + 1
                ev2 = self.evidence(); changes = self.sync(ev2)
                if self.reviews - reviews0 < self.cfg["reviews_per_level"]:
                    self.review(f"MISMATCH on {action_label(act)} ({cand.label}). Predicted: {pred.text}. Observed: {v.text}. "
                                "Which rulebook entry was wrong, and what rule explains the observation?", changes, before=before, bbox=v.diff.get("bbox"))
                else:
                    self.log("review cap reached for this level; programmatic update only")
                return "mismatch"
            if v.ok is None:
                self.observations += 1
                if i == len(cand.actions) - 1 or cand.kind in ("click", "key", "interact"):
                    # an untested action was observed: score the book against the new evidence, no review needed unless it contradicts
                    ev2 = self.evidence(); self.sync(ev2)   # the harness scores the book; the model sees the changes on its next decision
            if cand.kind == "key-run" and not res["changed"]:
                break
        return "ok"
