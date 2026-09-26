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

DEFAULTS = {"max_minutes": 20.0, "level_actions": 200, "max_actions": 2000, "reviews_per_level": 8, "sync_every": 4, "max_levels": 10}


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
        self._rules_cache = None; self._rules_at = -1
        self.last_choice: list[str] = []

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
        ev = Evidence(self.g)
        return ev

    def sync(self, ev: Evidence) -> list[str]:
        changes = self.book.sync(ev.facts(), level=self.g.level)
        if changes:
            self.log("rulebook <- evidence: " + " | ".join(changes)); self.save_book()
        return changes

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

    def review(self, event: str, changes: list[str], *, before=None, bbox=None) -> None:
        if self.model is None:
            return
        self.reviews += 1
        obj = self.model.review(self.g, self.book, event, changes, self.outcomes, before=before, bbox=bbox)
        if not obj:
            return
        done = self.book.apply_edits(obj.get("edits") or [], level=self.g.level)
        if obj.get("plan"):
            self.book.plan = str(obj["plan"])[:600]
        self.log(f"REVIEW ({event[:60]}): {done}\n" + self.book.render()); self.save_book()

    def choose(self, cands: list[Candidate], budget_text: str, ev: Evidence) -> Candidate:
        fallback = cands[0]   # untested first, then by priority (deterministic policy)
        if self.model is None or not cands:
            return fallback
        obj = self.model.decide(self.g, self.book, cands, self.outcomes, budget_text)
        if not obj:
            return fallback
        done = self.book.apply_edits(obj.get("edits") or [], level=self.g.level)
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
        while g.level == level0 and g.state != "WIN":
            if self._out_of_time():
                return self._lv(level0, a0, decisions, "time/actions budget")
            if g.actions_used - a0 >= self.cfg["level_actions"]:
                return self._lv(level0, a0, decisions, "level action cap")
            ev = self.evidence(); self.sync(ev)
            cands = build(g, ev, self.tried)
            if not cands:
                return self._lv(level0, a0, decisions, "no candidates")
            budget = f"actions used on this level {g.actions_used - a0} (cap {self.cfg['level_actions']}), total {g.actions_used}"
            cand = self.choose(cands, budget, ev); decisions += 1
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
                if wins:
                    self.log("win facts: " + " | ".join(wins)); self.save_book()
                seq = [action_label(x.action) for x in g.level_transitions(level0) if x.attempt == t.attempt]
                if g.state != "WIN":
                    self.review(f"LEVEL {level0} COMPLETED after {g.level_action_log[-1]} actions on the last attempt. Winning action sequence: {seq[-30:]}. "
                                f"The last action was {action_label(act)} ({cand.label}). Level {g.level} starts now (new board below): state which win condition proved "
                                f"true, mark it confirmed, and write the plan for the new level.", changes + wins)
                return "won" if g.state == "WIN" else "level"
            if res["game_over"]:
                g.reset()
                ev2 = self.evidence(); changes = self.sync(ev2)
                self.review(f"GAME OVER after {action_label(act)} ({cand.label}): the level was reset. Predicted: {pred.text}. Observed: {v.text}. "
                            "Add the rule that explains the game over (hazard / limit) and adjust the plan.", changes, before=before, bbox=v.diff.get("bbox"))
                return "over"
            if v.ok is False:
                self.mismatches += 1
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
