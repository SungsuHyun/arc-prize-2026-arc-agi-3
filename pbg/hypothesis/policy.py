"""hypothesis/policy.py — the one policy for every game:

    LOOK      the board and what actions did so far
    HYPOTHESISE   a document of the whole game (roles, controls, dynamics, win) + the tests that would settle the doubts
    TEST      run those 1-3 actions (never a blind walk)
    VERIFY    the hypothesis at pixel level on the level's log
    REVISE    with counter-examples until it verifies
    PLAN      the shortest action sequence to the hypothesised win; execute; a mismatch is a new counter-example

The mechanism priors are hints in the prompt, not a separate process. Everything is logged as events so the trail
"hypothesis -> test -> verdict -> revision" is readable in the replay viewer."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.budget import Budget
from ..core.events import EventLog
from ..core.types import Action, Scene, Transition
from ..memory.priors.mechanisms import mechanism_signatures
from ..memory.store import Memory
from ..orchestrator.loop import Result
from ..orchestrator.session import Session
from ..perception import Perception
from ..planner.astar import astar
from ..planner.common import state_key
from ..wml.sandbox import Sandbox
from .contract import hypothesis_prompt
from .verify import Verdict, verify, pixel_equal, _boxes


class Hypothesis:
    def __init__(self, code: str, ns: dict, n: int):
        self.code, self.ns, self.n = code, ns, n
        self.doc = ns.get("HYPOTHESIS") or {}
        self.model = ns["build_model"]()
        self.goal = ns["build_goal"]() if "build_goal" in ns else None
        self.verdict: Optional[Verdict] = None

    def call(self, name: str, scene: Scene, default):
        fn = self.ns.get(name)
        if fn is None:
            return default
        try:
            out = fn(scene)
            return out if out is not None else default
        except Exception:
            return default

    def ignore(self, scene: Scene) -> list:
        return self.call("ignore_boxes", scene, [])

    def summary(self) -> str:
        d = self.doc
        v = self.verdict
        vs = f"acc={v.accuracy:.2f} cov={v.coverage:.2f} n={v.n}" if v else "unverified"
        return f"h{self.n} [{vs}] {str(d.get('summary', ''))[:220]} | win: {str(d.get('win', ''))[:120]} | uncertain: {d.get('uncertain', [])}"


class HypothesisPolicy:
    def __init__(self, *, memory: Memory, llm, log=None, events_dir: Optional[Path] = None, max_seconds: Optional[float] = None,
                 budget_cfg: Optional[dict] = None, perception_cfg: Optional[dict] = None, K: int = 2, min_acc: float = 0.8,
                 max_rounds_per_level: int = 14, plan_time: float = 8.0):
        self.memory, self.llm = memory, llm
        self.log = log or (lambda *a, **k: None)
        self.events_dir, self.max_seconds = events_dir, max_seconds
        self.budget_cfg, self.perception_cfg = budget_cfg, perception_cfg
        self.K, self.min_acc, self.max_rounds, self.plan_time = K, min_acc, max_rounds_per_level, plan_time
        self.sandbox = Sandbox(log=self.log, allow_getattr=True)
        self.load_errors: list[str] = []

    # ── hypothesising ──
    def _generate(self, msgs: list[dict], k: int, out: list) -> None:
        try:
            kmsgs = msgs if k == 0 else msgs + [{"role": "user", "content": f"Alternative {k}: consider a different reading of the controls or the win condition; still return ONE complete ```python block."}]
            reply = self.llm.chat(kmsgs, purpose="world_model", use_cache=(k == 0))
            code = self.llm.extract_code(reply)
            if not code:
                self.log(f"hypothesis candidate {k}: no code block"); return
            ns = self.sandbox.load_namespace(code)
            if ns is None or "build_model" not in ns:
                err = self.sandbox.last_error or "build_model missing"
                self.log(f"hypothesis candidate {k}: load failed: {err[:160]}")
                self.load_errors.append(f"Your previous code failed to load: {err[:300]} -- fix it (numpy only, no imports beyond numpy, define build_model/build_goal/candidate_actions/test_actions/ignore_boxes).")
                return
            out.append(code)
        except Exception as e:
            self.log(f"hypothesis candidate {k} failed: {e!r}")

    def hypothesise(self, s: Session, log: list[Transition], prev: Optional[Hypothesis], counterexamples: list[str], n: int, rejected: list[str]) -> list[Hypothesis]:
        grid = np.asarray(s.frame.grid)
        errs = self.load_errors[-2:]; self.load_errors = []
        msgs = hypothesis_prompt(scene=s.scene, grid=grid, log=log, available=s.available_actions(), prior_signatures=mechanism_signatures(),
                                 previous_code=prev.code if prev else None, counterexamples=list(counterexamples) + errs, level=s.level,
                                 ignore_boxes=prev.ignore(s.scene) if prev else (), rejected=rejected)
        codes: list = []
        threads = [threading.Thread(target=self._generate, args=(msgs, k, codes), daemon=True) for k in range(self.K)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=float(self.llm.cfg.get("timeout", 300)) + 30)
        out = []
        for code in codes:
            ns = self.sandbox.load_namespace(code)
            if ns is None:
                continue
            try:
                out.append(Hypothesis(code, ns, n))
            except Exception as e:
                self.log(f"hypothesis build failed: {e!r}")
                self.load_errors.append(f"Your previous code loaded but build_model()/build_goal() raised {type(e).__name__}: {str(e)[:200]} -- fix it.")
        return out

    # ── planning on a verified hypothesis ──
    def plan(self, h: Hypothesis, s: Session, observed: dict) -> Optional[list[Action]]:
        goal = h.goal
        if goal is None:
            return None
        available = s.available_actions()
        def actions_fn(scene: Scene) -> list[Action]:
            cands = h.call("candidate_actions", scene, [])
            cands = [a for a in cands if isinstance(a, Action)]
            if not cands:
                cands = [a for a in available if a.type == "BUTTON"]
            return cands[:48]
        def successors(scene: Scene):
            sk = state_key(scene)
            for a in actions_fn(scene):
                nxt = observed.get((sk, a.label()))
                if nxt is None:
                    try:
                        nxt = h.model.predict(scene, a)
                    except Exception:
                        nxt = None
                if nxt is None:
                    continue
                yield a, nxt
        try:
            if goal.is_goal(s.scene):
                return []
            return astar(s.scene, successors, goal.is_goal, goal.progress, depth=80, time_limit=self.plan_time)
        except Exception as e:
            self.log(f"planning failed: {e!r}")
            return None

    # ── the loop ──
    def play(self, env, game_id: str, *, budget_total: Optional[int] = None) -> Result:
        t_start = time.time()
        budget = Budget(budget_total, self.budget_cfg)
        perception = Perception(self.perception_cfg)
        events = EventLog((self.events_dir / f"{game_id}.events.jsonl") if self.events_dir else None)
        s = Session(env, game_id, perception, self.memory, budget, log=self.log, max_seconds=self.max_seconds)
        s.start()
        hyp: Optional[Hypothesis] = None
        counterexamples: list[str] = []; rejected: list[str] = []
        observed: dict = {}
        level = s.level; rounds = 0; n_hyp = 0; stop = ""; llm_calls0 = self.llm.calls
        plans_executed = 0; tests_run = 0; tried_labels: set = set(); idle_rounds = 0

        def act(a: Action, kind: str) -> Optional[Transition]:
            t = s.act(a, kind)
            if t is not None and t.status_change is None and t.action.type != "RESET":
                observed[(state_key(t.before), a.label())] = t.after
            return t

        events.emit("START", "LOOK", "hypothesis policy: look, hypothesise, test, verify, revise, plan", budget_used=0)
        while not s.finished():
            if s.timed_out():
                stop = "timeout"; break
            if s.env.status().state == "GAME_OVER":
                events.emit("LOOK", "LOOK", "game over -> RESET (hypothesis kept)", budget_used=budget.used())
                s.act(Action.reset(), "reset"); continue
            if s.level != level:
                events.emit("LOOK", "LOOK", f"level {level} -> {s.level}: the hypothesis must verify again on the new board", budget_used=budget.used())
                level = s.level; rounds = 0; idle_rounds = 0; counterexamples = []; observed.clear()
                if hyp is not None:
                    hyp.verdict = None
            log = s.level_log()
            # verify the current hypothesis on this level's log
            if hyp is not None and log:
                hyp.verdict = verify(hyp.model, log, hyp.ignore)
            usable = hyp is not None and (hyp.verdict is not None and hyp.verdict.usable(self.min_acc))
            if usable:
                plan = self.plan(hyp, s, observed)
                if plan is None or (plan == [] and not hyp.goal.is_goal(s.scene)):
                    events.emit("PLAN", "TEST", f"no plan under h{hyp.n} for goal '{getattr(hyp.goal, 'name', '?')}' -> test the doubts", budget_used=budget.used())
                    counterexamples = (counterexamples + [f"Your model verified but the planner found NO action sequence from the current board to your goal '{getattr(hyp.goal, 'name', '?')}' using candidate_actions: the goal, the candidate actions or the dynamics are incomplete."])[-4:]
                    hyp.verdict = None
                    usable = False
                elif plan == []:
                    events.emit("PLAN", "TEST", "goal already true but no level-up: the win condition is wrong", budget_used=budget.used())
                    counterexamples.append("The goal predicate you wrote is already TRUE on the current board, yet the level did not end: the win condition is different.")
                    rejected.append(f"win='{str(hyp.doc.get('win', ''))[:80]}'")
                    usable = False
                else:
                    events.emit("PLAN", "EXECUTE", f"plan of {len(plan)} under h{hyp.n}: {[a.label() for a in plan[:10]]}", budget_used=budget.used())
                    plans_executed += 1
                    for a in plan:
                        pred = None
                        try:
                            pred = hyp.model.predict(s.scene, a)
                        except Exception:
                            pred = None
                        t = act(a, "plan")
                        if t is None:
                            break
                        if t.status_change:
                            events.emit("EXECUTE", "LOOK", f"{t.status_change} after {a.label()}", transition_id=t.id, budget_used=budget.used()); break
                        if pred is not None:
                            ok, n, win = pixel_equal(pred, t.after_frame.grid, _boxes(t.before, hyp.ignore))
                            if not ok:
                                ce = verify(hyp.model, [t], hyp.ignore).counterexamples
                                counterexamples = (counterexamples + ce)[-4:]
                                events.emit("EXECUTE", "REVISE", f"mismatch after {a.label()} ({n} px): counter-example recorded", transition_id=t.id, budget_used=budget.used())
                                break
                    continue
            # not usable: hypothesise (or revise), then run its tests
            if idle_rounds >= self.max_rounds:
                # only rounds that produced NO action count toward giving up: as long as tests or plans act, keep going until the clock ends
                stop = "UNRESOLVED"; events.emit("LOOK", "END", f"{idle_rounds} hypothesis rounds without any action", budget_used=budget.used()); break
            if self.llm.exhausted():
                stop = "llm_exhausted"; events.emit("LOOK", "END", "LLM call cap reached", budget_used=budget.used()); break
            rounds += 1; n_hyp += 1
            used_before = budget.used()
            t0 = time.time()
            cands = self.hypothesise(s, log, hyp, counterexamples, n_hyp, rejected)
            for c in cands:
                c.verdict = verify(c.model, log, c.ignore) if log else Verdict(0.0, 0.0, 0, 0, 0)
            cands.sort(key=lambda c: (c.verdict.accuracy, c.verdict.coverage), reverse=True)
            events.emit("HYPOTHESISE", "HYPOTHESISE", f"round {rounds}: {len(cands)} candidate(s) in {time.time() - t0:.0f}s", budget_used=budget.used())
            for c in cands:
                events.emit("HYPOTHESISE", "HYPOTHESISE", c.summary(), budget_used=budget.used())
                if self.events_dir:
                    hd = self.events_dir / f"{game_id}.hypotheses"; hd.mkdir(exist_ok=True)
                    (hd / f"h{c.n}_r{rounds}_acc{c.verdict.accuracy:.2f}.py").write_text(c.code + "\n\n# verdict: " + c.summary()
                                                                                + "\n" + "\n".join("# " + l for ce in c.verdict.counterexamples[:2] for l in ce.splitlines()))
            if not cands:
                continue
            best = cands[0]
            if hyp is None or hyp.verdict is None or best.verdict.accuracy >= hyp.verdict.accuracy:
                hyp = best
            if hyp.verdict.counterexamples:
                counterexamples = hyp.verdict.counterexamples[-4:]
            if hyp.verdict.usable(self.min_acc):
                events.emit("VERIFY", "PLAN", f"h{hyp.n} verified (acc {hyp.verdict.accuracy:.2f} on {hyp.verdict.n}) -> plan", budget_used=budget.used())
                idle_rounds = idle_rounds + 1 if budget.used() == used_before else 0
                continue
            tests = [a for a in hyp.call("test_actions", s.scene, []) if isinstance(a, Action)][:3]
            if not tests:
                tests = [a for a in hyp.call("candidate_actions", s.scene, []) if isinstance(a, Action)][:2]
            if not tests:
                tests = _fallback_probe(s, tried_labels)
                events.emit("TEST", "TEST", f"h{hyp.n} names no test -> fallback probe {[a.label() for a in tests]}", budget_used=budget.used())
            if not tests:
                events.emit("TEST", "HYPOTHESISE", f"h{hyp.n} names no test and nothing is left to probe: revise", budget_used=budget.used())
                continue
            events.emit("VERIFY", "TEST", f"h{hyp.n} not verified (acc {hyp.verdict.accuracy:.2f}, n {hyp.verdict.n}) -> tests {[a.label() for a in tests]}", budget_used=budget.used())
            for a in tests:
                t = act(a, "test")
                if t is None:
                    break
                tests_run += 1; tried_labels.add(a.label())
                if t.status_change:
                    events.emit("TEST", "LOOK", f"{t.status_change} after test {a.label()}", transition_id=t.id, budget_used=budget.used()); break
            idle_rounds = idle_rounds + 1 if budget.used() == used_before else 0
        st = s.env.status()
        if not stop:
            stop = st.state if st.state in ("WIN", "GAME_OVER") else ("timeout" if s.timed_out() else "budget")
        events.emit("LOOK", "END", stop, budget_used=budget.used())
        self.memory.save()
        return Result(game_id, st.state, st.level - 1 if st.state != "WIN" else (st.levels_total or st.level), st.levels_total, st.actions_used, budget.snapshot(),
                      self.llm.calls - llm_calls0, len(events.events), stop, dict(s.level_actions),
                      [{"name": f"h{hyp.n}", "score": round(hyp.verdict.accuracy, 3) if hyp.verdict else 0.0, "coverage": round(hyp.verdict.coverage, 3) if hyp.verdict else 0.0,
                        "verified": bool(hyp.verdict and hyp.verdict.usable(self.min_acc)), "origin": "hypothesis"}] if hyp else [],
                      [{"name": getattr(hyp.goal, "name", "?"), "win": hyp.doc.get("win", "")}] if hyp and hyp.goal else [],
                      {"rounds": n_hyp, "plans": plans_executed, "tests": tests_run}, round(time.time() - t_start, 1))


def _fallback_probe(s, tried: set) -> list:
    """One click on an object class not tried yet on this level (buttons first if the game has them)."""
    from ..probe.clickmap import object_key
    avail = s.available_actions()
    out = [a for a in avail if a.type == "BUTTON" and a.label() not in tried][:1]
    if out or not any(a.type == "CLICK" for a in avail):
        return out
    seen = set()
    strips = {r.id for r in s.scene.regions if r.kind_hint == "ui_strip"}
    for o in sorted(s.scene.objects, key=lambda o: o.area):
        k = object_key(o)
        if o.region in strips or k in seen:
            continue
        seen.add(k)
        a = Action.click(*o.center)
        if a.label() not in tried:
            return [a]
    return []
