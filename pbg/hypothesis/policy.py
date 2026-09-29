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
import zlib
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.budget import Budget
from ..core.contracts import RuleModel
from ..core.events import EventLog
from ..core.types import Action, Scene, Transition
from ..memory.priors.mechanisms import mechanism_signatures
from ..memory.store import Memory
from ..orchestrator.loop import Result
from ..orchestrator.session import Session
from ..perception import Perception
from ..planner.astar import astar
from ..planner.common import action_set, state_key
from ..probe.semantics import classify_actions
from ..goal.infer import GoalInference
from ..wml.refine import WorldModelLab
from ..wml.sandbox import Sandbox
from .contract import hypothesis_prompt
from .explore import explore_level
from .guard import GuardedModel, ModelTimeout, call_with_deadline
from .verify import Verdict, verify, pixel_equal, _boxes


MODEL_SECONDS = 2.0     # deadline per call into LLM-written code (docs/029: an unbounded loop froze a game thread)


class Hypothesis:
    def __init__(self, code: str, ns: dict, n: int):
        self.code, self.ns, self.n = code, ns, n
        self.doc = ns.get("HYPOTHESIS") or {}
        self.model = GuardedModel(call_with_deadline(ns["build_model"], seconds=MODEL_SECONDS), MODEL_SECONDS)
        self.goal = call_with_deadline(ns["build_goal"], seconds=MODEL_SECONDS) if "build_goal" in ns else None
        self.verdict: Optional[Verdict] = None
        self.timeouts = 0            # deadline hits in goal / helper functions (model calls are counted by GuardedModel)
        self.origin = "llm"

    def timeout_note(self) -> str:
        """Counter-example text when this hypothesis' code hit the deadline (empty otherwise)."""
        n = self.timeouts + getattr(self.model, "timeouts", 0)
        if not n:
            return ""
        where = getattr(self.model, "last_timeout", "") or "a goal/helper function"
        return (f"Your code ran past the {MODEL_SECONDS:.0f}-second limit {n} time(s) ({where}): it has an unbounded loop. "
                "Every loop must be bounded by the grid size, the object count or a fixed step count; the timed-out calls counted as wrong predictions.")

    def roled(self, scene: Scene) -> Scene:
        """The scene with this hypothesis' roles applied (every hypothesis function expects them)."""
        try:
            return self.model.with_roles(scene) if hasattr(self.model, "with_roles") else scene
        except Exception:
            return scene

    def _timed(self, fn, *args, default):
        if self.timeouts + getattr(self.model, "timeouts", 0) >= 2:
            return default                # dead code: spend no more time on it
        try:
            out = call_with_deadline(fn, *args, seconds=MODEL_SECONDS)
            return out if out is not None else default
        except ModelTimeout:
            self.timeouts += 1
            return default
        except Exception:
            return default

    def call(self, name: str, scene: Scene, default):
        fn = self.ns.get(name)
        if fn is None:
            return default
        return self._timed(fn, self.roled(scene), default=default)

    def is_goal(self, scene: Scene) -> bool:
        if self.goal is None:
            return False
        return bool(self._timed(self.goal.is_goal, self.roled(scene), default=False))

    def progress(self, scene: Scene) -> float:
        if self.goal is None:
            return 0.0
        try:
            return float(self._timed(self.goal.progress, self.roled(scene), default=0.0))
        except Exception:
            return 0.0

    def ignore(self, scene: Scene) -> list:
        return self.call("ignore_boxes", scene, [])

    def summary(self) -> str:
        d = self.doc
        v = self.verdict
        vs = f"acc={v.accuracy:.2f} chg={v.change_accuracy:.2f}({v.changed}) approx={v.approx:.2f} cov={v.coverage:.2f} n={v.n}" if v else "unverified"
        if self.timeouts + getattr(self.model, "timeouts", 0):
            vs += f" TIMEOUTS={self.timeouts + getattr(self.model, 'timeouts', 0)}"
        return f"h{self.n} [{vs}] {str(d.get('summary', ''))[:220]} | win: {str(d.get('win', ''))[:120]} | uncertain: {d.get('uncertain', [])}"


class InducedHypothesis(Hypothesis):
    """A prior-composed (deterministic) world model as a hypothesis candidate (docs/029 P1): the induction of wml/
    (move / wall / collect / click-permutation / effect-table rules from the action semantics) competes with the LLM's
    hypotheses under the same pixel verifier, paired with the goal templates ranked for its roles. This is the model-free
    path: on games where the local model writes nothing pixel-exact (lp85), the composed model still plans."""

    def __init__(self, wh, goals: list, n: int, actions_fn):
        self.code, self.ns, self.n = wh.code or "", {}, n
        self.name = wh.name
        self.doc = {"summary": f"induced model '{wh.name}' composed from the action semantics and the mechanism priors",
                    "win": goals[0].name if goals else "(no goal template fits)", "uncertain": []}
        self.model = GuardedModel(wh.model, MODEL_SECONDS)
        self.goals = list(goals)
        self.goal = self.goals[0] if self.goals else None
        self.verdict: Optional[Verdict] = None
        self.timeouts = 0
        self.origin = "induced"
        self._actions_fn = actions_fn

    def call(self, name: str, scene: Scene, default):
        if name == "candidate_actions":
            try:
                return self._actions_fn(scene) or default
            except Exception:
                return default
        return default


GD_STEPS_PER_LEVEL = 6       # goal-directed steps a verified-but-planless model may take per level before the model is questioned
APPROX_ACTIONS_PER_LEVEL = 24  # actions an approximate model may spend per level on closed-loop plans (ls20 spent 66 for nothing)


class HypothesisPolicy:
    def __init__(self, *, memory: Memory, llm, log=None, events_dir: Optional[Path] = None, max_seconds: Optional[float] = None,
                 budget_cfg: Optional[dict] = None, perception_cfg: Optional[dict] = None, K: int = 2, min_acc: float = 0.8,
                 max_rounds_per_level: int = 14, plan_time: float = 8.0, explore: int = 0):
        self.memory, self.llm = memory, llm
        self.log = log or (lambda *a, **k: None)
        self.events_dir, self.max_seconds = events_dir, max_seconds
        self.budget_cfg, self.perception_cfg = budget_cfg, perception_cfg
        self.K, self.min_acc, self.max_rounds, self.plan_time = K, min_acc, max_rounds_per_level, plan_time
        self.explore = int(explore)     # level-1 exploration cap before the first hypothesis (0 = off; docs/029)
        self.sandbox = Sandbox(log=self.log, allow_getattr=True)
        self.load_errors: list[str] = []
        # the deterministic induction of wml/ and the goal template library run without an LLM: their models are
        # candidates next to the LLM's, verified by the same pixel verifier (docs/029 P1)
        self.wml = WorldModelLab(llm=None, memory=memory, log=self.log)
        self.goal_inf = GoalInference(memory, None, self.wml.sandbox, log=self.log)

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
        hints = list(counterexamples) + errs
        prev_code = prev.code if (prev is not None and prev.origin == "llm") else None
        if prev is not None and prev.origin == "induced" and prev.verdict is not None:
            hints.append(f"A deterministic model composed from the action semantics ('{prev.name}') already predicts {prev.verdict.accuracy:.0%} of this level's "
                         "transitions exactly; your hypothesis must be at least as exact, and must add the win condition and the candidate actions.")
        msgs = hypothesis_prompt(scene=s.scene, grid=grid, log=log, available=s.available_actions(), prior_signatures=mechanism_signatures(),
                                 previous_code=prev_code, counterexamples=hints, level=s.level,
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

    # ── deterministic candidates (docs/029 P1-3) ──
    def induced_candidates(self, s: Session, log: list[Transition], n: int, limit: int = 3, ids: Optional[dict] = None) -> list[InducedHypothesis]:
        """Prior-composed models induced from this level's log, each paired with the goal templates ranked for its roles.
        `ids` keeps one number per model name across rounds, so a no-plan verdict against a model sticks to it."""
        if not log:
            return []
        ids = ids if ids is not None else {}
        try:
            sem = classify_actions(log)
            available = s.available_actions()
            H = self.wml.refine(log, [], self.memory.priors("mechanisms"), scene=s.scene, semantics=sem, available=available, use_llm=False)
        except Exception as e:
            self.log(f"induction failed: {e!r}")
            return []
        out: list[InducedHypothesis] = []
        for wh in H[:limit]:
            roles_fn = wh.model.with_roles if isinstance(wh.model, RuleModel) else None
            try:
                G = self.goal_inf.refine(log, [], self.memory.priors("goals"), s.level, s.scene, roles_fn=roles_fn, model=wh.model)
            except Exception as e:
                self.log(f"goal inference failed: {e!r}"); G = []
            actions_fn = (lambda scene, sem=sem, av=available: action_set(scene, av, semantics=sem))
            out.append(InducedHypothesis(wh, G[:3], ids.setdefault(wh.name, 1000 + len(ids)), actions_fn))
        return out

    def goal_directed_step(self, h: Hypothesis, s: Session, observed: dict, tried_here: set, seen_states: set) -> Optional[Action]:
        """When a usable model yields no plan (docs/029 P1-4): the candidate action whose predicted outcome raises the goal's
        progress the most, or leads to a board not seen yet; unknown outcomes count as mildly informative."""
        cands = [a for a in h.call("candidate_actions", s.scene, []) if isinstance(a, Action)]
        for a in action_set(s.scene, s.available_actions()):
            if a not in cands:
                cands.append(a)
        sk = state_key(s.scene); cur = h.progress(s.scene)
        best: Optional[Action] = None; best_score = 0.0
        for a in cands[:48]:
            if (sk, a.label()) in tried_here:
                continue
            nxt = observed.get((sk, a.label()))
            if nxt is None:
                nxt = h.model.predict(s.scene, a)
            if nxt is None:
                score = 0.15
            else:
                nk = state_key(nxt)
                if nk == sk:
                    continue                      # predicted no-op
                score = (h.progress(nxt) - cur) + (0.3 if nk not in seen_states else 0.0)
            if score > best_score:
                best, best_score = a, score
        return best

    # ── planning on a verified hypothesis ──
    def plan(self, h: Hypothesis, s: Session, observed: dict) -> Optional[list[Action]]:
        """Shortest path under h's model to its goal; an induced hypothesis carries several template goals and the first
        one that yields a plan becomes its goal."""
        goals = [g for g in (getattr(h, "goals", None) or [h.goal]) if g is not None]
        if not goals:
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
        per_goal = max(3.0, self.plan_time / len(goals))
        for g in goals[:3]:
            h.goal = g
            try:
                if h.is_goal(s.scene):
                    return []
                plan = astar(s.scene, successors, h.is_goal, h.progress, depth=80, time_limit=per_goal)
            except Exception as e:
                self.log(f"planning failed: {e!r}"); plan = None
            if plan:
                return plan
        h.goal = goals[0]
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
        plans_executed = 0; tests_run = 0; tried_labels: set = set(); idle_rounds = 0; approx_budget = 6
        tried_here: set = set()             # (board state, action label) already run as a test: never repeat it from the same board
        gd_steps = 0                        # goal-directed steps taken on this level (P1-4)
        induced_ids: dict = {}              # induced model name -> stable hypothesis number
        llm_best: Optional[Hypothesis] = None   # best LLM candidate of the last round: its tests run when an induced model tops the ranking
        approx_actions = 0                  # actions spent under approximate plans on this level (P2-6)
        no_plan_n: Optional[int] = None     # hypothesis that verified but produced no plan (keep it only while nothing verifies better)
        repeats_dropped = 0

        def act(a: Action, kind: str) -> Optional[Transition]:
            t = s.act(a, kind)
            if t is not None and t.status_change is None and t.action.type != "RESET":
                observed[(state_key(t.before), a.label())] = t.after
            return t

        events.emit("START", "LOOK", "hypothesis policy: look, hypothesise, test, verify, revise, plan", budget_used=0)
        explored = None
        if self.explore > 0 and s.level == 1:
            # act first on level 1 (the cheapest level to spend actions on), hypothesise from what happened
            explored = explore_level(s, act, cap=self.explore, seed=zlib.crc32(game_id.encode()),
                                     emit=lambda m: events.emit("EXPLORE", "LOOK", m, budget_used=budget.used()))
        while not s.finished():
            if s.timed_out():
                stop = "timeout"; break
            if s.env.status().state == "GAME_OVER":
                events.emit("LOOK", "LOOK", "game over -> RESET (hypothesis kept)", budget_used=budget.used())
                s.act(Action.reset(), "reset"); continue
            if s.level != level:
                events.emit("LOOK", "LOOK", f"level {level} -> {s.level}: the hypothesis must verify again on the new board", budget_used=budget.used())
                level = s.level; rounds = 0; idle_rounds = 0; approx_budget = 6; counterexamples = []; observed.clear(); tried_here.clear(); no_plan_n = None; gd_steps = 0; approx_actions = 0
                if hyp is not None:
                    hyp.verdict = None
            log = s.level_log()
            # verify the current hypothesis on this level's log
            if hyp is not None and log:
                hyp.verdict = verify(hyp.model, log, hyp.ignore)
            exact = hyp is not None and hyp.verdict is not None and hyp.verdict.usable(self.min_acc)
            approx = (not exact) and hyp is not None and hyp.verdict is not None and hyp.verdict.approximate() and approx_budget > 0 and approx_actions < APPROX_ACTIONS_PER_LEVEL
            usable = exact or approx
            if usable:
                if approx:
                    approx_budget -= 1
                plan = self.plan(hyp, s, observed)
                if plan is None or (plan == [] and not hyp.is_goal(s.scene)):
                    seen_states = {state_key(t.before) for t in log} | {state_key(t.after) for t in log}
                    a = self.goal_directed_step(hyp, s, observed, tried_here, seen_states) if gd_steps < GD_STEPS_PER_LEVEL else None
                    if a is not None:
                        # act on the model instead of asking the LLM to rethink: one step that raises progress / reaches a new board
                        gd_steps += 1
                        pred = hyp.model.predict(s.scene, a)
                        t = act(a, "plan")
                        if t is None:
                            break
                        tried_here.add((state_key(t.before), a.label()))
                        events.emit("PLAN", "EXECUTE", f"no plan under h{hyp.n} -> goal-directed step {a.label()} ({gd_steps}/{GD_STEPS_PER_LEVEL})", transition_id=t.id, budget_used=budget.used())
                        if t.status_change:
                            events.emit("EXECUTE", "LOOK", f"{t.status_change} after {a.label()}", transition_id=t.id, budget_used=budget.used())
                        elif pred is not None:
                            ok, n, win = pixel_equal(pred, t.after_frame.grid, _boxes(t.before, hyp.ignore))
                            if not ok:
                                counterexamples = (counterexamples + verify(hyp.model, [t], hyp.ignore).counterexamples)[-4:]
                                events.emit("EXECUTE", "REVISE", f"mismatch after goal-directed {a.label()} ({n} px): counter-example recorded", transition_id=t.id, budget_used=budget.used())
                        continue
                    events.emit("PLAN", "TEST", f"no plan under h{hyp.n} for goal '{getattr(hyp.goal, 'name', '?')}' -> test the doubts", budget_used=budget.used())
                    counterexamples = (counterexamples + [f"Your model verified but the planner found NO action sequence from the current board to your goal '{getattr(hyp.goal, 'name', '?')}' using candidate_actions: the goal, the candidate actions or the dynamics are incomplete."])[-4:]
                    hyp.verdict = None; no_plan_n = hyp.n
                    usable = False
                elif plan == []:
                    events.emit("PLAN", "TEST", "goal already true but no level-up: the win condition is wrong", budget_used=budget.used())
                    counterexamples.append("The goal predicate you wrote is already TRUE on the current board, yet the level did not end: the win condition is different.")
                    rejected.append(f"win='{str(hyp.doc.get('win', ''))[:80]}'")
                    usable = False
                else:
                    mode = "exact" if exact else f"approximate (approx {hyp.verdict.approx:.2f}): first 2 steps, then re-think"
                    events.emit("PLAN", "EXECUTE", f"plan of {len(plan)} under h{hyp.n} [{mode}]: {[a.label() for a in plan[:10]]}", budget_used=budget.used())
                    plans_executed += 1
                    for a in (plan if exact else plan[:2]):
                        if not exact:
                            approx_actions += 1
                            if approx_actions == APPROX_ACTIONS_PER_LEVEL:
                                events.emit("EXECUTE", "REVISE", f"approximate model spent {APPROX_ACTIONS_PER_LEVEL} actions on this level without a level-up: exact verification required from now on", budget_used=budget.used())
                                counterexamples = (counterexamples + [f"Acting on your approximate model for {APPROX_ACTIONS_PER_LEVEL} actions did not end the level: the dynamics or the win condition are wrong in a way the closeness score hides."])[-4:]
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
            # the current hypothesis' accuracy even when its verdict was cleared (no plan): a worse candidate must not replace it
            cur_key = verify(hyp.model, log, hyp.ignore).rank_key() if (hyp is not None and log) else (-1.0,)
            t0 = time.time()
            cands = self.hypothesise(s, log, hyp, counterexamples, n_hyp, rejected)
            cands += self.induced_candidates(s, log, n_hyp, ids=induced_ids)
            for c in cands:
                c.verdict = verify(c.model, log, c.ignore) if log else Verdict(0.0, 0.0, 0, 0, 0)
                if c.timeout_note():
                    c.verdict.counterexamples = [c.timeout_note()] + c.verdict.counterexamples
                    events.emit("VERIFY", "VERIFY", f"h{c.n} hit the {MODEL_SECONDS:.0f}s deadline: {getattr(c.model, 'last_timeout', '')[:120]}", budget_used=budget.used())
            cands.sort(key=lambda c: c.verdict.rank_key(), reverse=True)
            llm_best = next((c for c in cands if c.origin == "llm"), llm_best)
            events.emit("HYPOTHESISE", "HYPOTHESISE", f"round {rounds}: {len(cands)} candidate(s) in {time.time() - t0:.0f}s", budget_used=budget.used())
            for c in cands:
                events.emit("HYPOTHESISE", "HYPOTHESISE", c.summary(), budget_used=budget.used())
                if self.events_dir:
                    hd = self.events_dir / f"{game_id}.hypotheses"; hd.mkdir(exist_ok=True)
                    (hd / f"h{c.n}{'i' if c.origin == 'induced' else ''}_r{rounds}_acc{c.verdict.accuracy:.2f}.py").write_text(c.code + "\n\n# verdict: " + c.summary()
                                                                                + "\n" + "\n".join("# " + l for ce in c.verdict.counterexamples[:2] for l in ce.splitlines()))
            if not cands:
                continue
            best = cands[0]
            if hyp is None or best.verdict.rank_key() >= cur_key or (no_plan_n == hyp.n and best.verdict.usable(self.min_acc) and best.n != no_plan_n):
                hyp = best
            else:
                events.emit("HYPOTHESISE", "HYPOTHESISE", f"kept h{hyp.n} (chg/acc {cur_key[0]:.2f}/{cur_key[1]:.2f}): best new candidate h{best.n} is worse ({best.verdict.change_accuracy:.2f}/{best.verdict.accuracy:.2f})", budget_used=budget.used())
                hyp.verdict = verify(hyp.model, log, hyp.ignore)
            if hyp.verdict.counterexamples:
                counterexamples = hyp.verdict.counterexamples[-4:]
            if hyp.verdict.usable(self.min_acc) and hyp.n != no_plan_n:
                events.emit("VERIFY", "PLAN", f"h{hyp.n} verified (acc {hyp.verdict.accuracy:.2f} on {hyp.verdict.n}) -> plan", budget_used=budget.used())
                idle_rounds = idle_rounds + 1 if budget.used() == used_before else 0
                continue
            src = hyp if (hyp.origin == "llm" or llm_best is None) else llm_best     # an induced model names no experiments: the LLM's run
            tests = [a for a in src.call("test_actions", s.scene, []) if isinstance(a, Action)][:2]
            attempts = [a for a in src.call("attempt_actions", s.scene, []) if isinstance(a, Action) and a not in tests][:1]
            tests = tests + attempts          # learn AND try: one move toward the hypothesised win per round
            sk = state_key(s.scene)
            fresh = [a for a in tests if (sk, a.label()) not in tried_here]
            if len(fresh) < len(tests):
                repeats_dropped += len(tests) - len(fresh)
                dropped = [a.label() for a in tests if a not in fresh]
                events.emit("TEST", "TEST", f"dropped tests already run from this board: {dropped}", budget_used=budget.used())
                counterexamples = (counterexamples + [f"You proposed {dropped} as tests, but they were already run from this exact board and their outcome is in the log: propose DIFFERENT actions."])[-4:]
            tests = fresh
            if not tests:
                tests = [a for a in hyp.call("candidate_actions", s.scene, []) if isinstance(a, Action) and (sk, a.label()) not in tried_here][:2]
            if not tests:
                tests = _fallback_probe(s, tried_labels)
                events.emit("TEST", "TEST", f"h{hyp.n} names no test -> fallback probe {[a.label() for a in tests]}", budget_used=budget.used())
            if not tests:
                events.emit("TEST", "HYPOTHESISE", f"h{hyp.n} names no test and nothing is left to probe: revise", budget_used=budget.used())
                continue
            events.emit("VERIFY", "TEST", f"h{hyp.n} not verified (acc {hyp.verdict.accuracy:.2f}, n {hyp.verdict.n}) -> tests+attempt {[a.label() for a in tests]}", budget_used=budget.used())
            for a in tests:
                t = act(a, "test")
                if t is None:
                    break
                tests_run += 1; tried_labels.add(a.label()); tried_here.add((state_key(t.before), a.label()))
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
                        "verified": bool(hyp.verdict and hyp.verdict.usable(self.min_acc)), "origin": hyp.origin}] if hyp else [],
                      [{"name": getattr(hyp.goal, "name", "?"), "win": hyp.doc.get("win", "")}] if hyp and hyp.goal else [],
                      {"rounds": n_hyp, "plans": plans_executed, "tests": tests_run, "repeats_dropped": repeats_dropped, "gd_steps": gd_steps, "explore": explored}, round(time.time() - t_start, 1))


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
