"""orchestrator/loop.py — the control loop (spec §12): a state machine PROBE -> HYPOTHESIZE -> PLAN -> EXECUTE driven
only by events (mismatch, level-up, stuck, no plan). LLM calls happen only in HYPOTHESIZE. Every transition is logged
to events.jsonl with a reason; budget caps come from budget.yaml."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.budget import Budget
from ..core.contracts import Hypothesis, RuleModel
from ..core.events import EventLog
from ..core.types import Action, Scene, Transition
from ..goal.infer import GoalInference
from ..goal.templates import GoalInstance
from ..memory.store import Memory
from ..perception import Perception
from ..planner.execute import Planner
from ..probe.protocol import Probe
from ..probe.semantics import ActionSemantics, classify_actions
from ..wml.evaluate import evaluate
from ..wml.refine import WorldModelLab
from .session import Session
from .knowledge import LevelKnowledge
from .transfer import level_novelty, structure_vector, transfer_hypotheses


@dataclass
class Result:
    game_id: str
    state: str
    levels_completed: int
    levels_total: Optional[int]
    actions: int
    budget: dict
    llm_calls: int
    events: int
    stop_reason: str
    level_actions: dict = field(default_factory=dict)
    hypotheses: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    semantics: dict = field(default_factory=dict)
    elapsed: float = 0.0

    def to_json(self) -> dict:
        return self.__dict__.copy()


class Orchestrator:
    def __init__(self, *, memory: Memory, llm=None, budget_cfg: Optional[dict] = None, perception_cfg: Optional[dict] = None, log=None,
                 events_dir: Optional[Path] = None, use_llm: bool = True, max_seconds: Optional[float] = None, planner_kwargs: Optional[dict] = None,
                 max_levels: int = 20, max_resets: int = 6, min_plan_score: float = 0.6):
        self.memory, self.llm = memory, llm
        self.max_resets = max_resets
        self.min_plan_score = min_plan_score
        self.budget_cfg, self.perception_cfg = budget_cfg, perception_cfg
        self.log = log or (lambda *a, **k: None)
        self.events_dir = events_dir
        self.use_llm = use_llm and llm is not None
        self.max_seconds = max_seconds
        self.planner_kwargs = planner_kwargs or {}
        self.max_levels = max_levels

    def play(self, env, game_id: str, *, budget_total: Optional[int] = None) -> Result:
        t_start = time.time()
        budget = Budget(budget_total, self.budget_cfg)
        perception = Perception(self.perception_cfg)
        events = EventLog((self.events_dir / f"{game_id}.events.jsonl") if self.events_dir else None)
        s = Session(env, game_id, perception, self.memory, budget, log=self.log, max_seconds=self.max_seconds)
        mem = self.memory.load(game_id)
        wml = WorldModelLab(llm=self.llm if self.use_llm else None, memory=self.memory, log=self.log)
        goal_inf = GoalInference(self.memory, self.llm if self.use_llm else None, wml.sandbox, log=self.log)
        # knowledge asset of this game: click response map, goal template wins/fails, demoted goals; accumulates across
        # levels (and runs) so nothing a level taught is re-learned with actions
        knowledge = LevelKnowledge.load(self.memory.knowledge_path(game_id))
        goal_inf.knowledge = knowledge; goal_inf.demoted.update(knowledge.demoted)
        self.log(f"knowledge: {knowledge.summary()}")
        planner = Planner(log=self.log, **self.planner_kwargs)
        planner.click_map = knowledge.clicks
        probe = Probe(s)
        scene = s.start()
        env.calibrate_click()
        semantics = ActionSemantics(mem.action_semantics or {})
        H: list[Hypothesis] = []; G: list[GoalInstance] = []
        # transfer: verified model of this game (retry) or of similar games (spec §13)
        H += transfer_hypotheses(self.memory, game_id, scene, wml.sandbox, log=self.log)
        self.memory.save_structure(game_id, structure_vector(scene))
        state = "PLAN" if (H and mem.world_model_code) else "PROBE"
        events.emit("START", state, "verified model in memory" if state == "PLAN" else "no model", budget_used=0)
        stop = ""; experiments_this_round = 0; hypothesize_rounds = 0; no_plan_rounds = 0; reprobe_rounds = 0
        last_level = s.level; level_start_step = 0; last_mismatch: Optional[Transition] = None
        prev_level_scene: Optional[Scene] = scene
        plan: Optional[list[Action]] = None
        touched: set = set(); exploring = False
        last_refine_n = -1; resets_without_progress = 0; game_overs = 0
        tried_experiments: set = set(); bumped: set = set(); clicked: set = set(); gated_rounds = 0
        idle_iters = 0; last_used = -1
        walk_dry = 0; last_action = None; fresh_level = False; llm_waits_this_level = 0
        while not s.finished():
            knowledge.clicks.extend(s.transitions[-8:])
            if budget.used() == last_used:
                idle_iters += 1
                if idle_iters >= 8 and state in ("PLAN", "HYPOTHESIZE") and not wml.job_running():
                    events.emit(state, "PROBE", "no action for 8 loop iterations -> walk probe (liveness)", budget_used=budget.used())
                    state = "PROBE"; idle_iters = 0
            else:
                idle_iters = 0; last_used = budget.used()
            if s.env.status().state == "GAME_OVER":
                game_overs += 1
                if game_overs > 20:
                    stop = "GAME_OVER"; break
                events.emit(state, "PLAN", f"game over #{game_overs} (during {state}) -> RESET", budget_used=budget.used())
                s.act(Action.reset(), "plan"); planner.stuck.reset(); touched.clear(); state = "PLAN"
                continue
            if s.level < last_level:
                # the game restarted from level 1 (engine semantics of a RESET on a fresh level): start the level bookkeeping over
                events.emit(state, "PLAN", f"game restarted at level {s.level} (was {last_level}); level bookkeeping reset", budget_used=budget.used())
                last_level = s.level; prev_level_scene = s.scene; planner.stuck.reset(); touched.clear(); bumped.clear(); tried_experiments.clear()
                no_plan_rounds = 0; reprobe_rounds = 0; resets_without_progress = 0; state = "PLAN"; continue
            if s.level > last_level:
                # level transition procedure (spec §13)
                novelty, novel_ids = level_novelty(prev_level_scene, s.scene)
                self.memory.record_level_note(game_id, last_level, {"level": last_level, "actions_used": s.level_actions.get(last_level, 0),
                                                                     "novelty": novelty, "replaced_rules": [], "added_rules": []})
                events.emit(state, "PLAN" if novelty == 0 else "PROBE", f"level {last_level} -> {s.level}, novelty {novelty}", budget_used=budget.used())
                fresh_level = novelty > 0; knowledge.hyp_triggers = set(); knowledge.hyp_inert = set(); knowledge.clicks.skip_untried = set(); llm_waits_this_level = 0
                last_level = s.level; prev_level_scene = s.scene; level_start_step = s.step_idx; walk_dry = 0
                knowledge.levels_seen.add(s.level); knowledge.demoted = dict(goal_inf.demoted); knowledge.save(self.memory.knowledge_path(game_id))
                planner.stuck.reset(); no_plan_rounds = 0; reprobe_rounds = 0; resets_without_progress = 0; touched.clear(); clicked.clear()
                if s.level > self.max_levels:
                    stop = "max_levels"; break
                if novelty == 0 and H:
                    state = "PLAN"
                else:
                    state = "LOCAL_PROBE"
                    self._novel_ids = novel_ids
            if state == "PROBE":
                cap = budget.cap("initial_probe") if not s.transitions else budget.cap("reprobe", s.level)
                kind = "initial" if not s.transitions else "reprobe"
                if kind == "reprobe":
                    reprobe_rounds += 1
                    if walk_dry >= 2:
                        cap = 0     # the click map has nothing untried or responsive left: do not walk, wait/reset instead
                    if cap <= 0 and wml.job_running():
                        # nothing cheap left to try: wait for the pending LLM candidates rather than spend actions (spec §15)
                        events.emit(state, "HYPOTHESIZE", "reprobe budget exhausted, waiting for the llm job", budget_used=budget.used())
                        wml.wait_job(min(240.0, max(0.0, (s.deadline - time.time()) if s.deadline else 240.0)))
                        state = "HYPOTHESIZE"; continue
                    if cap > 0 and not _walk_useful(knowledge, s) and H:
                        # nothing new to click: one goal-directed step (the responsive action whose predicted outcome raises the
                        # top goal's progress the most, never the inverse of the last action) instead of a round-robin walk
                        a = _goal_directed_action(knowledge, s, H, G, last_action)
                        if a is not None:
                            t_ = s.act(a, "reprobe")
                            if t_ is not None:
                                last_action = a
                                events.emit("PROBE", "HYPOTHESIZE", f"goal-directed probe {a.label()}", transition_id=t_.id, budget_used=budget.used())
                                semantics = classify_actions(s.transitions, semantics); knowledge.clicks.extend(s.transitions[-3:])
                                state = "HYPOTHESIZE"; experiments_this_round = 0; continue
                        walk_dry += 1; cap = 0
                    if cap <= 0:
                        # re-exploration budget exhausted: reset as the last resort (spec §15); the reprobe allowance
                        # starts again after the reset (a new attempt), and the game is only abandoned after 3 resets
                        # without a level-up
                        if s.reset_allowed and resets_without_progress < int(self.max_resets):
                            if s.actions_since_reset == 0:
                                # nothing happened since the last reset: a second RESET would restart the whole game.
                                # Grant a fresh re-exploration allowance and walk (act) instead of spinning.
                                events.emit(state, "PROBE", "reset refused (no action since last reset) -> fresh walk allowance", budget_used=budget.used())
                                budget.reset_level(s.level, "reprobe")
                                if wml.job_running():
                                    wml.wait_job(60.0)
                                cap = budget.cap("reprobe", s.level)
                            events.emit(state, "HYPOTHESIZE", "reprobe budget exhausted -> RESET", budget_used=budget.used())
                            acted = s.actions_since_reset > 0
                            s.act(Action.reset(), "reprobe"); planner.stuck.reset()
                            if acted:
                                resets_without_progress += 1     # a refused reset is not an attempt
                            budget.reset_level(s.level, "reprobe"); touched.clear(); walk_dry = 0; state = "HYPOTHESIZE"; continue
                        stop = "UNRESOLVED"; events.emit(state, "END", f"no progress after {self.max_resets} resets", budget_used=budget.used()); break
                if kind == "initial":
                    res = probe.run_initial(cap, do_reset_probe=False)
                elif fresh_level and any(a.type == "CLICK" for a in s.available_actions()):
                    # new board: hypothesise roles/relations from the frame and test them with one click per trigger candidate
                    fresh_level = False
                    res = probe.run_hypothesis_probe(min(cap, 12))
                else:
                    agent_ids = set()
                    if H and isinstance(H[0].model, RuleModel):
                        agent_ids = {o.id for o in H[0].model.with_roles(s.scene).objects if o.role and "agent" in o.role}
                    dirs = {int(k[6:]): (int(v["displacement"][0]) and (v["displacement"][0] > 0) - (v["displacement"][0] < 0), (v["displacement"][1] > 0) - (v["displacement"][1] < 0))
                            for k, v in semantics.items() if k.startswith("ACTION") and isinstance(v, dict) and v.get("class") == "MOVE" and v.get("displacement")}
                    step = max([abs(v["displacement"][0]) + abs(v["displacement"][1]) for k, v in semantics.items() if k.startswith("ACTION") and isinstance(v, dict) and v.get("displacement")] or [1])
                    dirs = {k: (d[0] * step, d[1] * step) for k, d in dirs.items()}
                    res = probe.run_walk(cap, agent_ids=agent_ids, dirs=dirs, bumped=bumped, clicked=clicked, click_map=knowledge.clicks)
                    walk_dry = walk_dry + 1 if res.actions_used == 0 else 0
                semantics = classify_actions(s.transitions, semantics)
                knowledge.clicks.extend(res.transitions); knowledge.save(self.memory.knowledge_path(game_id))
                if getattr(res, "hypotheses", None) is not None:
                    knowledge.apply_hypotheses(res)
                    events.emit("PROBE", "PROBE", f"hypotheses: {res.hypotheses.summary()}", budget_used=budget.used())
                    for v in (res.verdicts or [])[:12]:
                        events.emit("PROBE", "PROBE", f"verdict: {v}", budget_used=budget.used())
                self.memory.save_semantics(game_id, semantics.to_json())
                events.emit("PROBE", "HYPOTHESIZE", f"{kind} probe done: {res.actions_used} actions, steps {res.steps_done}", budget_used=budget.used())
                state = "HYPOTHESIZE"; experiments_this_round = 0
            elif state == "LOCAL_PROBE":
                cap = budget.cap("level_probe", s.level)
                res = probe.run_local(cap, novel_ids=getattr(self, "_novel_ids", []), mismatch=last_mismatch, navigate=None)
                semantics = classify_actions(s.transitions, semantics)
                events.emit("LOCAL_PROBE", "HYPOTHESIZE", f"local probe: {res.actions_used} actions", budget_used=budget.used())
                state = "HYPOTHESIZE"; experiments_this_round = 0
            elif state == "HYPOTHESIZE":
                hypothesize_rounds += 1
                for t_ in s.transitions[len(planner.observed_seen):]:
                    planner.observed_seen.append(t_.id)
                    if t_.status_change is None and t_.action.type != "RESET":
                        planner.observed[(_state_key(t_.before), t_.action.label())] = t_.after
                level_note = self._level_note(mem, s.level)
                allow_llm = self.use_llm and budget.allows_llm() and not budget.low()
                if len(s.transitions) != last_refine_n or (wml._job is not None and not wml.job_running()):
                    semantics = classify_actions(s.transitions, semantics)
                    H = wml.refine(s.transitions, H, self.memory.priors("mechanisms"), scene=s.scene, semantics=semantics, available=s.available_actions(),
                                   level_note=level_note, use_llm=False, knowledge=knowledge)
                    last_refine_n = len(s.transitions)
                    best = H[0] if H else None
                    if allow_llm and (best is None or best.score < 1.0 or best.coverage < 1.0) and not wml.job_running():
                        if wml.start_llm_job(s.transitions, best, self.memory.priors("mechanisms"), scene=s.scene, semantics=semantics, level_note=level_note,
                                             K=(2 if budget.fraction_left() < 0.5 else 4)):
                            events.emit("HYPOTHESIZE", "HYPOTHESIZE", f"llm job started in background (round {wml.async_rounds}, log {len(s.transitions)})", budget_used=budget.used())
                budget.llm_calls = wml.llm_calls + goal_inf.__dict__.get("llm_calls", 0)
                roles_fn = (lambda sc, m=H[0].model: m.with_roles(sc)) if H and isinstance(H[0].model, RuleModel) else None
                G = goal_inf.refine(s.transitions, G, self.memory.priors("goals"), s.level, s.scene, roles_fn=roles_fn, model=H[0].model if H else None)
                exp = None
                model_ok = bool(H) and H[0].usable(self.min_plan_score)
                if model_ok and not budget.low() and budget.allows("experiment", level=s.level) and experiments_this_round < 6:
                    exp = wml.most_informative_action(H, s.scene, s.available_actions(), semantics=semantics, extra_clicks=planner.responsive, click_map=knowledge.clicks,
                                                      exclude=tried_experiments, state_key=_state_key(s.scene))   # never repeat an experiment from the same state
                if exp is not None:
                    tried_experiments.add((_state_key(s.scene), exp.label()))
                    t = s.act(exp, "experiment"); experiments_this_round += 1
                    events.emit("HYPOTHESIZE", "HYPOTHESIZE", f"experiment {exp.label()} (gain over {len(H)} hypotheses)", transition_id=t.id if t else None, budget_used=budget.used())
                    continue
                if H and H[0].verified and H[0].code and H[0].origin != "induced":
                    self.memory.promote_model(game_id, H[0].code, {"score": H[0].score, "coverage": H[0].coverage, "n": len(s.transitions)})
                self.memory.save_hypotheses(game_id, [{"name": h.name, "score": h.score, "coverage": h.coverage, "code": h.code, "origin": h.origin} for h in H])
                top = f"top h={H[0].name} score={H[0].score:.2f} chg={H[0].change_score:.2f} cov={H[0].coverage:.2f}" if H else "no hypothesis"
                events.emit("HYPOTHESIZE", "PLAN", f"{top}; goals={len(G)} top={G[0].name if G else None}", budget_used=budget.used())
                state = "PLAN"
            elif state == "PLAN":
                if wml._job is not None and not wml.job_running():
                    events.emit("PLAN", "HYPOTHESIZE", "llm job finished -> merge candidates", budget_used=budget.used()); state = "HYPOTHESIZE"; continue
                if H and not H[0].usable(self.min_plan_score) and not H[0].verified:
                    # quality gate: a model that explains < 60% of the log is not worth executing plans on; gather evidence instead
                    if wml.job_running() and budget.cap("reprobe", s.level) <= 0 and llm_waits_this_level < 1:
                        # one wait per level at most: after that the loop acts (goal-directed step) instead of idling
                        llm_waits_this_level += 1
                        events.emit("PLAN", "HYPOTHESIZE", f"model score {H[0].score:.2f}/change {H[0].change_score:.2f} below gate; waiting for the llm job", budget_used=budget.used())
                        wml.wait_job(120.0); state = "HYPOTHESIZE"; continue
                    if budget.cap("reprobe", s.level) > 0:
                        # walk while the click map can still learn; otherwise PROBE takes one goal-directed step (an action
                        # whose predicted outcome raises the top goal's progress) -- idling on the LLM is the last resort
                        why = "walk probe" if _walk_useful(knowledge, s) else "goal-directed step"
                        events.emit("PLAN", "PROBE", f"model score {H[0].score:.2f}/change {H[0].change_score:.2f} below gate -> {why}", budget_used=budget.used()); state = "PROBE"; continue
                    a = _goal_directed_action(knowledge, s, H, G, last_action) if H else None
                    if a is not None:
                        t_ = s.act(a, "reprobe")
                        if t_ is not None:
                            last_action = a
                            events.emit("PLAN", "HYPOTHESIZE", f"below gate, reprobe spent -> goal-directed step {a.label()}", transition_id=t_.id, budget_used=budget.used())
                            semantics = classify_actions(s.transitions, semantics); knowledge.clicks.extend(s.transitions[-3:])
                            state = "HYPOTHESIZE"; experiments_this_round = 0; continue
                    if wml.job_running():
                        llm_waits_this_level += 1
                        events.emit("PLAN", "HYPOTHESIZE", "below gate, reprobe spent -> waiting for the llm job", budget_used=budget.used())
                        wml.wait_job(120.0); state = "HYPOTHESIZE"; continue
                    if not self.use_llm or not wml.job_running():
                        # nothing left to learn from cheaply: ask the LLM for a goal/model or let the low-quality plan through once
                        gated_rounds += 1
                        if gated_rounds > 3:
                            events.emit("PLAN", "PLAN", "gate lifted after 3 rounds without new evidence", budget_used=budget.used())
                        else:
                            events.emit("PLAN", "HYPOTHESIZE", "below gate, reprobe spent, no llm job -> hypothesize", budget_used=budget.used()); state = "HYPOTHESIZE"; continue
                if not H or not G:
                    if not H and s.transitions:
                        # nothing explains the log: fall back to probing with untried actions / clicks
                        state = "PROBE" if budget.cap("reprobe", s.level) > 0 else "HYPOTHESIZE"
                        events.emit("PLAN", state, "no hypothesis" if not H else "no goal", budget_used=budget.used())
                        if state == "HYPOTHESIZE" and hypothesize_rounds > 3:
                            stop = "no_hypothesis"; break
                        continue
                    if not G:
                        G = goal_inf.refine(s.transitions, G, self.memory.priors("goals"), s.level, s.scene,
                                            roles_fn=(lambda sc, m=H[0].model: m.with_roles(sc)) if H and isinstance(H[0].model, RuleModel) else None)
                        if not G:
                            no_plan_rounds += 1
                            state = "PROBE" if no_plan_rounds > 1 else "HYPOTHESIZE"
                            events.emit("PLAN", state, "no goal candidates", budget_used=budget.used())
                            continue
                plan = planner.search(s.scene, H, G, available=s.available_actions(), semantics=semantics, visited=planner.stuck.visited)
                if plan is None and H and not H[0].verified:
                    # exploration fallback (spec §10 'model unverified -> experiment'): reach an untouched object to gather evidence
                    roles_fn = (lambda sc, m=H[0].model: m.with_roles(sc)) if isinstance(H[0].model, RuleModel) else (lambda sc: sc)
                    from ..goal.templates import t_explore
                    ex = t_explore(roles_fn(s.scene), {"touched": touched})
                    if ex:
                        plan = planner.search(s.scene, H[:1], ex, available=s.available_actions(), semantics=semantics, visited=planner.stuck.visited)
                        if plan:
                            events.emit("PLAN", "EXECUTE", f"explore plan of {len(plan)} toward an untouched object", budget_used=budget.used())
                            state = "EXECUTE"; exploring = True
                            continue
                if plan is None:
                    no_plan_rounds += 1
                    if H and H[0].verified and not getattr(H[0].model, "level_scoped", False):
                        # (a verified permutation table only knows the positions it saw: no plan there says nothing about the goal)
                        G = goal_inf.demote(G, 0)
                        reason = "no plan under verified model -> goal demoted"
                    elif no_plan_rounds % 2 == 0:
                        planner.depth = min(240, planner.depth * 2); reason = f"no plan -> depth {planner.depth}"
                    else:
                        reason = "no plan under unverified model -> experiment"
                    events.emit("PLAN", "HYPOTHESIZE", reason, budget_used=budget.used())
                    state = "HYPOTHESIZE"
                    if no_plan_rounds >= 6:
                        if wml.job_running() and budget.cap("reprobe", s.level) <= 0:
                            events.emit("PLAN", "HYPOTHESIZE", "no plan, reprobe spent -> waiting for the llm job", budget_used=budget.used())
                            wml.wait_job(120.0); no_plan_rounds = 0; continue
                        if _walk_useful(knowledge, s):
                            events.emit("PLAN", "PROBE", "no plan after 6 rounds -> reprobe", budget_used=budget.used()); state = "PROBE"; no_plan_rounds = 0
                        else:
                            walk_dry += 1
                            events.emit("PLAN", "HYPOTHESIZE", "no plan after 6 rounds, nothing new to click -> hypothesize", budget_used=budget.used()); no_plan_rounds = 0
                    continue
                events.emit("PLAN", "EXECUTE", f"plan of {len(plan)}: {[a.label() for a in plan[:8]]}", budget_used=budget.used())
                state = "EXECUTE"; exploring = False
            elif state == "EXECUTE":
                r = planner.execute(plan or [], H, s, G)
                goal_inf.observe_progress(G, s.scene)
                _mark_touched(touched, s.scene, H)
                if exploring and r.kind == "EXHAUSTED":
                    exploring = False
                    events.emit("EXECUTE", "HYPOTHESIZE", f"explore plan done ({r.executed} actions)", budget_used=budget.used()); state = "HYPOTHESIZE"; continue
                if r.kind == "PROGRESS":
                    events.emit("EXECUTE", "PLAN", f"level up after {r.executed} actions", transition_id=r.t.id if r.t else None, budget_used=budget.used())
                    won = planner.last_info.goal if planner.last_info else None
                    knowledge.record_win(last_level, won, s.step_idx - level_start_step)
                    # harvest class-level rules from every prior-composed model that verified on this level (the plan may
                    # have used an LLM model whose rules carry no class description)
                    n_rules = sum(knowledge.record_rules(h.model) for h in H if h.origin == "induced" and isinstance(h.model, RuleModel) and h.score / max(h.coverage, 1e-9) >= 0.9)
                    self.log(f"knowledge: {n_rules} class-level click rule(s) harvested; asset now {list(knowledge.click_rules)}")
                    if won is not None:
                        self.memory.record_usage(won.template, used=True, verified=True)
                    G = goal_inf.refine(s.transitions, G, self.memory.priors("goals"), s.level, s.scene)
                    state = "PLAN"; last_mismatch = None
                elif r.kind == "MISMATCH":
                    last_mismatch = r.t
                    self._mark_violations(H, r.t)
                    events.emit("EXECUTE", "HYPOTHESIZE", f"prediction mismatch at step {r.executed}: {str(r.detail.get('mismatch'))[:120]}",
                                transition_id=r.t.id if r.t else None, budget_used=budget.used())
                    state = "HYPOTHESIZE"; experiments_this_round = 0
                elif r.kind == "STUCK":
                    events.emit("EXECUTE", "PROBE", f"stuck: {r.detail.get('reason')}", transition_id=r.t.id if r.t else None, budget_used=budget.used())
                    planner.stuck.reset(); state = "PROBE"
                elif r.kind == "EXHAUSTED":
                    reached = planner.last_info.goal if planner.last_info else None
                    if reached is not None and reached.template != "explore":
                        roles_fn = (lambda sc, m=H[0].model: m.with_roles(sc)) if H and isinstance(H[0].model, RuleModel) else (lambda sc: sc)
                        if reached.is_goal(roles_fn(s.scene)):
                            # the predicate holds and nothing happened: it is not the win condition (stronger than 'unreachable')
                            idx = next((i for i, g in enumerate(G) if g.name == reached.name), None)
                            if idx is not None:
                                goal_inf.demote(G, idx, penalty=-0.6)
                            knowledge.record_fail(reached)
                            events.emit("EXECUTE", "PLAN", f"plan exhausted ({r.executed} actions): goal {reached.name} reached without level-up -> demoted", budget_used=budget.used())
                            state = "PLAN"; continue
                    events.emit("EXECUTE", "PLAN", f"plan exhausted ({r.executed} actions)", budget_used=budget.used()); state = "PLAN"
                elif r.kind == "FAILED":
                    # GAME_OVER: the level restarts on RESET; retry from PLAN with memory kept (spec §12)
                    game_overs += 1
                    events.emit("EXECUTE", "PLAN", f"game over #{game_overs} -> RESET, retry from PLAN with memory kept", transition_id=r.t.id if r.t else None, budget_used=budget.used())
                    if game_overs > 20:
                        stop = "GAME_OVER"; break
                    s.act(Action.reset(), "plan"); planner.stuck.reset(); touched.clear()
                    state = "PLAN"
                else:
                    events.emit("EXECUTE", "PLAN", f"execution error: {r.detail}", budget_used=budget.used()); state = "PLAN"
                    if s.errors > 20:
                        stop = "env_errors"; break
        st = s.env.status()
        if not stop:
            stop = st.state if st.state in ("WIN", "GAME_OVER") else ("timeout" if s.timed_out() else "budget")
        events.emit(state, "END", stop, budget_used=budget.used())
        self.memory.save_semantics(game_id, semantics.to_json())
        knowledge.clicks.extend(s.transitions[-8:]); knowledge.demoted = dict(goal_inf.demoted); knowledge.save(self.memory.knowledge_path(game_id))
        self.memory.save()
        return Result(game_id, st.state, st.level - 1 if st.state != "WIN" else (st.levels_total or st.level), st.levels_total, st.actions_used, budget.snapshot(),
                      wml.llm_calls, len(events.events), stop, dict(s.level_actions),
                      [{"name": h.name, "score": round(h.score, 3), "coverage": round(h.coverage, 3), "verified": h.verified, "origin": h.origin} for h in H],
                      [g.to_json() for g in G[:5]], {k: v for k, v in semantics.items() if not k.startswith("_")}, round(time.time() - t_start, 1))

    # ── helpers ──
    @staticmethod
    def _mark_violations(H: list[Hypothesis], t: Optional[Transition]) -> None:
        if t is None:
            return
        for h in H:
            if t.id not in h.violations:
                h.violations.append(t.id)
            if isinstance(h.model, RuleModel):
                s = h.model.with_roles(t.before)
                for r in h.model.rules():
                    try:
                        if r.applies(s, t.action):
                            r.confidence = 0.0; r.violated_by.append(t.id)
                    except Exception:
                        pass

    @staticmethod
    def _level_note(mem, level: int) -> str:
        n = mem.level_notes.get(str(level - 1)) if mem.level_notes else None
        return f"previous level note: {n}" if n else ""


def _mark_touched(touched: set, scene: Scene, H: list[Hypothesis]) -> None:
    """Objects the agent overlaps or touches are 'touched' (for the exploration goal)."""
    if not H or not isinstance(H[0].model, RuleModel):
        return
    from ..goal.templates import _adjacent
    s = H[0].model.with_roles(scene)
    for a in [o for o in s.objects if o.role and "agent" in o.role]:
        for o in s.objects:
            if o.id != a.id and (a.overlaps(o) or _adjacent(a, o)):
                touched.add(o.identity())


def _state_key(scene: Scene) -> tuple:
    """State identity for experiment bookkeeping: objects outside ui strips (counters must not make states look new)."""
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    return tuple(sorted(o.identity() for o in scene.objects if o.region not in strips))


def _walk_useful(knowledge, s) -> bool:
    """A walk probe is worth its actions only when the click map still has an untried class on this scene, or a
    responsive class not yet re-checked on this level; button games always walk (buttons are cheap to re-press)."""
    if not any(a.type == "CLICK" for a in s.available_actions()) or any(a.type == "BUTTON" for a in s.available_actions()):
        return True
    from ..probe.clickmap import click_key
    cmap = knowledge.clicks
    tried_here = cmap.tried_level.get(s.level, set())
    # clicks per trigger OBJECT on this level (two buttons of one class are two triggers to learn)
    per_obj: dict = {}
    for t in s.level_log():
        if t.action.type == "CLICK":
            for o in t.before.objects:
                if o.bbox[0] <= t.action.row < o.bbox[2] and o.bbox[1] <= t.action.col < o.bbox[3]:
                    per_obj[tuple(o.bbox)] = per_obj.get(tuple(o.bbox), 0) + 1
    for a in cmap.rank(s.scene, level=s.level, include_inert=False, limit=64):
        key = click_key(s.scene, a.row, a.col)
        st = cmap.status(key)
        if st == "untried" or (st == "responsive" and key not in tried_here):
            return True
        if st == "responsive":
            box = next((tuple(o.bbox) for o in s.scene.objects if o.bbox[0] <= a.row < o.bbox[2] and o.bbox[1] <= a.col < o.bbox[3]), None)
            if box is not None and per_obj.get(box, 0) < 2:
                return True    # induction needs repeated observations of each reacting trigger (one click is not a rule)
    return False


def _goal_directed_action(knowledge, s, H, G, last_action):
    """The responsive action (click map) whose predicted next state has the highest top-goal progress; unknown
    outcomes count as informative (progress 0.5); the inverse of the last action (same objects moved back) is skipped."""
    from ..probe.clickmap import click_key
    if not G:
        return None
    goal = G[0]; model = H[0].model
    cands = [a for a in s.available_actions() if a.type == "BUTTON"] + knowledge.clicks.rank(s.scene, level=s.level, include_inert=False, limit=16)
    if not cands:
        return None
    scored = []
    for a in cands:
        if a.type == "CLICK" and knowledge.clicks.status(click_key(s.scene, a.row, a.col)) == "untried":
            scored.append((0.6, a)); continue
        try:
            nxt = model.predict(s.scene, a)
        except Exception:
            nxt = None
        if nxt is None:
            scored.append((0.5, a)); continue
        if nxt is s.scene or _state_key(nxt) == _state_key(s.scene):
            continue                       # predicted no-op: nothing to learn or gain
        if last_action is not None:
            try:
                back = model.predict(nxt, last_action)
            except Exception:
                back = None
            if back is not None and _state_key(back) == _state_key(s.scene) and a.label() != last_action.label():
                pass                       # a cancels last_action: allowed only if it raises progress (checked below)
        scored.append((goal.progress(nxt), a))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    if last_action is not None and best.label() != last_action.label():
        # never undo the previous step unless it is the only option
        try:
            nxt = model.predict(s.scene, best)
            if nxt is not None and last_action is not None:
                undo = model.predict(nxt, last_action)
                if undo is not None and _state_key(undo) == _state_key(s.scene) and len(scored) > 1:
                    best = scored[1][1]
        except Exception:
            pass
    return best
