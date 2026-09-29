"""goal/infer.py — Goal Inference (spec §9): template instantiation, level-up filtering, confidence, LLM template
proposals when no candidate exists."""
from __future__ import annotations

from typing import Optional

from ..core.types import Scene, Transition
from .filter import filter_by_levelup
from .templates import GoalInstance, instantiate_all


class GoalInference:
    def __init__(self, memory=None, llm=None, sandbox=None, log=None):
        self.memory, self.llm, self.sandbox = memory, llm, sandbox
        self.log = log or (lambda *a, **k: None)
        self.demoted: dict[str, float] = {}
        self.knowledge = None                     # LevelKnowledge of the game (set by the orchestrator)
        self.llm_calls = 0
        self._proposals: dict[int, int] = {}      # level -> LLM goal proposals made (procedural goals the templates cannot express)

    def refine(self, log: list[Transition], current: list[GoalInstance], priors: Optional[dict], level: int, scene: Scene,
               *, roles_fn=None, ctx: Optional[dict] = None, model=None) -> list[GoalInstance]:
        usage = (priors or {}).get("usage_stats") if priors else None
        if self.knowledge is not None:
            usage = self.knowledge.goal_stats(usage)      # what won earlier levels of THIS game comes first
        s = roles_fn(scene) if roles_fn else scene
        fresh = instantiate_all(s, ctx, usage)
        by_name = {g.name: g for g in fresh}
        # keep confidence / history of goals we already had
        for g in current:
            if g.name in by_name:
                by_name[g.name].confidence = max(by_name[g.name].confidence, g.confidence)
                by_name[g.name].progress_weight = g.progress_weight
            elif g.origin == "llm":
                by_name[g.name] = g
        G = list(by_name.values())
        if self.knowledge is not None:
            for g in G:
                g.confidence = min(1.0, g.confidence + self.knowledge.goal_structure_bonus(g))
        for name, pen in self.demoted.items():
            for g in G:
                if g.name == name:
                    g.confidence = max(0.0, g.confidence + pen)
        # no template goal, or every template goal was demoted (reached without a level-up): the win condition is
        # probably procedural (a sequence, a pattern rule) -> ask the LLM, at most twice per level
        exhausted = bool(G) and max(g.confidence for g in G) < 0.25 and not any(g.origin == "llm" for g in G)
        if (not G or exhausted) and self.llm is not None and self._proposals.get(level, 0) < 2:
            self._proposals[level] = self._proposals.get(level, 0) + 1
            rejected = [g.name for g in G if self.demoted.get(g.name, 0.0) < 0]
            G = G + self.propose_with_llm(s, log, rejected=rejected)
        before = list(G)
        G = filter_by_levelup(G, log, roles_fn=roles_fn, model=model)
        if not G and before:
            # the level-up filter removed every candidate (true goal not in the library yet): keep them at half confidence
            for g in before:
                g.confidence *= 0.5
            G = before
        G.sort(key=lambda g: (-g.confidence, -g.progress(s), g.name))
        return G

    def demote(self, G: list[GoalInstance], idx: int = 0, penalty: float = -0.4) -> list[GoalInstance]:
        """Planner found no plan under a verified model: the top goal is probably wrong (spec §10)."""
        if not G:
            return G
        g = G[idx]
        g.confidence = max(0.0, g.confidence + penalty)
        self.demoted[g.name] = self.demoted.get(g.name, 0.0) + penalty
        G.sort(key=lambda x: -x.confidence)
        return G

    def observe_progress(self, G: list[GoalInstance], scene: Scene) -> None:
        """Track monotonicity of the top goal's progress (non-monotone -> weight 0.5, spec §9)."""
        for g in G[:2]:
            p = g.progress(scene)
            g.history.append(p)
            g.history = g.history[-15:]
            if len(g.history) >= 6 and any(g.history[i] < g.history[i - 1] - 1e-9 for i in range(1, len(g.history))):
                g.progress_weight = 0.5

    def propose_with_llm(self, scene: Scene, log: list[Transition], rejected: Optional[list[str]] = None) -> list[GoalInstance]:
        from ..wml.prompts import goal_template_prompt
        from ..wml.context import observation_lines, select_context
        from ..perception.summarize import scene_text
        if self.llm is None or self.sandbox is None:
            return []
        ctx = select_context(log, [], cap=30)
        msgs = goal_template_prompt(scene_text(scene), observation_lines(ctx), rejected=rejected or [])
        try:
            reply = self.llm.chat(msgs, purpose="goal_template", image=None)
            self.llm_calls += 1
        except Exception as e:
            self.log(f"goal llm failed: {e!r}")
            return []
        code = self.llm.extract_code(reply)
        if not code:
            return []
        goal = self.sandbox.load_goal(code)
        if goal is None:
            return []
        goal.origin = "llm"; goal.code = code; goal.confidence = 0.3      # a proposal, ranked below template goals until the level-up filter confirms it
        if goal.is_goal(scene):
            return []
        p = goal.progress(scene)
        if not (0.0 <= p < 1.0):
            return []
        return [goal]

    @staticmethod
    def _summ(t: Transition) -> str:
        from ..perception.summarize import summarize
        return summarize(t)
