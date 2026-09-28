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

    def refine(self, log: list[Transition], current: list[GoalInstance], priors: Optional[dict], level: int, scene: Scene,
               *, roles_fn=None, ctx: Optional[dict] = None, model=None) -> list[GoalInstance]:
        usage = (priors or {}).get("usage_stats") if priors else None
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
        for name, pen in self.demoted.items():
            for g in G:
                if g.name == name:
                    g.confidence = max(0.0, g.confidence + pen)
        if not G and self.llm is not None:
            G = self.propose_with_llm(s, log)
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

    def propose_with_llm(self, scene: Scene, log: list[Transition]) -> list[GoalInstance]:
        from ..wml.prompts import goal_template_prompt
        from ..perception.summarize import scene_text
        if self.llm is None or self.sandbox is None:
            return []
        msgs = goal_template_prompt(scene_text(scene), [self._summ(t) for t in log[-20:]])
        try:
            reply = self.llm.chat(msgs, purpose="goal_template", image=None)
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
