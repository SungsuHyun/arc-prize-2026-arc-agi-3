"""docs/029 priority 1-2: plan() tries every untried goal of a board once and skips dead goals; a goal that held without
a level-up is demoted for the game; a level-up credits the goal that became true (LevelKnowledge.goal_wins)."""
from pathlib import Path

from pbg.core.types import Action
from pbg.goal.infer import GoalInference
from pbg.goal.templates import GoalInstance
from pbg.hypothesis.policy import HypothesisPolicy, _demote_goal, _record_level_win
from pbg.memory.store import Memory
from pbg.orchestrator.knowledge import LevelKnowledge
from pbg.perception import Perception
from pbg.tests.unit.synthetic import frame
from pbg.tests.unit.test_hypothesis_induced import _Session, _policy


def _goal(name, is_goal=lambda s: False):
    return GoalInstance(name, name.split("(")[0], {}, is_goal, lambda s: 0.0)


class _NoopModel:
    def predict(self, sc, a): return sc
    def with_roles(self, sc): return sc


class _H:
    n = 5; origin = "induced"; model = _NoopModel(); goal = None; goals = None; _goals_level = None; _goals_version = 0
    def call(self, name, sc, default): return default
    def roled(self, sc): return sc
    def is_goal(self, sc): return bool(self.goal.is_goal(sc)) if self.goal else False
    def progress(self, sc): return 0.0


def test_plan_tries_each_goal_once_per_board_and_skips_dead(tmp_path):
    p = Perception(); s = _Session(p); pol = _policy(tmp_path); pol.plan_time = 0.2
    h = _H(); h.goals = [_goal("a(x)"), _goal("b(x)"), _goal("c(x)")]; h._goals_level = s.level
    tried, dead = set(), {"b(x)"}
    assert pol.plan(h, s, {}, s.level_log(), tried, dead) is None      # unreachable under a no-op model
    assert {t[1] for t in tried} == {"a(x)", "c(x)"}                    # b skipped, a and c tried once
    n = len(tried)
    assert pol.plan(h, s, {}, s.level_log(), tried, dead) is None and len(tried) == n   # nothing left for this board: no re-planning


def test_plan_returns_empty_for_a_goal_already_true(tmp_path):
    p = Perception(); s = _Session(p); pol = _policy(tmp_path)
    h = _H(); h.goals = [_goal("done(x)", lambda sc: True)]; h._goals_level = s.level
    assert pol.plan(h, s, {}, s.level_log(), set(), set()) == []


def test_demote_goal_updates_inference_and_knowledge(tmp_path):
    gi = GoalInference(Memory(tmp_path / "m"), None, None); kn = LevelKnowledge()
    g = _goal("reach(agent,target)")
    _demote_goal(gi, kn, g)
    assert gi.demoted["reach(agent,target)"] < 0 and kn.goal_fails["reach"] == 1


def test_level_win_credits_the_goal_that_became_true():
    p = Perception(); s = _Session(p); kn = LevelKnowledge()
    last = s.level_log()[-1]
    agent = lambda sc: [o for o in sc.objects if o.color == 12][0].bbox
    after_key = agent(last.after)
    assert agent(last.before) != after_key
    became_true = _goal("reach(a,b)", lambda sc: agent(sc) == after_key)   # false before, true after the last move
    always = _goal("all_removed(x)", lambda sc: True)
    h = _H(); h.goals = [always, became_true]
    name = _record_level_win(kn, h, s.level_log(), 1, 3, None)
    assert name == "reach(a,b)" and kn.goal_wins["reach"] == 1 and kn.won_goals[0]["level"] == 1
    # with no goal that flipped, the plan's goal is credited
    kn2 = LevelKnowledge(); h2 = _H(); h2.goals = [_goal("x(y)")]
    assert _record_level_win(kn2, h2, s.level_log(), 1, 3, _goal("fallback(q)")) == "fallback(q)"
