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


def test_contact_exploration_walks_onto_the_nearest_untouched_object(tmp_path):
    """Contact exploration (ls20): with an exact move model and no goal that plans, the policy plans to put the agent ON
    the nearest object it has not touched yet (overlap), falling back to standing next to it."""
    from pbg.hypothesis.policy import ContactGoal, contact_targets
    p = Perception(); s = _Session(p); pol = _policy(tmp_path); pol.plan_time = 3.0

    class _Move:
        """Buttons 1-4 move the two-colour agent by 4 cells; the board is otherwise static."""
        D = {1: (-4, 0), 2: (4, 0), 3: (0, -4), 4: (0, 4)}
        def predict(self, sc, a):
            if a.type != "BUTTON" or a.id not in self.D:
                return sc
            dr, dc = self.D[a.id]
            return sc.copy(objects=[o.moved(dr, dc) if o.color in (12, 9) else o for o in sc.objects])
        def with_roles(self, sc):
            objs = []
            for o in sc.objects:
                o = o.moved(0, 0); o.role = "agent" if o.color in (12, 9) else ("collectible" if o.color == 2 else "wall" if o.color == 5 else "unknown"); objs.append(o)
            return sc.copy(objects=objs)

    class _H:
        n = 1; origin = "induced"; model = _Move(); goal = None; goals = None
        def call(self, name, sc, default): return default
        def roled(self, sc): return self.model.with_roles(sc)
    h = _H()
    targets = contact_targets(h.roled(s.scene), set(), set())
    assert targets and all(o.role != "wall" for o in targets)
    goal = ContactGoal(targets[0], "overlap")
    plan = pol.plan_to(h, s, {}, goal, time_limit=2.5)
    assert plan, "a path onto the nearest untouched object must exist"
    sc = s.scene
    for a in plan:
        sc = h.model.predict(sc, a)
    assert goal.is_goal(h.roled(sc))
    assert not ContactGoal(targets[0], "overlap").is_goal(h.roled(s.scene))     # not touched at the start


def test_equal_alternatives_returns_other_usable_variants(tmp_path):
    """Variants the log cannot tell apart (same change accuracy) are offered as alternatives when the top one cannot plan."""
    p = Perception(); s = _Session(p); pol = _policy(tmp_path)
    ids = {}
    cands = pol.induced_candidates(s, s.level_log(), 1, ids=ids, limit=6)
    from pbg.hypothesis.verify import verify
    for c in cands:
        c.verdict = verify(c.model, s.level_log(), c.ignore)
    usable = [c for c in cands if c.verdict.usable(0.8)]
    if len(usable) >= 2:
        top = max(usable, key=lambda c: c.verdict.rank_key())
        alts = pol.equal_alternatives(top, s, s.level_log(), ids, pol.wml, pol.goal_inf, None, 0.8)
        assert alts and all(a.name != top.name for a in alts) and all(a.verdict.usable(0.8) for a in alts)
    else:
        assert pol.equal_alternatives(usable[0] if usable else None, s, s.level_log(), ids, pol.wml, pol.goal_inf, None, 0.8) is not None
