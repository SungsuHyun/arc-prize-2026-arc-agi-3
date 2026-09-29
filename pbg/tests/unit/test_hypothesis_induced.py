"""docs/029 P1: deterministic induced models compete as hypothesis candidates, and a usable model without a plan takes a
goal-directed step instead of idling."""
from pathlib import Path

from pbg.core.budget import Budget
from pbg.core.types import Action
from pbg.hypothesis.policy import HypothesisPolicy, InducedHypothesis
from pbg.memory.store import Memory
from pbg.perception import Perception
from pbg.tests.unit.synthetic import frame


class _Status:
    state = "NOT_FINISHED"


class _Env:
    def status(self):
        return _Status()


class _Session:
    """A level log of three RIGHT moves (the synthetic agent moves 4 columns per press)."""
    def __init__(self, p: Perception):
        self.level = 1; self.env = _Env()
        frames = [frame(i, agent=(30, 30 + 4 * i)) for i in range(4)]
        scenes = [p.parse(frames[0], None)]
        self.transitions = []
        for i in range(1, 4):
            scenes.append(p.parse(frames[i], scenes[-1]))
            self.transitions.append(p.transition(scenes[-2], Action.button(4), scenes[-1], before_frame=frames[i - 1], after_frame=frames[i],
                                                 status_change=None, tid=f"g:1:{i}", level=1, step_idx=i))
        self.scene = scenes[-1]

    def available_actions(self):
        return [Action.button(i) for i in (1, 2, 3, 4)]

    def level_log(self):
        return self.transitions


def _policy(tmp_path: Path) -> HypothesisPolicy:
    class _LLM:
        cfg = {"timeout": 1}; calls = 0
        def exhausted(self): return True
    return HypothesisPolicy(memory=Memory(tmp_path / "mem"), llm=_LLM())


def test_induced_candidates_have_the_hypothesis_interface(tmp_path):
    p = Perception(); s = _Session(p); pol = _policy(tmp_path)
    cands = pol.induced_candidates(s, s.level_log(), n=1)
    assert cands, "the move semantics must induce at least one model"
    c = cands[0]
    assert isinstance(c, InducedHypothesis) and c.origin == "induced" and c.name
    from pbg.hypothesis.verify import verify
    v = verify(c.model, s.level_log(), c.ignore)
    assert v.n == 3                                              # verified on every transition of the log
    acts = c.call("candidate_actions", s.scene, [])
    assert acts and all(isinstance(a, Action) for a in acts)     # plannable: buttons from the action set
    assert "induced model" in c.summary()


def test_goal_directed_step_prefers_progress(tmp_path):
    p = Perception(); s = _Session(p); pol = _policy(tmp_path)

    class _Goal:
        name = "go right"
        def is_goal(self, sc): return False
        def progress(self, sc):
            ag = [o for o in sc.objects if o.color == 12]
            return ag[0].center[1] / 64 if ag else 0.0

    class _Model:
        """RIGHT moves the agent 4 columns; every other button is a no-op."""
        def predict(self, sc, a):
            if a.type == "BUTTON" and a.id == 4:
                objs = [o.moved(0, 4) if o.color in (12, 9) else o for o in sc.objects]
                return sc.copy(objects=objs)
            return sc
        def with_roles(self, sc): return sc
        def assign_roles(self, sc): return {}
        def rules(self): return []

    class _H:
        goal = _Goal(); model = _Model(); goals = [goal]
        def call(self, name, sc, default): return default
        def progress(self, sc): return self.goal.progress(sc)
        def is_goal(self, sc): return False
    a = pol.goal_directed_step(_H(), s, {}, set(), set())
    assert a is not None and a.type == "BUTTON" and a.id == 4
    # the same step is not repeated from the same board once it was tried there
    from pbg.planner.common import state_key
    assert pol.goal_directed_step(_H(), s, {}, {(state_key(s.scene), "ACTION4")}, set()) != Action.button(4)


def test_nothing_happens_model_is_not_usable():
    """docs/029: a model that predicts 'no change' is exact on every no-op transition and useless; the verdict's change
    accuracy must keep it from ranking above a model that explains the moves."""
    from pbg.hypothesis.verify import verify
    p = Perception(); s = _Session(p)

    class _Noop:
        def predict(self, sc, a): return sc
        def with_roles(self, sc): return sc
    v = verify(_Noop(), s.level_log(), None)
    assert v.changed == 3 and v.change_correct == 0 and v.change_accuracy == 0.0
    assert v.accuracy == 0.0 and not v.usable(0.8)

    class _Right:
        def predict(self, sc, a):
            return sc.copy(objects=[o.moved(0, 4) if o.color in (12, 9) else o for o in sc.objects]) if a.id == 4 else sc
        def with_roles(self, sc): return sc
    w = verify(_Right(), s.level_log(), None)
    assert w.change_accuracy > v.change_accuracy and w.rank_key() > v.rank_key()


def test_induced_ids_are_stable_across_rounds(tmp_path):
    p = Perception(); s = _Session(p); pol = _policy(tmp_path); ids = {}
    a = pol.induced_candidates(s, s.level_log(), n=1, ids=ids)
    b = pol.induced_candidates(s, s.level_log(), n=7, ids=ids)
    assert a and b and {c.name: c.n for c in a} == {c.name: c.n for c in b if c.name in {x.name for x in a}}


def test_goal_directed_step_never_returns_an_unavailable_action(tmp_path):
    """lp85 (click-only) ended after 4 minutes: the LLM's candidate ACTION1 was refused and the game loop broke out."""
    p = Perception(); s = _Session(p); pol = _policy(tmp_path)
    s.available_actions = lambda: [Action("CLICK")]          # click-only game

    class _Goal:
        name = "g"
        def is_goal(self, sc): return False
        def progress(self, sc): return 0.0

    class _Model:
        def predict(self, sc, a): return sc.copy(objects=[o.moved(0, 1) for o in sc.objects])   # every action changes the board
        def with_roles(self, sc): return sc

    class _H:
        goal = _Goal(); model = _Model(); goals = [goal]
        def call(self, name, sc, default): return [Action.button(1), Action.button(4)] if name == "candidate_actions" else default
        def progress(self, sc): return 0.0
        def is_goal(self, sc): return False
    a = pol.goal_directed_step(_H(), s, {}, set(), set())
    assert a is None or a.type == "CLICK"
