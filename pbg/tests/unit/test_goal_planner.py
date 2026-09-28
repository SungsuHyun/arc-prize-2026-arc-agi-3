from pbg.core.types import Action
from pbg.goal.filter import filter_by_levelup
from pbg.goal.templates import instantiate_all
from pbg.perception import Perception
from pbg.planner.execute import Planner
from pbg.probe.semantics import classify_actions
from pbg.wml.evaluate import evaluate, make_hypothesis
from pbg.wml.induce import induce_hypotheses
from pbg.tests.unit.synthetic import frame
from pbg.tests.unit.test_wml import _walk_log


def _model_and_scene():
    ts = _walk_log(32)
    sem = classify_actions(ts)
    cands = induce_hypotheses(ts, sem, [Action.button(i) for i in (1, 2, 3, 4)])
    name, m = max(cands, key=lambda nm: evaluate(nm[1], ts).score)
    return ts, sem, make_hypothesis(m, evaluate(m, ts), name=name)


def test_goal_templates_instantiate_with_roles():
    ts, sem, h = _model_and_scene()
    scene = h.model.with_roles(ts[-1].after)
    G = instantiate_all(scene)
    names = {g.name for g in G}
    assert any(n.startswith("reach(agent") for n in names) and any(n.startswith("all_removed") for n in names)
    assert all(not g.is_goal(scene) for g in G)


def test_levelup_filter_keeps_true_goal():
    ts, sem, h = _model_and_scene()
    roles = h.model.with_roles
    G = instantiate_all(roles(ts[0].before))
    target = next(g for g in G if g.name == "all_removed(2)")
    # fabricate a level-up transition: the level's last state has no red item left (the next level's board has items again)
    p = Perception()
    before = roles(p.parse(frame(0, agent=(40, 36), items=()), None))
    after = roles(p.parse(frame(1, agent=(30, 30), items=((40, 40),)), None))
    from pbg.core.types import Transition, Diff
    lv = Transition(before, Action.button(4), after, Diff(is_noop=False), "LEVEL_UP", [], "g:1:99", 1, 99)
    kept = filter_by_levelup(G + [target], ts + [lv])
    assert any(g.name == "all_removed(2)" for g in kept)
    assert all(g.name != "sort_by(row,color)" or g.is_goal(before) for g in kept)


def test_planner_reaches_collectible_and_stops_on_mismatch():
    ts, sem, h = _model_and_scene()
    scene = ts[-1].after
    roles = h.model.with_roles
    G = [g for g in instantiate_all(roles(scene)) if g.name.startswith("reach(agent")]
    G.sort(key=lambda g: g.name not in ("reach(agent,collectible)", "reach(agent,color:2)"))
    planner = Planner(time_limit=2.0)
    plan = planner.search(scene, [h], G[:1] or G, available=[Action.button(i) for i in (1, 2, 3, 4)], semantics=sem)
    assert plan, planner.last_info.reason if planner.last_info else "no plan"
    assert len(plan) <= 12

    class FakeSession:
        def __init__(self, s):
            self.scene = s; self.p = Perception(); self.n = 0
        def act(self, a, kind="plan"):
            self.n += 1
            # the environment ignores the action (agent does not move) -> mismatch on the first step
            cur = self.p.parse(frame(self.n, agent=(30, 30)), None)
            from pbg.core.types import Transition, Diff
            t = Transition(self.scene, a, cur, Diff(is_noop=True), None, [], f"g:1:{self.n}", 1, self.n)
            self.scene = cur
            return t
    r = planner.execute(plan, [h], FakeSession(scene), G)
    assert r.kind == "MISMATCH" and r.executed == 1
