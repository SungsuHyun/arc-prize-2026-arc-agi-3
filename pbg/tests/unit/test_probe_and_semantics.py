from pbg.core.types import Action
from pbg.perception import Perception
from pbg.probe.semantics import classify_actions
from pbg.tests.unit.synthetic import frame


def _log():
    p = Perception()
    prev = p.parse(frame(0, agent=(30, 30)), None)
    ts = []
    plan = [(1, (26, 30)), (2, (30, 30)), (2, (34, 30)), (5, (34, 30)), (3, (34, 26)), (5, (34, 26)), (4, (34, 30))]
    for i, (b, pos) in enumerate(plan, 1):
        cur = p.parse(frame(i, agent=pos), prev)
        t = p.transition(prev, Action.button(b), cur); t.id = f"g:1:{i}"; ts.append(t); prev = t.after
    return ts


def test_classify_move_and_noop():
    sem = classify_actions(_log())
    assert sem["ACTION1"]["class"] == "MOVE" and tuple(sem["ACTION1"]["displacement"]) == (-4, 0)
    assert sem["ACTION2"]["class"] == "MOVE" and tuple(sem["ACTION2"]["displacement"]) == (4, 0)
    assert sem["ACTION5"]["class"] == "NOOP"   # two observations from two different states
    assert set(sem.moves()) == {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    assert sem.mover_ids()


def test_single_noop_observation_is_unknown():
    p = Perception()
    s0 = p.parse(frame(0), None); s1 = p.parse(frame(1), s0)
    t = p.transition(s0, Action.button(7), s1); t.id = "g:1:1"
    sem = classify_actions([t])
    assert sem["ACTION7"]["class"] == "UNKNOWN"



def test_click_map_ranks_untried_then_responsive_and_skips_inert():
    """Click response map: clicks are clustered by target class; inert classes are not re-clicked, responsive ones are."""
    import numpy as np
    from pbg.core.types import Action, Frame
    from pbg.perception import Perception
    from pbg.probe.clickmap import ClickMap, click_key
    g = np.full((64, 64), 3, dtype=np.int8)
    g[10:14, 10:14] = 9; g[20:24, 10:14] = 9          # two buttons (same class)
    g[30:34, 20:40] = 5                                # a bar (inert)
    g[50:54, 30:34] = 4                                # a mover
    p = Perception()
    s0 = p.parse(Frame(g.tolist(), 1, 0), None)
    def step(scene, row, col, g_after, tid):
        s1 = p.parse(Frame(g_after.tolist(), 1, 1), scene)
        t = p.transition(scene, Action.click(row, col), s1, before_frame=Frame(g.tolist(), 1, 0), after_frame=Frame(g_after.tolist(), 1, 1), tid=tid, level=1)
        return t
    g1 = g.copy(); g1[50:54, 30:34] = 3; g1[50:54, 34:38] = 4      # button click moves the mover
    m = ClickMap()
    m.add(step(s0, 11, 11, g1, "t1"))          # button: responsive
    m.add(step(s0, 31, 25, g, "t2"))           # bar: inert
    assert m.status(click_key(s0, 11, 11)) == "responsive" and m.status(click_key(s0, 31, 25)) == "inert"
    ranked = m.rank(s0, level=1, include_inert=True)
    labels = [a.label() for a in ranked]
    assert labels[0] == "CLICK(51,31)"                        # the untried mover class comes first
    assert "CLICK(11,11)" in labels and "CLICK(21,11)" in labels   # every object of the responsive class
    assert not any(a.row == 31 for a in ranked)               # the inert bar was clicked on this level already: skipped
    ranked2 = m.rank(s0, level=2, include_inert=True)
    assert any(a.row == 31 for a in ranked2) and [a.label() for a in ranked2][-1].startswith("CLICK(31")   # next level: once, last
    assert not any(a.row == 31 for a in m.rank(s0, level=2, include_inert=False))
    m2 = ClickMap.from_json(m.to_json())
    assert m2.n == m.n and m2.responsive == m.responsive and m2.tried_level == m.tried_level


def test_level_knowledge_goal_stats_roundtrip(tmp_path):
    from pbg.orchestrator.knowledge import LevelKnowledge
    from pbg.goal.templates import GoalInstance
    k = LevelKnowledge()
    g = GoalInstance("align_color(11,col)", "align_color", {}, lambda s: False, lambda s: 0.0)
    k.record_win(1, g, 4); k.record_fail(GoalInstance("inside_frame(5)", "inside_frame", {}, lambda s: False, lambda s: 0.0))
    st = k.goal_stats({"reach": {"games_used": 4, "games_verified": 1}})
    assert st["align_color"]["games_verified"] / st["align_color"]["games_used"] >= 0.75 and st["reach"]["games_used"] == 4
    k.save(tmp_path / "k.json"); k2 = LevelKnowledge.load(tmp_path / "k.json")
    assert k2.goal_wins["align_color"] == 1 and k2.goal_fails["inside_frame"] == 1 and k2.won_goals[0]["level"] == 1
