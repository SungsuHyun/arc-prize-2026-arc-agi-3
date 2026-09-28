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
