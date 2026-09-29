import numpy as np

from pbg.core.types import Action
from pbg.perception import Perception, summarize
from pbg.tests.unit.synthetic import frame


def test_regions_objects_and_strip():
    p = Perception()
    s = p.parse(frame(), None)
    kinds = {r.kind_hint for r in s.regions}
    assert "ui_strip" in kinds and any(r.bg_color == 3 for r in s.regions)
    colors = {c for o in s.objects for c in o.colors}   # the two-colour agent is one composite object from the first frame
    assert {12, 9, 5, 2, 11} <= colors
    assert all(o.region for o in s.objects)


def test_tracking_and_diff_move():
    p = Perception()
    s0 = p.parse(frame(0, agent=(30, 30)), None)
    s1 = p.parse(frame(1, agent=(34, 30)), s0)
    t = p.transition(s0, Action.button(2), s1)
    moved = dict(t.diff.moved)
    cyan = next(o for o in s0.objects if o.color == 12)
    assert moved.get(cyan.id) == (4, 0) and not t.diff.is_noop
    assert "moved (+4,+0)" in summarize(t)


def test_merge_confirmation_after_three_comoves():
    p = Perception({"merge_confirm_count": 3, "track_cost_max": 20.0, "composite_min_area": 10 ** 6})   # composites off: test the co-movement path
    prev = p.parse(frame(0, agent=(20, 30)), None)
    ts = []
    for i in range(1, 5):
        cur = p.parse(frame(i, agent=(20 + 4 * i, 30)), prev)
        ts.append(p.transition(prev, Action.button(2), cur))
        prev = ts[-1].after
    assert p.merge_log(), "adjacent co-moving components must be merged"
    fused = [o for o in prev.objects if len(o.colors) > 1]
    assert fused and set(fused[0].colors) == {9, 12} and fused[0].area == 16
    p.refit(ts)
    assert any(len(o.colors) > 1 for o in ts[0].before.objects), "past scenes are refit after a merge"


def test_noop_and_timing():
    p = Perception()
    s0 = p.parse(frame(0), None); s1 = p.parse(frame(1), s0)
    t = p.transition(s0, Action.button(5), s1, before_frame=frame(0), after_frame=frame(1))
    assert t.diff.is_noop and t.diff.pixel_changes == 0
    assert max(p.timings) < 0.05


def test_render_roundtrip_of_scene():
    p = Perception()
    f = frame()
    s = p.parse(f, None)
    g = s.render()
    assert np.array_equal(g, f.grid)



def test_rect_composites_and_solo_dissolution():
    """A two-colour rectangle becomes one composite object; a part that moves alone dissolves it (with refit)."""
    import numpy as np
    from pbg.core.types import Frame
    from pbg.perception import Perception
    g = np.full((64, 64), 5, dtype=np.int8)
    g[10:15, 10:20] = 0; g[15:20, 10:20] = 3            # target pattern: black over green, a 10x10 rectangle
    g[40:43, 40:43] = 7; g[40:43, 43:50] = 9             # an agent (7) standing next to a wall segment (9): also a rectangle
    p = Perception()
    s0 = p.parse(Frame(g.tolist(), 1, 0), None)
    comps = [o for o in s0.objects if o.composite]
    assert {tuple(o.bbox) for o in comps} == {(10, 10, 20, 20), (40, 40, 43, 50)}
    pat = next(o for o in comps if o.bbox == (10, 10, 20, 20))
    assert set(pat.colors) == {0, 3} and pat.color_mask is not None and pat.color_mask[0, 0] == 0 and pat.color_mask[9, 0] == 3
    # the agent moves down: the (agent, wall) composite must dissolve, the pattern stays fused
    g2 = g.copy(); g2[40:43, 40:43] = 5; g2[45:48, 40:43] = 7
    s1 = p.parse(Frame(g2.tolist(), 1, 0), s0)
    from pbg.core.types import Action
    t = p.transition(s0, Action.button(2), s1, before_frame=Frame(g.tolist(), 1, 0), after_frame=Frame(g2.tolist(), 1, 0))
    assert p.merge_confirmed          # dissolution -> callers refit
    assert not any(o.composite and 9 in o.colors for o in t.before.objects)
    assert any(o.composite and set(o.colors) == {0, 3} for o in t.after.objects)
    moved = dict(t.diff.moved)
    agent = next(o for o in t.after.objects if o.color == 7)
    assert moved.get(agent.id) == (5, 0) and not t.diff.appeared and not t.diff.disappeared


def test_match_pattern_goal():
    import numpy as np
    from pbg.core.types import Frame
    from pbg.perception import Perception
    from pbg.goal.templates import instantiate_all
    g = np.full((64, 64), 5, dtype=np.int8)
    g[10:15, 10:20] = 0; g[15:20, 10:20] = 3            # target
    g[40:50, 30:40] = 0                                  # blank canvas of the same size
    p = Perception(); s = p.parse(Frame(g.tolist(), 1, 0), None)
    goals = [x for x in instantiate_all(s, {}) if x.template == "match_pattern"]
    assert goals, "no match_pattern goal"
    tgt = next(o for o in s.objects if o.composite); cv = next(o for o in s.objects if o.bbox == (40, 30, 50, 40))
    gi = next(x for x in goals if x.params["target"] == tgt.id and x.params["canvas"] == cv.id)
    assert gi.progress(s) == 0.5 and gi.estimate(s) == 1.0
    g2 = g.copy(); g2[45:50, 30:40] = 3
    s2 = p.parse(Frame(g2.tolist(), 1, 0), s)
    assert gi.is_goal(s2) and gi.progress(s2) == 1.0
