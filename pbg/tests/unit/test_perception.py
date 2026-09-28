import numpy as np

from pbg.core.types import Action
from pbg.perception import Perception, summarize
from pbg.tests.unit.synthetic import frame


def test_regions_objects_and_strip():
    p = Perception()
    s = p.parse(frame(), None)
    kinds = {r.kind_hint for r in s.regions}
    assert "ui_strip" in kinds and any(r.bg_color == 3 for r in s.regions)
    colors = {o.color for o in s.objects}
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
    p = Perception({"merge_confirm_count": 3, "track_cost_max": 20.0})
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
