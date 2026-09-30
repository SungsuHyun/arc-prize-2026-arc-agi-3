"""docs/029 priority 3: a click-is-a-destination game (r11l-style) is induced from the log and verifies at pixel level."""
import numpy as np

from pbg.core.types import Action, Frame
from pbg.perception import Perception
from pbg.wml.point import induce_point_hypotheses, point_click_candidates
from pbg.hypothesis.verify import verify


def _grid(block):
    g = np.full((64, 64), 4, dtype=np.int8); g[4:60, 4:60] = 3
    g[50:54, 10:14] = 5                             # a static block (never moves)
    r, c = block; g[r:r + 3, c:c + 3] = 12          # the piece
    return g


def _log():
    """Three clicks; the piece steps one cell toward each click along the axis of greater distance."""
    p = Perception()
    pos = [(30, 30), (30, 31), (29, 31), (29, 30)]
    clicks = [Action.click(30, 50), Action.click(10, 31), Action.click(29, 5)]
    frames = [Frame(_grid(x), 1, i) for i, x in enumerate(pos)]
    scenes = [p.parse(frames[0], None)]; log = []
    for i, a in enumerate(clicks, 1):
        scenes.append(p.parse(frames[i], scenes[-1]))
        log.append(p.transition(scenes[-2], a, scenes[-1], before_frame=frames[i - 1], after_frame=frames[i], status_change=None, tid=f"g:1:{i}", level=1, step_idx=i))
    return log, scenes[-1]


def test_point_model_is_induced_and_verifies():
    log, scene = _log()
    models = induce_point_hypotheses(log, {}, [Action("CLICK")])
    assert models and models[0][0].startswith("point_4_s1"), [m[0] for m in models]   # a 1-cell step, not a jump
    name, model = models[0]
    v = verify(model, log, None)
    assert v.n == 3 and v.accuracy == 1.0 and v.change_accuracy == 1.0
    roled = model.with_roles(scene)
    assert any(o.role == "mover" for o in roled.objects)
    cands = point_click_candidates(roled, 1)
    assert len(cands) >= 8 and all(a.type == "CLICK" for a in cands)   # 8 steering clicks + destinations


def test_point_model_not_induced_from_unrelated_moves():
    """Moves that do not point toward the click (a permutation) must not yield a point model."""
    p = Perception()
    frames = [Frame(_grid((30, 30)), 1, 0), Frame(_grid((40, 40)), 1, 1), Frame(_grid((30, 30)), 1, 2)]
    scenes = [p.parse(frames[0], None)]; log = []
    for i, a in enumerate([Action.click(5, 5), Action.click(5, 5)], 1):
        scenes.append(p.parse(frames[i], scenes[-1]))
        log.append(p.transition(scenes[-2], a, scenes[-1], before_frame=frames[i - 1], after_frame=frames[i], status_change=None, tid=f"g:1:{i}", level=1, step_idx=i))
    assert induce_point_hypotheses(log, {}, [Action("CLICK")]) == []


def test_jump_model_is_induced_when_the_piece_lands_on_the_click():
    p = Perception()
    pos = [(30, 30), (10, 40), (45, 20)]                      # the 3x3 piece is centred on each click
    clicks = [Action.click(11, 41), Action.click(46, 21)]
    frames = [Frame(_grid(x), 1, i) for i, x in enumerate(pos)]
    scenes = [p.parse(frames[0], None)]; log = []
    for i, a in enumerate(clicks, 1):
        scenes.append(p.parse(frames[i], scenes[-1]))
        log.append(p.transition(scenes[-2], a, scenes[-1], before_frame=frames[i - 1], after_frame=frames[i], status_change=None, tid=f"g:1:{i}", level=1, step_idx=i))
    models = induce_point_hypotheses(log, {}, [Action("CLICK")])
    assert models and models[0][0].startswith("point_jump"), [m[0] for m in models]
    v = verify(models[0][1], log, None)
    assert v.accuracy == 1.0 and v.change_accuracy == 1.0
    cands = point_click_candidates(models[0][1].with_roles(scenes[-1]), 1)
    assert any(a == Action.click(51, 11) for a in cands)     # the static block's centre is a destination candidate
