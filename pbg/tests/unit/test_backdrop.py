"""docs/029: a predicted scene paints the cells a moved object vacates with the colour last seen there (Scene.backdrop),
not with the region background — the ls20 move model collapsed because a corridor under the piece rendered wrongly."""
import numpy as np

from pbg.core.types import Action, Frame
from pbg.perception import Perception
from pbg.tests.unit.synthetic import frame


def _grid(piece, corridor=True):
    g = np.full((64, 64), 4, dtype=np.int8); g[4:60, 4:60] = 3          # floor 3 on background 4
    if corridor:
        g[30:46, 29:34] = 4                                              # a corridor of background colour inside the floor (wider than the piece)
    r, c = piece; g[r:r + 3, c:c + 3] = 12                               # the piece
    return g


def test_render_round_trips_the_parsed_frame():
    p = Perception()
    f = frame(0)
    s = p.parse(f, None)
    strips = [r.bbox for r in s.regions if r.kind_hint == "ui_strip"]
    d = s.render() != f.grid
    for r0, c0, r1, c1 in strips:
        d[r0:r1, c0:c1] = False
    assert not d.any()


def test_vacated_cells_show_what_was_under_the_piece():
    """The piece walks into the corridor (a static object) and the model moves it out again: the corridor must be drawn
    where the piece was. The corridor object is kept in the scene as occluded while the piece covers it."""
    p = Perception()
    s0 = p.parse(Frame(_grid((20, 20)), 1, 0), None)
    sa = p.parse(Frame(_grid((25, 30)), 1, 1), s0)                # the piece walks up to the corridor (the corridor is static scenery now)
    s1 = p.parse(Frame(_grid((36, 30)), 1, 2), sa)                # and stands inside it
    assert int(s1.backdrop[21, 21]) == 3                           # the floor it left is visible again (backdrop)
    assert int(s1.backdrop[37, 31]) == 4                           # under the piece: the corridor, recorded while it was static scenery
    piece = [o for o in s1.objects if o.color == 12][0]
    pred = s1.copy(objects=[o.moved(-16, -10) if o.id == piece.id else o for o in s1.objects])   # the model moves it back out
    r = pred.render()
    assert r[36:39, 30:33].tolist() == [[4] * 3] * 3      # corridor colour where the piece was
    assert r[20:23, 20:23].tolist() == [[12] * 3] * 3    # the piece at its predicted place
    s2 = p.parse(Frame(_grid((20, 20)), 1, 3), s1)                 # it really moves out: the corridor is segmented again, same id
    assert not any(o.occluded for o in s2.objects)


def test_backdrop_and_occlusion_reset_per_level():
    p = Perception()
    s0 = p.parse(Frame(_grid((20, 20)), 1, 0), None)
    s1 = p.parse(Frame(_grid((36, 30)), 1, 1), s0)
    assert int(s1.backdrop[21, 21]) == 3
    s2 = p.parse(Frame(_grid((36, 30)), 2, 0), None)      # a new board: uncovered cells recorded, object cells unknown, nothing carried
    assert int(s2.backdrop[21, 21]) == 3 and int(s2.backdrop[37, 31]) == -1 and not any(o.occluded for o in s2.objects)
