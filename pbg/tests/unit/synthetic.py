"""Synthetic grids for unit tests: a green board on yellow background, a two-colour agent, walls, a collectible and
a bottom gauge strip."""
from __future__ import annotations

import numpy as np

from pbg.core.types import Frame

H = W = 64


def board(agent=(30, 30), walls=((20, 20),), items=((40, 40),), gauge: int = 40) -> np.ndarray:
    g = np.full((H, W), 4, dtype=np.int8)
    g[8:56, 8:56] = 3                                  # board
    for r, c in walls:
        g[r:r + 4, c:c + 4] = 5                        # wall blocks
    for r, c in items:
        g[r:r + 2, c:c + 2] = 2                        # collectibles
    ar, ac = agent
    g[ar:ar + 2, ac:ac + 4] = 12                       # agent top (cyan)
    g[ar + 2:ar + 4, ac:ac + 4] = 9                    # agent bottom (brown)
    g[61:63, 4:4 + gauge] = 11                         # gauge strip at the bottom edge
    return g


def frame(step: int = 0, **kw) -> Frame:
    return Frame(board(**kw), 1, step)
