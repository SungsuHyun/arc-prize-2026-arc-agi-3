"""perception/render.py — PNG rendering of a scene / grid (LLM auxiliary input). Pure zlib PNG writer, no PIL needed."""
from __future__ import annotations

import struct
import zlib

import numpy as np

PALETTE = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
           (127, 219, 255), (135, 12, 37), (255, 255, 255), (177, 13, 201), (57, 204, 204), (1, 255, 112), (255, 153, 204), (60, 120, 120)]


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def grid_png(grid: np.ndarray, scale: int = 8, gridlines: bool = False) -> bytes:
    g = np.asarray(grid)
    h, w = g.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    pal = np.array(PALETTE, dtype=np.uint8)
    rgb = pal[np.clip(g, 0, 15)]
    img[:] = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    if gridlines and scale >= 4:
        img[::scale, :, :] = img[::scale, :, :] // 2
        img[:, ::scale, :] = img[:, ::scale, :] // 2
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h * scale))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w * scale, h * scale, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))


def render_scene(scene, scale: int = 8) -> bytes:
    return grid_png(scene.render(), scale)
