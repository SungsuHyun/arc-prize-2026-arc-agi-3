"""core/types.py — the only data types modules exchange (spec §4).

Raw grids travel between Env Wrapper and Perception only; everything above sees Scene / Diff / Transition.
Serialisation is JSON (grids as lists of hex strings, masks as run-length strings)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np

ActionType = Literal["BUTTON", "CLICK", "RESET"]
HEX = "0123456789abcdef"


def grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(grid, dtype=np.int8).tobytes()).hexdigest()[:16]


def grid_to_rows(grid: np.ndarray) -> list[str]:
    return ["".join(HEX[int(v) & 15] for v in row) for row in grid]


def rows_to_grid(rows: list[str]) -> np.ndarray:
    return np.array([[HEX.index(ch) for ch in r] for r in rows], dtype=np.int8)


@dataclass(frozen=True)
class Action:
    type: ActionType
    id: Optional[int] = None          # BUTTON: 1..N (engine ACTION{id})
    row: Optional[int] = None         # CLICK: grid cell coordinates
    col: Optional[int] = None

    # ── constructors ──
    @staticmethod
    def button(i: int) -> "Action":
        return Action("BUTTON", id=int(i))

    @staticmethod
    def click(row: int, col: int) -> "Action":
        return Action("CLICK", row=int(row), col=int(col))

    @staticmethod
    def reset() -> "Action":
        return Action("RESET")

    # ── views ──
    @property
    def key(self) -> str:
        """Action class key used by semantics / rules: 'ACTION1'.. for buttons, 'CLICK', 'RESET'."""
        if self.type == "BUTTON":
            return f"ACTION{self.id}"
        return self.type

    def label(self) -> str:
        if self.type == "CLICK":
            return f"CLICK({self.row},{self.col})"
        return self.key

    def to_json(self) -> dict:
        d: dict = {"type": self.type}
        if self.id is not None:
            d["id"] = self.id
        if self.row is not None:
            d["row"], d["col"] = self.row, self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "Action":
        return Action(d["type"], id=d.get("id"), row=d.get("row"), col=d.get("col"))

    def __str__(self) -> str:  # pragma: no cover
        return self.label()


@dataclass
class Frame:
    grid: np.ndarray                  # (H, W) int8 colour indices
    level: int
    step_idx: int
    hash: str = ""                    # stable hash of grid

    def __post_init__(self):
        self.grid = np.asarray(self.grid, dtype=np.int8)
        if not self.hash:
            self.hash = grid_hash(self.grid)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.grid.shape)  # type: ignore[return-value]

    def to_json(self) -> dict:
        return {"rows": grid_to_rows(self.grid), "level": self.level, "step_idx": self.step_idx, "hash": self.hash}

    @staticmethod
    def from_json(d: dict) -> "Frame":
        return Frame(rows_to_grid(d["rows"]), int(d["level"]), int(d["step_idx"]), d.get("hash", ""))


@dataclass
class Region:
    id: str                           # "R0", "R1", ...
    bg_color: int
    bbox: tuple[int, int, int, int]   # r0, c0, r1, c1 (exclusive)
    kind_hint: Literal["board", "panel", "ui_strip", "unknown"] = "unknown"
    mask: Optional[np.ndarray] = None # bbox-sized bool: the region's real pixels when it is not a full rectangle (corridors)

    @property
    def area(self) -> int:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0) * (c1 - c0)

    @property
    def center(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)

    def contains(self, r: int, c: int) -> bool:
        r0, c0, r1, c1 = self.bbox
        if not (r0 <= r < r1 and c0 <= c < c1):
            return False
        return True if self.mask is None else bool(self.mask[r - r0, c - c0])

    def to_json(self) -> dict:
        d = {"id": self.id, "bg_color": self.bg_color, "bbox": list(self.bbox), "kind_hint": self.kind_hint}
        if self.mask is not None:
            d["mask"] = mask_to_rle(self.mask)
        return d

    @staticmethod
    def from_json(d: dict) -> "Region":
        bbox = tuple(d["bbox"])
        m = rle_to_mask(d["mask"], (bbox[2] - bbox[0], bbox[3] - bbox[1])) if d.get("mask") else None
        return Region(d["id"], int(d["bg_color"]), bbox, d.get("kind_hint", "unknown"), m)


def mask_to_rle(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=bool).ravel()
    out, cur, run = [], False, 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            out.append(str(run)); cur, run = v, 1
    out.append(str(run))
    return ",".join(out)


def rle_to_mask(rle: str, shape: tuple[int, int]) -> np.ndarray:
    runs = [int(x) for x in rle.split(",")] if rle else []
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    pos, cur = 0, False
    for run in runs:
        if cur:
            flat[pos:pos + run] = True
        pos += run; cur = not cur
    return flat.reshape(shape)


def shape_signature(mask: np.ndarray) -> str:
    """Normalised mask hash: crop to bbox, top-left align, hash bytes. Colour is not part of it (spec §6)."""
    m = np.asarray(mask, dtype=bool)
    rows = np.where(m.any(axis=1))[0]; cols = np.where(m.any(axis=0))[0]
    if len(rows) == 0:
        return "empty"
    m = m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return hashlib.sha1(np.packbits(m).tobytes() + bytes(m.shape)).hexdigest()[:12]


def shape_signature_d4(mask: np.ndarray) -> str:
    """Signature invariant under the 8 rotations/reflections (the minimum over the D4 orbit)."""
    m = np.asarray(mask, dtype=bool)
    variants = []
    for k in range(4):
        r = np.rot90(m, k)
        variants.append(shape_signature(r)); variants.append(shape_signature(np.fliplr(r)))
    return min(variants)


@dataclass
class Object:
    id: int                           # tracking id across frames
    color: int                        # representative colour (multi-colour objects: most frequent)
    colors: tuple[int, ...]
    bbox: tuple[int, int, int, int]   # r0, c0, r1, c1 (exclusive)
    mask: np.ndarray                  # bbox-sized bool
    area: int
    shape_sig: str
    region: str
    role: Optional[str] = None
    color_mask: Optional[np.ndarray] = None   # bbox-sized int8 (-1 outside) for multi-colour objects
    parts: Optional[list] = None              # components a fused object was built from (tracking only, not serialised)
    composite: bool = False                   # adjacent components whose union is a filled rectangle (a pattern / canvas / button)

    @property
    def center(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def shape_sig_d4(self) -> str:
        return shape_signature_d4(self.mask)

    def cells(self) -> list[tuple[int, int]]:
        r0, c0 = self.bbox[0], self.bbox[1]
        rr, cc = np.nonzero(self.mask)
        return [(int(r) + r0, int(c) + c0) for r, c in zip(rr, cc)]

    def moved(self, dr: int, dc: int) -> "Object":
        r0, c0, r1, c1 = self.bbox
        return Object(self.id, self.color, self.colors, (r0 + dr, c0 + dc, r1 + dr, c1 + dc), self.mask, self.area,
                      self.shape_sig, self.region, self.role, self.color_mask)

    def recolored(self, color: int) -> "Object":
        cm = None if self.color_mask is None else np.where(self.color_mask >= 0, color, -1).astype(np.int8)
        return Object(self.id, int(color), (int(color),), self.bbox, self.mask, self.area, self.shape_sig, self.region, self.role, cm)

    def overlaps(self, other: "Object") -> bool:
        a, b = self.bbox, other.bbox
        if a[0] >= b[2] or b[0] >= a[2] or a[1] >= b[3] or b[1] >= a[3]:
            return False
        r0, c0, r1, c1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        ma = self.mask[r0 - a[0]:r1 - a[0], c0 - a[1]:c1 - a[1]]
        mb = other.mask[r0 - b[0]:r1 - b[0], c0 - b[1]:c1 - b[1]]
        return bool((ma & mb).any())

    def with_mask(self, mask: np.ndarray, color_mask: Optional[np.ndarray] = None, origin: Optional[tuple[int, int]] = None) -> "Object":
        """Copy with a new pixel layout (bbox-sized bool mask, optional int8 colour mask with -1 outside) anchored at origin."""
        r0, c0 = origin if origin is not None else (self.bbox[0], self.bbox[1])
        mask = np.asarray(mask, dtype=bool)
        cm = None if color_mask is None else np.asarray(color_mask, dtype=np.int8)
        if cm is not None:
            vals = [int(v) for v in np.unique(cm[mask])] if mask.any() else [self.color]
            counts = {v: int(((cm == v) & mask).sum()) for v in vals}
            color = max(counts, key=counts.get); colors = tuple(sorted(counts))
        else:
            color, colors = self.color, (self.color,)
        return Object(self.id, int(color), colors, (r0, c0, r0 + mask.shape[0], c0 + mask.shape[1]), mask, int(mask.sum()), shape_signature(mask),
                      self.region, self.role, color_mask=cm, parts=None, composite=self.composite)

    def rotated(self, k: int = 1) -> "Object":
        """Copy rotated by k x 90 degrees clockwise about the bbox centre (colour layout rotates too)."""
        k = int(k) % 4
        m = np.rot90(self.mask, -k)
        cm = None if self.color_mask is None else np.rot90(self.color_mask, -k)
        cr, cc = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
        r0 = int(round(cr - m.shape[0] / 2)); c0 = int(round(cc - m.shape[1] / 2))
        return self.with_mask(m, cm, (r0, c0))

    def flipped(self, axis: int = 1) -> "Object":
        """Copy mirrored: axis=1 left-right, axis=0 top-bottom."""
        m = np.flip(self.mask, axis)
        cm = None if self.color_mask is None else np.flip(self.color_mask, axis)
        return self.with_mask(m, cm)

    def identity(self) -> tuple:
        """What evaluation compares (spec §8): role-free object identity."""
        if self.color_mask is not None and len(self.colors) >= 2:
            # multi-colour objects differ by their colour layout too (a half-stamped canvas is not a blank one)
            return (int(self.color), tuple(int(x) for x in self.bbox), self.shape_sig, hashlib.sha1(self.color_mask.tobytes()).hexdigest()[:8])
        return (int(self.color), tuple(int(x) for x in self.bbox), self.shape_sig)

    def to_json(self) -> dict:
        d = {"id": self.id, "color": self.color, "colors": list(self.colors), "bbox": list(self.bbox), "mask": mask_to_rle(self.mask),
             "area": self.area, "shape_sig": self.shape_sig, "region": self.region, "role": self.role}
        if self.color_mask is not None:
            d["color_mask"] = [[int(v) for v in row] for row in self.color_mask]
        return d

    @staticmethod
    def from_json(d: dict) -> "Object":
        bbox = tuple(d["bbox"])
        shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        cm = np.array(d["color_mask"], dtype=np.int8) if d.get("color_mask") is not None else None
        return Object(int(d["id"]), int(d["color"]), tuple(d["colors"]), bbox, rle_to_mask(d["mask"], shape), int(d["area"]),
                      d["shape_sig"], d["region"], d.get("role"), cm)


@dataclass
class Scene:
    frame_hash: str
    grid_shape: tuple[int, int]
    regions: list[Region]
    objects: list[Object]
    aux: dict = field(default_factory=dict)   # hidden-state estimates (spec §8)

    # ── lookups ──
    def get(self, obj_id: int) -> Optional[Object]:
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None

    def region(self, rid: str) -> Optional[Region]:
        for r in self.regions:
            if r.id == rid:
                return r
        return None

    def by_role(self, role: str) -> list[Object]:
        return [o for o in self.objects if o.role == role]

    def by_color(self, color: int) -> list[Object]:
        return [o for o in self.objects if o.color == color]

    def in_region(self, rid: str) -> list[Object]:
        return [o for o in self.objects if o.region == rid]

    def region_at(self, r: int, c: int) -> Optional[Region]:
        best = None
        for reg in self.regions:
            if reg.contains(r, c) and (best is None or reg.area < best.area):
                best = reg
        return best

    def key(self) -> tuple:
        """Hashable identity of the scene for search / dedupe (objects + aux)."""
        objs = tuple(sorted(o.identity() for o in self.objects))
        aux = tuple(sorted((k, repr(v)) for k, v in self.aux.items() if not k.startswith("_")))
        return (objs, aux)

    def copy(self, objects: Optional[list[Object]] = None, aux: Optional[dict] = None) -> "Scene":
        return Scene(self.frame_hash, self.grid_shape, list(self.regions), list(self.objects if objects is None else objects),
                     dict(self.aux if aux is None else aux))

    def render(self) -> np.ndarray:
        """Rasterise regions + objects back into a grid (background = region colours)."""
        h, w = self.grid_shape
        g = np.full((h, w), self.aux.get("_global_bg", 0), dtype=np.int8)
        for reg in sorted(self.regions, key=lambda r: -r.area):
            r0, c0, r1, c1 = reg.bbox
            if reg.mask is not None:
                sub = g[r0:r1, c0:c1]
                sub[reg.mask] = reg.bg_color
            else:
                g[r0:r1, c0:c1] = reg.bg_color
        for o in self.objects:
            r0, c0, r1, c1 = o.bbox
            rr0, cc0, rr1, cc1 = max(r0, 0), max(c0, 0), min(r1, h), min(c1, w)
            if rr1 <= rr0 or cc1 <= cc0:
                continue
            sub = o.mask[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
            if o.color_mask is not None:
                cm = o.color_mask[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
                g[rr0:rr1, cc0:cc1] = np.where(sub, cm, g[rr0:rr1, cc0:cc1])
            else:
                g[rr0:rr1, cc0:cc1] = np.where(sub, o.color, g[rr0:rr1, cc0:cc1])
        return g

    def to_json(self) -> dict:
        return {"frame_hash": self.frame_hash, "grid_shape": list(self.grid_shape), "regions": [r.to_json() for r in self.regions],
                "objects": [o.to_json() for o in self.objects], "aux": {k: v for k, v in self.aux.items() if not k.startswith("_") or k == "_global_bg"}}

    @staticmethod
    def from_json(d: dict) -> "Scene":
        return Scene(d["frame_hash"], tuple(d["grid_shape"]), [Region.from_json(r) for r in d["regions"]],
                     [Object.from_json(o) for o in d["objects"]], dict(d.get("aux") or {}))


@dataclass
class Diff:
    moved: list[tuple[int, tuple[int, int]]] = field(default_factory=list)      # (obj_id, (dr, dc))
    appeared: list[Object] = field(default_factory=list)
    disappeared: list[Object] = field(default_factory=list)
    recolored: list[tuple[int, int, int]] = field(default_factory=list)          # (obj_id, old, new)
    reshaped: list[tuple[int, str, str]] = field(default_factory=list)           # (obj_id, old_sig, new_sig)
    pixel_changes: int = 0
    is_noop: bool = True

    def to_json(self) -> dict:
        return {"moved": [[i, list(d)] for i, d in self.moved], "appeared": [o.id for o in self.appeared],
                "disappeared": [o.id for o in self.disappeared], "recolored": [list(x) for x in self.recolored],
                "reshaped": [list(x) for x in self.reshaped], "pixel_changes": self.pixel_changes, "is_noop": self.is_noop}

    def kinds(self) -> set[str]:
        k = set()
        if self.moved: k.add("moved")
        if self.appeared: k.add("appeared")
        if self.disappeared: k.add("disappeared")
        if self.recolored: k.add("recolored")
        if self.reshaped: k.add("reshaped")
        return k


@dataclass
class Transition:
    before: Scene
    action: Action
    after: Scene
    diff: Diff
    status_change: Optional[Literal["LEVEL_UP", "WIN", "GAME_OVER"]] = None
    intermediate_frames: list[Frame] = field(default_factory=list)
    id: str = ""                      # "{game_id}:{level}:{step_idx}" (spec §11)
    level: int = 1
    step_idx: int = 0
    before_frame: Optional[Frame] = None
    after_frame: Optional[Frame] = None

    def to_json(self) -> dict:
        return {"id": self.id, "level": self.level, "step_idx": self.step_idx, "action": self.action.to_json(),
                "before": self.before.to_json(), "after": self.after.to_json(), "diff": self.diff.to_json(),
                "status_change": self.status_change, "n_intermediate": len(self.intermediate_frames)}


def transitions_by_level(log: Iterable[Transition]) -> dict[int, list[Transition]]:
    out: dict[int, list[Transition]] = {}
    for t in log:
        out.setdefault(t.level, []).append(t)
    return out
