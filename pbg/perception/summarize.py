"""perception/summarize.py — one-line natural-language Diff summary for LLM context (spec §6 step 6)."""
from __future__ import annotations

import numpy as np

from ..core.types import Object, Scene, Transition

COLOR_NAMES = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "grey", 6: "magenta", 7: "orange", 8: "sky", 9: "brown",
               10: "white", 11: "purple", 12: "cyan", 13: "lime", 14: "pink", 15: "teal"}


def obj_tag(o: Object) -> str:
    role = f" {o.role}" if o.role and o.role != "unknown" else ""
    colors = f"{COLOR_NAMES.get(o.color, o.color)}={o.color}" if len(o.colors) == 1 else "colors=" + "+".join(str(c) for c in o.colors)
    return f"obj#{o.id}({colors}{role}, {o.height}x{o.width}@{o.bbox[0]},{o.bbox[1]})"


def summarize(t: Transition, max_len: int = 200) -> str:
    parts = []
    b = {o.id: o for o in t.before.objects}; a = {o.id: o for o in t.after.objects}
    for oid, (dr, dc) in t.diff.moved:
        o = b.get(oid) or a.get(oid)
        if o:
            parts.append(f"{obj_tag(o)} moved ({dr:+d},{dc:+d})")
    for oid, old, new in t.diff.recolored:
        o = b.get(oid) or a.get(oid)
        if o:
            parts.append(f"obj#{oid} recolored {old}->{new}")
    for oid, old, new in t.diff.reshaped:
        o = a.get(oid) or b.get(oid)
        if o:
            parts.append(f"obj#{oid} reshaped (now {o.height}x{o.width}, area {o.area})")
    for o in t.diff.appeared:
        parts.append(f"{obj_tag(o)} appeared")
    for o in t.diff.disappeared:
        parts.append(f"{obj_tag(o)} disappeared")
    if not parts:
        body = "no change" if t.diff.is_noop else f"{t.diff.pixel_changes} pixels changed (no object-level change)"
    else:
        body = "; ".join(parts) + "; others unchanged"
    status = f" [{t.status_change}]" if t.status_change else ""
    text = f"{t.action.label()} -> {body}{status}"
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def scene_text(scene: Scene, max_objects: int = 40) -> str:
    """Structured scene description for prompts (regions + objects)."""
    lines = [f"grid {scene.grid_shape[0]}x{scene.grid_shape[1]}; regions: " + ", ".join(
        f"{r.id}(bg={r.bg_color},{r.kind_hint},rows {r.bbox[0]}-{r.bbox[2] - 1},cols {r.bbox[1]}-{r.bbox[3] - 1})" for r in scene.regions)]
    objs = sorted(scene.objects, key=lambda o: (-o.area, o.id))[:max_objects]
    for o in objs:
        lines.append(f"  {obj_tag(o)} area={o.area} sig={o.shape_sig[:6]} region={o.region}" + (f" colors={list(o.colors)}" if len(o.colors) > 1 else ""))
    if len(scene.objects) > max_objects:
        lines.append(f"  ... {len(scene.objects) - max_objects} more objects")
    if scene.aux:
        aux = {k: v for k, v in scene.aux.items() if not k.startswith("_")}
        if aux:
            lines.append(f"  aux={aux}")
    return "\n".join(lines)


def crop_text(o, max_rows: int = 14, max_cols: int = 20) -> list[str]:
    """Rows of an object's pixels as hex digits ('.' outside its mask), clipped to max_rows x max_cols."""
    cm = o.color_mask if o.color_mask is not None else np.where(o.mask, o.color, -1)
    rows = []
    for r in range(min(o.height, max_rows)):
        rows.append("".join("." if cm[r, c] < 0 else format(int(cm[r, c]), "x") for c in range(min(o.width, max_cols))) + ("~" if o.width > max_cols else ""))
    if o.height > max_rows:
        rows.append("~")
    return rows


def transform_crops(t: Transition, max_area: int = 400) -> list[str]:
    """Before/after crops of objects that changed shape or colour layout in a transition (what a summary line hides:
    a rotation, a flip, a stamped canvas)."""
    b = {o.id: o for o in t.before.objects}; a = {o.id: o for o in t.after.objects}
    ids = [x[0] for x in t.diff.reshaped] + [x[0] for x in t.diff.recolored if x[0] not in {y[0] for y in t.diff.reshaped}]
    out = []
    for oid in ids[:3]:
        ob, oa = b.get(oid), a.get(oid)
        if ob is None or oa is None or max(ob.area, oa.area) > max_area:
            continue
        out.append(f"  obj#{oid} before @({ob.bbox[0]},{ob.bbox[1]}) {ob.height}x{ob.width}:")
        out += ["    " + r for r in crop_text(ob)]
        out.append(f"  obj#{oid} after @({oa.bbox[0]},{oa.bbox[1]}) {oa.height}x{oa.width}:")
        out += ["    " + r for r in crop_text(oa)]
    return out
