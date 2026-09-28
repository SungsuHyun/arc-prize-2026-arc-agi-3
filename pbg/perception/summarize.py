"""perception/summarize.py — one-line natural-language Diff summary for LLM context (spec §6 step 6)."""
from __future__ import annotations

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
