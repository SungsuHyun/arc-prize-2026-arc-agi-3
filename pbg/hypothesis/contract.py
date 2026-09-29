"""hypothesis/contract.py — the game-hypothesis document the reasoning model writes, and the observations it gets.

The policy (hypothesis/policy.py) is the same for every game: look at the board, write a hypothesis of the whole game
(roles, controls, dynamics, win condition), name the actions that would test what is uncertain, run those, verify the
hypothesis at pixel level, revise with the counter-examples, and only plan on a verified hypothesis. Nothing in this
module knows a game; the priors of the mechanism library are offered as hints, never applied by themselves."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import Action, Scene, Transition
from ..wml.prompts import API_SUMMARY

CONTRACT = '''You are playing an unknown pixel board game (64x64 grid, colours 0-15) through an agent. You never see the rules:
you infer them from the board and from what actions did. Work like a scientist: state a HYPOTHESIS of the whole game,
say what is uncertain, and name the cheapest actions that would settle it. Every action costs score, so test before
you plan and plan only what the hypothesis supports.

Reply with exactly ONE ```python block defining:

HYPOTHESIS = {
  "summary": "one paragraph: what the objects are, what the player controls, what the win condition probably is",
  "roles": {"<colour or description>": "<role>", ...},
  "controls": "how actions change the board (be concrete: which click targets or buttons do what, by how much)",
  "win": "the win condition as you believe it",
  "uncertain": ["what you still do not know", ...],
}

def role_fn(scene) -> dict:            # object id -> role name (strings; use the vocabulary below or custom:<name>)
def build_model() -> RuleModel         # RuleModel(rules, role_fn, default="unknown"); rules predict the NEXT scene exactly
def build_goal():                      # object with name, is_goal(scene) -> bool, progress(scene) -> float in [0,1]
def candidate_actions(scene) -> list:  # the actions worth planning with in this state (Action.button(i) / Action.click(row, col));
                                       # for click games include the POINTS that matter (targets, gaps), not only objects
def test_actions(scene) -> list:       # 1-3 actions whose outcome best discriminates the "uncertain" items (empty if none)
def ignore_boxes(scene) -> list:       # [(r0, c0, r1, c1), ...] display elements (counters, timers) to ignore when checking predictions

Keep the code compact (well under 200 lines): simple geometric rules beat elaborate simulations, and a reply cut off by
the length limit is worthless. Rules for the code: numpy only; no game names; no coordinates memorised from this level except through scene queries;
no comments that argue -- put reasoning in HYPOTHESIS. The prediction check compares the rendered predicted grid with the
real next grid pixel by pixel outside ignore_boxes: a rule that moves an object must move ALL its pixels correctly.
If an action is truly unpredictable to you, return None from the rule (UNKNOWN) rather than guessing a no-op.
Roles vocabulary: agent, mover, button, bar, wall, target, marker, key, door, collectible, hazard, indicator, decoration.
'''


def frame_text(grid: np.ndarray, rows: Optional[tuple[int, int]] = None, cols: Optional[tuple[int, int]] = None) -> str:
    """The grid (or a window of it) as hex rows with a row index, one char per cell."""
    g = np.asarray(grid)
    r0, r1 = rows or (0, g.shape[0]); c0, c1 = cols or (0, g.shape[1])
    head = "     " + "".join(str(c % 10) for c in range(c0, c1))
    lines = [head] + [f"{r:3d}: " + "".join(format(int(v), "x") for v in g[r, c0:c1]) for r in range(r0, r1)]
    return "\n".join(lines)


def scene_objects_text(scene: Scene, limit: int = 120) -> str:
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    lines = ["regions: " + ", ".join(f"{r.id}(bg={r.bg_color},{r.kind_hint},rows {r.bbox[0]}-{r.bbox[2] - 1},cols {r.bbox[1]}-{r.bbox[3] - 1})" for r in scene.regions)]
    objs = sorted(scene.objects, key=lambda o: (o.region in strips, -o.area, o.id))
    for o in objs[:limit]:
        col = f"colours={list(o.colors)}" if len(o.colors) > 1 else f"colour={o.color}"
        lines.append(f"  obj#{o.id} {col} bbox=({o.bbox[0]},{o.bbox[1]})-({o.bbox[2]},{o.bbox[3]}) size={o.height}x{o.width} area={o.area}"
                     + (" [ui strip]" if o.region in strips else ""))
    if len(objs) > limit:
        lines.append(f"  ... {len(objs) - limit} more")
    return "\n".join(lines)


def change_window(t: Transition, pad: int = 2) -> Optional[tuple[int, int, int, int]]:
    b = np.asarray(t.before_frame.grid); a = np.asarray(t.after_frame.grid)
    ys, xs = np.nonzero(a != b)
    if len(ys) == 0:
        return None
    return (max(0, int(ys.min()) - pad), max(0, int(xs.min()) - pad), min(b.shape[0], int(ys.max()) + 1 + pad), min(b.shape[1], int(xs.max()) + 1 + pad))


def transition_text(t: Transition, ignore_boxes: list = (), max_rows: int = 22, max_cols: int = 40) -> str:
    """What one action did: the action (with click position), the changed pixels, and before/after windows of the change."""
    b = np.asarray(t.before_frame.grid); a = np.asarray(t.after_frame.grid)
    mask = a != b
    for (r0, c0, r1, c1) in ignore_boxes:
        mask[r0:r1, c0:c1] = False
    n = int(mask.sum())
    head = f"[{t.id}] {t.action.label()}"
    if t.status_change:
        head += f"  -> {t.status_change}"
    if n == 0:
        return head + " -> no change (outside ignored boxes)"
    ys, xs = np.nonzero(mask)
    r0, c0, r1, c1 = max(0, int(ys.min()) - 1), max(0, int(xs.min()) - 1), min(64, int(ys.max()) + 2), min(64, int(xs.max()) + 2)
    lines = [head + f" -> {n} px changed in rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1}"]
    if t.action.type == "CLICK":
        lines.append(f"    click at ({t.action.row},{t.action.col}); change window centre ({(r0 + r1) // 2},{(c0 + c1) // 2})")
    # object-level summary
    bo = {o.id: o for o in t.before.objects}; ao = {o.id: o for o in t.after.objects}
    parts = []
    for oid, (dr, dc) in t.diff.moved[:6]:
        o = bo.get(oid)
        if o is not None:
            parts.append(f"obj#{oid}(colour {o.color}, {o.height}x{o.width}) moved ({dr:+d},{dc:+d})")
    for oid, old, new in t.diff.recolored[:4]:
        parts.append(f"obj#{oid} recoloured {old}->{new}")
    for o in t.diff.appeared[:4]:
        parts.append(f"obj#{o.id}(colour {o.color}) appeared at ({o.bbox[0]},{o.bbox[1]})")
    for o in t.diff.disappeared[:4]:
        parts.append(f"obj#{o.id}(colour {o.color}) disappeared from ({o.bbox[0]},{o.bbox[1]})")
    if parts:
        lines.append("    " + "; ".join(parts))
    if (r1 - r0) <= max_rows and (c1 - c0) <= max_cols:
        lines.append("    before:"); lines += ["      " + l for l in frame_text(b, (r0, r1), (c0, c1)).splitlines()]
        lines.append("    after:"); lines += ["      " + l for l in frame_text(a, (r0, r1), (c0, c1)).splitlines()]
        if t.intermediate_frames:
            lines.append(f"    ({len(t.intermediate_frames)} intermediate animation frames between them)")
    else:
        lines.append(f"    (change window {r1 - r0}x{c1 - c0} too large to print)")
    return "\n".join(lines)


def hypothesis_prompt(*, scene: Scene, grid: np.ndarray, log: list[Transition], available: list[Action], prior_signatures: str,
                      previous_code: Optional[str], counterexamples: list[str], level: int, ignore_boxes: list = (), max_transitions: int = 24,
                      rejected: list[str] = ()) -> list[dict]:
    buttons = sorted({a.id for a in available if a.type == "BUTTON"})
    acts = ("buttons " + ", ".join(f"ACTION{i}" for i in buttons) if buttons else "") + (" and " if buttons and any(a.type == "CLICK" for a in available) else "") + \
           ("clicks anywhere on the grid" if any(a.type == "CLICK" for a in available) else "")
    user = [f"# Level {level}. Available actions: {acts}.", "", "# Current board (hex digits = colours)", frame_text(grid), "",
            "# Objects the perception found (small marks included)", scene_objects_text(scene), ""]
    if log:
        recent = log[-max_transitions:]
        user += [f"# What actions did so far ({len(log)} actions on this level; last {len(recent)} shown, oldest first)"]
        user += [transition_text(t, ignore_boxes) for t in recent]
        user.append("")
    else:
        user += ["# No action has been taken yet on this level: propose the hypothesis from the board alone and the first tests.", ""]
    if counterexamples:
        user += ["# Counter-examples: where the previous hypothesis predicted wrongly (predicted vs real)"] + counterexamples + [""]
    if rejected:
        user += ["# Rejected earlier (do not repeat): " + "; ".join(rejected)]
    if previous_code:
        user += ["# Previous hypothesis code (revise it; keep what verified)", "```python", previous_code, "```", ""]
    if prior_signatures:
        user += ["# Mechanism library you may call (signatures)", prior_signatures, ""]
    user.append("Write the hypothesis document and code now.")
    return [{"role": "system", "content": CONTRACT + "\n" + API_SUMMARY}, {"role": "user", "content": "\n".join(user)}]
