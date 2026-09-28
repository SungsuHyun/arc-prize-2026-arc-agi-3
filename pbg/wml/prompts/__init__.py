"""wml/prompts — prompt builders for the World Model Lab and Goal Inference (spec §14).

System prompt = core/types.py + core/contracts.py verbatim + the rule-writing contract; user prompt = prior signatures,
the current model code (violated rules marked), selected observation summaries, and the instruction."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

CORE = Path(__file__).resolve().parents[2] / "core"

CONTRACT = """You write a WORLD MODEL for an unknown pixel board game as executable Python.
Hard rules:
- Define `def build_model() -> RuleModel`. Build it from a list of `Rule(name, applies, effect, confidence)` objects and a
  role-assignment function: `RuleModel(rules, role_fn, default="unknown")`.
- Rules reference objects by ROLE (agent, wall, target, key, door, pushable, collectible, hazard, indicator, decoration,
  or custom:<name>), never by object id, never by absolute coordinates. Read every constant (colours, step sizes,
  regions) from the Scene when you can; the role function decides which object is which.
- The role function `role_fn(scene) -> dict[obj_id, role]` may use colour, shape, size, region and position relative
  to regions (e.g. 'the object that moved' is not available: infer from appearance).
- Prefer the prior mechanisms listed below (already imported: call them by name). Effects compose in rule order:
  a move rule first, then collision/cancel rules, then counters. `scene.aux['_before']` holds the pre-action scene.
- Return UNKNOWN (an effect returning None, or `unknown_for(buttons)`) for transitions you cannot explain; never
  invent behaviour to force coverage.
- Rules for ui_strip regions (gauges/counters) are optional: evaluation ignores objects in ui_strip regions and objects
  whose role is 'indicator'. Do not write rules for them.
- Region ids (R0, R1, ...) are re-numbered every frame: identify a region by kind_hint, bg colour and size, never by id.
  Object ids are tracking ids and can change when an object reshapes: identify objects by colour/shape/region.
- Every button/click the log shows must get a prediction from your rules (an unexplained action = UNKNOWN only for
  that action). A catch-all UNKNOWN rule, if any, must come last and must not swallow explained actions.
- Prediction is checked object by object (colour, bbox, shape) against the real next frame, so displacements and
  cancel conditions must be exact (step size in pixels, blocked by which colours/roles, board bounds).
- No imports except numpy, math, itertools, dataclasses, typing, collections. No file/network access. Do not compare
  game ids. Do not hard-code lists of coordinates.
- CODE ONLY: no analysis, no reasoning, no long docstrings or comments (at most one short comment per rule). Work out the
  mechanics silently and write the final rules. Keep the whole reply under 120 lines of code.
- Do not model gauges / counters / timers in ui_strip regions: evaluation ignores them and they waste rules.
Reply with exactly ONE ```python code block (the model) and optionally ONE ```json block {"notes": "..."}.

# Worked example (a different game: a blue 4x4 agent moves 4 px per button on a black board, grey blocks are walls,
# red 2x2 pieces vanish when the agent steps on them, button 5 does nothing). Adapt the pattern, never the numbers.
```python
def role_fn(scene):
    roles = {}
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    for o in scene.objects:
        if o.region in strips:
            roles[o.id] = "indicator"
        elif o.color == 1 and o.area >= 12:
            roles[o.id] = "agent"
        elif o.color == 5:
            roles[o.id] = "wall"
        elif o.color == 2:
            roles[o.id] = "collectible"
        else:
            roles[o.id] = "decoration"
    return roles

def build_model():
    dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    move_buttons = lambda s, a: a.type == "BUTTON" and a.id in dirs
    rules = [
        Rule("move_agent", move_buttons, move_role("agent", dirs, step=4)),
        Rule("wall_blocks", move_buttons, cancel_move_if_overlap("agent", "wall")),
        Rule("stay_on_board", move_buttons, cancel_move_if_off_floor("agent", (0,))),   # floor colour read from the scene
        Rule("collect", move_buttons, collect_on_overlap("agent", "collectible")),
        Rule("noop_button", lambda s, a: a.type == "BUTTON" and a.id == 5, noop_for((5,))),
    ]
    return RuleModel(rules, role_fn, default="unknown")
```"""


def core_sources() -> str:
    parts = []
    for name in ("types.py", "contracts.py"):
        p = CORE / name
        if p.exists():
            parts.append(f"# ==== core/{name} ====\n" + p.read_text())
    return "\n\n".join(parts)


def system_prompt() -> str:
    return CONTRACT + "\n\n# Reference: the data types and interfaces (verbatim)\n" + core_sources()


def world_model_prompt(*, signatures: str, current_code: Optional[str], violated_rules: list[str], observations: list[str],
                       semantics_text: str, scene_text: str, hidden_state_hint: bool = False, level_note: str = "",
                       diagnosis: Optional[str] = None) -> list[dict]:
    user = ["# Prior mechanisms (call by name; signatures only)", signatures, ""]
    user += ["# Action semantics measured by the probe", semantics_text, ""]
    user += ["# Current scene (regions and objects)", scene_text, ""]
    if current_code:
        code = current_code
        for r in violated_rules:
            code = code.replace(f'"{r}"', f'"{r}"  # VIOLATED').replace(f"'{r}'", f"'{r}'  # VIOLATED")
        user += ["# Current model code (rules marked # VIOLATED failed on the log)", "```python", code, "```", ""]
    user += ["# Observations (action -> object-level diff), newest last", *[f"- {o}" for o in observations], ""]
    if level_note:
        user += ["# Level note", level_note, ""]
    if diagnosis:
        user += ["# Diagnosis of the last violation", diagnosis, ""]
    if hidden_state_hint:
        user += ["# Hidden state", "The same scene and action gave different results more than once while the environment is deterministic. "
                 "Introduce state variables in scene.aux (e.g. the last k actions, the selected object, a counter) and update them in your rules.", ""]
    if current_code:
        user += ["# Instruction", "Make the MINIMAL change that explains the violations: fix only the violated rules or add a new rule. "
                 "Do not create new constants; read them from the Scene. Return UNKNOWN for transitions you cannot explain."]
    else:
        user += ["# Instruction", "Write a rule-based model that reproduces every observation above. Assign roles from appearance. "
                 "Return UNKNOWN for transitions you cannot explain."]
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": "\n".join(user)}]


def goal_template_prompt(scene_text: str, observations: list[str]) -> list[dict]:
    sys = ("You propose the WIN CONDITION of an unknown pixel board game as executable Python. Define `def build_goal()` returning an "
           "object with `name: str`, `is_goal(scene) -> bool` and `progress(scene) -> float in [0,1]`. Use roles/colours/regions read from the "
           "Scene, no coordinates. Reply with exactly ONE ```python block.\n\n" + core_sources())
    user = "# Scene\n" + scene_text + "\n\n# Recent observations\n" + "\n".join(f"- {o}" for o in observations) + \
        "\n\nNo goal template matched this scene. Propose the most plausible win condition as code."
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def diagnose_prompt(rule_code: str, violation_summaries: list[str], scene_text: str) -> list[dict]:
    sys = "You diagnose why a rule of a pixel-board-game world model failed after a level change. Reply with ONE ```json block {\"rule\": ..., \"cause\": ..., \"fix_hint\": ...}."
    user = "# Violated rule\n```python\n" + rule_code + "\n```\n\n# Violating transitions\n" + "\n".join(f"- {v}" for v in violation_summaries) + "\n\n# Scene\n" + scene_text
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def critique_prompt(previous: list[dict], score: float, coverage: float, mismatches: list[str], unknowns: list[str]) -> list[dict]:
    lines = [f"Your model was verified on the whole log: accuracy {score:.2f}, coverage {coverage:.2f}."]
    if mismatches:
        lines += ["Wrong predictions (predicted vs observed):", *[f"- {m}" for m in mismatches[:6]]]
    if unknowns:
        lines += ["UNKNOWN (no prediction) for transitions such as:", *[f"- {u}" for u in unknowns[:4]]]
    lines.append("Fix the rules so these transitions are predicted exactly; keep what already works. Return the corrected full code in ONE ```python block.")
    return previous + [{"role": "user", "content": "\n".join(lines)}]


def repair_prompt(previous: list[dict], error: str) -> list[dict]:
    return previous + [{"role": "user", "content": f"Your code failed to load or run:\n```\n{error[-1200:]}\n```\nReturn the corrected full code in ONE ```python block."}]
