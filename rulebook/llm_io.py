"""Model I/O: three prompt types (INIT rulebook / DECIDE action / REVIEW rulebook) and tolerant JSON parsing.
Every call sends a fresh, complete brief — the rulebook and the record are the memory, not the chat history."""
from __future__ import annotations

import json
import re
from typing import Optional

from arcnav.frame import Frame
from arcnav.llm import ChatClient

from .book import Rulebook

KEY_HELP = ("UP/DOWN/LEFT/RIGHT/SPACE/ACTION7 are keys (their meaning is unknown until tested; SPACE and ACTION7 are often 'interact' "
            "or 'rotate'); MOUSE clicks a cell (row, col). Each level restarts on GAME OVER. The score per level is (baseline_actions/"
            "your_actions)^2, so the fewest actions win; an unfinished level scores 0.")

SCHEMA_HELP = """Entry format: {"text": "<one sentence>", "kind": "<kind>", "params": {...}}. Use a machine-checkable kind whenever it fits so
the program can verify the entry against every recorded action (otherwise kind "other"):
  rules: move {"action":"UP","dx":0,"dy":-4}  (dx = columns, dy = rows, in pixels; negative dy = up)
         wall {"color":k}  noop {"action":"SPACE"}  collect {"color":k}  hazard {"color":k}  gauge {"color":k,"per_action":-1}
         refill {"color":k,"amount":n}  click {"color":k}  (say in text what clicking that colour does)  other {}
  win:   reach {"reach":k}  collect_all {"collect":k}  collect_reach {"collect":k,"reach":k2}  click_sequence {"sequence":[...]}  other {}
  env:   kind "other" (avatar colour, walls, HUD/counters, distinct objects and their likely roles)."""

INIT_SYSTEM = """You write the initial RULEBOOK for a program that plays an unknown 64x64 grid puzzle game (ARC-AGI-3). Colours are 0-15,
shown as hex digits (row 0 at the top). Nobody knows the rules: write them as HYPOTHESES to be tested by the program, which will
confirm or refute each one from real actions and ask you to revise. Be concrete (colours, positions, key names) and short.
""" + KEY_HELP + "\n" + SCHEMA_HELP + """
Answer with JSON only:
{"env": [entry, ...], "rules": [entry, ...], "win": [entry, ...], "plan": "<what to test first, then how to win>"}
At most 8 env, 10 rules, 4 win entries."""

DECIDE_SYSTEM = """You choose the next action for a program playing a grid puzzle game. You see the RULEBOOK (hypotheses; [OK] confirmed,
[?] untested, [X] refuted), the ENTITIES (areas P0.. = panels/floors; objects #id with colour, size, position, grouped by area), the
board, the WIN CONDITION PROGRESS the program evaluates on the current board, and the CANDIDATE actions with the program's
deterministic prediction of what each will do ('never'/'unknown' = untested). 'plan:...' candidates are multi-action programs the
harness built to make a win condition true; 'submit' presses the submit button. Pick ONE candidate label. Prefer: (1) a plan or
submit candidate whose prediction says a win condition will hold, (2) an action the plan calls for that makes progress, (3) an action
that tests an unverified rule or reveals an unknown effect cheaply, (4) never a candidate marked [tried here] or predicted to change
nothing unless the rulebook explains why it would differ now. Objects that did nothing twice are hidden. Actions cost score.
Answer with JSON only: {"choice": "<exact label>", "expect": "<what you expect, one line>", "roles": {"<#id or 'colour c in Pk'>": "<role>"},
"edits": [<optional rulebook edits>]}. 'roles' (optional) names what things are: button, submit, mark, template, piece, anchor, hole, frame, hud, wall.
Edit ops: {"op":"add","section":"env|rules|win","text":..,"kind":..,"params":{}} {"op":"confirm|refute|remove","id":"R3","note":".."}
{"op":"edit","id":"R3","text":..}. Keep edits rare and factual."""

REVIEW_SYSTEM = """You maintain the RULEBOOK of a program playing a grid puzzle game. Something did not go as the rulebook predicted, or a
level ended. Revise the rulebook so that it explains ALL recorded evidence: refute or edit wrong entries, add the rule that explains the
surprise (machine-checkable kind when possible), update the win condition and the plan. Program-verified entries (source harness, with
for/against counts) are facts; do not contradict them. Think in terms of OBJECT ROLES, not colours: the same colour can be an editable
mark in one area and a read-only template in another; small blobs can be anchors that move a piece; corner marks or dotted outlines are
frames/holes that a block or piece must fill; a button whose earlier clicks did nothing may be the submit button. Be concrete and short.
""" + SCHEMA_HELP + """
Answer with JSON only: {"roles": {"<#id or 'colour c in Pk'>": "<role>"}, "procedure": "<the steps that win a level, in terms of roles>",
"edits": [ {"op":"add","section":"rules","text":"..","kind":"..","params":{}}, {"op":"refute","id":"R2","note":".."},
{"op":"edit","id":"W1","text":".."}, {"op":"remove","id":"E4"} ], "plan": "<updated plan>"}"""


PYTHON_TOOL = {"type": "function", "function": {"name": "python", "description": "Run Python code against the game library. Use act()/click()/press() inside it to play; print what you need to see.",
                                                 "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to execute"}}, "required": ["code"]}}}

CODER_SYSTEM = """You play an unknown 64x64 grid puzzle game (colours 0-15 as hex digits, row 0 at the top) by writing Python that runs against a
game library. The score per level is (baseline_actions/your_actions)^2, so use few actions; unfinished levels score 0. Levels of one game
share rules, so what won level N is the plan for level N+1 (adapted to the new board).

Variables (rebuilt every call): board (64x64 ints), ascii, level, valid_actions, objects (dicts: id,color,size,bbox=(r0,c0,r1,c1),center=(r,c),
region,hud,cells), regions [(id,colour,bbox)], entities (text), transitions (last 60: action, before, after, level), book (the RULEBOOK text:
[OK] confirmed / [?] hypothesis / [X] refuted), goal (win-condition progress lines), candidates [(label, program prediction)],
plans [(label, prediction, actions)], plan_actions {label: actions}, notes (your own memory).
Functions: act(a) executes one action or a list — a = 'UP'|'DOWN'|'LEFT'|'RIGHT'|'SPACE'|'ACTION7' or (row, col) for a click; every action
is checked against the program's prediction and the result is printed; act stops at a level end / game over / 30 actions per call.
click(r,c), press(key, n), path_to(r,c) -> key list on the learned floor map, predict(a) -> what the program expects, scene() -> fresh board
and objects, check_rule(fn) scores fn(before_board, action)->after_board|None on every recorded transition (>=0.8 on >=4 makes it a
verified rule the predictor uses), define_goal(name, fn(objects, board)->bool, needs_submit=False) registers a win condition the program
evaluates and refutes automatically, note(text, section, kind, params) adds a rulebook entry, refute(id, why), set_plan(text), remember(text).

How to work: read book/goal/candidates/plans first. If a plan candidate says a win condition will hold, run it: act(plan_actions[label]).
Otherwise write the procedure as code (loops over objects, several actions per call, not one click at a time), verify a hypothesis about
an effect with check_rule before relying on it, and define the win condition with define_goal so the program can tell you when it holds.
Every call must execute at least one action (act/click/press) unless it defines a rule or goal. Never repeat an action that changed
nothing; candidates already tried in this state are marked. Clicks are act((row, col)) — coordinates from objects/candidates. Write ONE
python call per turn. Do your reasoning by ACTING, not by writing: at most two short comment lines, no analysis essays, no plans in
comments; a call that only prints or reads state is wasted (the board, objects and predictions are already in this message). Batch
several actions per call when the procedure is clear. No prose outside the tool call."""


def objects_text(frame: Frame, limit: int = 10) -> str:
    by: dict[int, list] = {}
    hud = []
    for n in frame.segmentation["nodes"]:
        (hud if n["hud"] else by.setdefault(n["color"], [])).append(n)
    rows = []
    for c, ns in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ns.sort(key=lambda n: n["pixels"])
        pos = ", ".join(f"({n['center'][0]},{n['center'][1]}){n['pixels']}px" for n in ns[:limit])
        rows.append(f"  colour {c}: {len(ns)} object(s) at {pos}{' ...' if len(ns) > limit else ''}")
    if hud:
        rows.append("  HUD strips (edge counters/gauges): " + ", ".join(f"colour {n['color']} {n['pixels']}px rows {n['bbox'][0]}-{n['bbox'][2]} cols {n['bbox'][1]}-{n['bbox'][3]}" for n in hud[:6]))
    return "\n".join(rows) or "  (no objects)"


def entities_text(frame: Frame) -> str:
    from .entities import build_scene, scene_text
    try:
        return scene_text(build_scene(frame))
    except Exception:
        return objects_text(frame)


def crop(frame: Frame, bbox, pad: int = 2) -> str:
    r0, c0, r1, c1 = bbox
    r0, c0, r1, c1 = max(0, r0 - pad), max(0, c0 - pad), min(63, r1 + pad), min(63, c1 + pad)
    lines = frame.ascii.splitlines()
    return f"rows {r0}-{r1}, cols {c0}-{c1}:\n" + "\n".join(f"{r:2d} {lines[r][c0:c1 + 1]}" for r in range(r0, r1 + 1))


def parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    body = m.group(1) if m else None
    if body is None:
        i = text.find("{")
        if i < 0:
            return None
        depth = 0; end = None
        for j, ch in enumerate(text[i:], i):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j; break
        body = text[i:end + 1] if end else text[i:]
    for cand in (body, re.sub(r",\s*([\]}])", r"\1", body), re.sub(r",\s*([\]}])", r"\1", body).replace("'", '"')):
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


class Model:
    def __init__(self, client: ChatClient, *, think_tokens: int = 12000, decide_tokens: int = 1500, review_tokens: int = 6000, log=print):
        self.client, self.think_tokens, self.decide_tokens, self.review_tokens, self.log = client, think_tokens, decide_tokens, review_tokens, log
        self.calls = {"init": 0, "decide": 0, "review": 0, "failed": 0}
        self.seconds = 0.0

    def _call(self, kind: str, system: str, user: str, *, think: bool) -> Optional[dict]:
        self.calls[kind] += 1
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt, th in enumerate([think, False] if think else [False, False]):
            budget = (self.think_tokens if kind == "init" else self.review_tokens) if th else (self.decide_tokens if kind == "decide" else 4000)
            override = {"chat_template_kwargs": {"enable_thinking": th}, "max_tokens": budget}
            if attempt == 1 and not think:   # the first no-think answer rambled past the budget: demand the bare object
                msgs = msgs[:2] + [{"role": "user", "content": "Reply with ONLY the JSON object on one line. No explanation."}]
            try:
                r = self.client.chat(msgs, tools=None, tool_choice="none", override=override)
            except Exception as e:
                self.log(f"[model:{kind}] call failed: {e!r}"); self.calls["failed"] += 1
                return None
            self.seconds += r.latency
            content = r.message.get("content", "") or ""
            obj = parse_json(content)
            self.log(f"[model:{kind}] think={th} {r.latency:.0f}s finish={r.finish_reason} out={len(content)} chars reasoning={len(r.reasoning)} chars"
                     + ("" if obj else " — NO JSON") + (f"\n  reasoning: {r.reasoning[:600]!r}" if r.reasoning else "") + f"\n  answer: {content[:1200]}")
            if obj is not None:
                return obj
            if attempt == 0:
                self.log(f"[model:{kind}] retry " + ("without thinking" if think else "with a JSON-only reminder"))
        self.calls["failed"] += 1
        return None

    # ── prompts ──────────────────────────────────────────────────────────
    def init_rulebook(self, game, think: bool = True) -> Optional[dict]:
        f = game.frame
        user = (f"GAME: {game.levels_total or '?'} levels; level {game.level}. Valid actions now: {game.valid_actions}.\n"
                f"ENTITIES (areas P0.. = panels/floors/walls; objects #id colour size @(row,col), grouped by area):\n{entities_text(f)}\n\n"
                f"BOARD (64x64 hex digits, row 0 at the top):\n{f.ascii}\n\n"
                "Write the initial rulebook (hypotheses) and the plan. Name the role of each area and object group (env entries).")
        return self._call("init", INIT_SYSTEM, user, think=think)

    def code_turn(self, game, book, cands: list, plans: list, goal: dict, outcomes: list[str], budget_text: str, notes: str, prev_outputs: list, nudge: str = "") -> tuple:
        """One coder-mode turn: returns (code, text). Code comes from the python tool call, or from a ```python block in the content."""
        f = game.frame
        parts = [book.render(), f"GAME: level {game.level} of {game.levels_total or '?'}; {budget_text}; valid actions {game.valid_actions}."]
        if goal and goal.get("lines"):
            parts.append("WIN CONDITION PROGRESS (program-evaluated):\n  " + "\n  ".join(goal["lines"]))
        if nudge:
            parts.append("NOTE: " + nudge)
        if notes:
            parts.append("YOUR NOTES:\n" + notes)
        if prev_outputs:
            parts.append("YOUR LAST CALLS AND THEIR OUTPUT:\n" + "\n".join(prev_outputs[-2:]))
        if outcomes:
            parts.append("RECENT ACTIONS (newest last):\n  " + "\n  ".join(outcomes[-8:]))
        parts.append(f"ENTITIES (areas P0.. and the objects in each; #id colour size @(row,col)):\n{entities_text(f)}")
        parts.append(f"BOARD:\n{f.ascii}")
        parts.append("PLAN CANDIDATES (program-made; run with act(plan_actions[label])):\n  " + ("\n  ".join(f"{c.label} -> {c.prediction[:200]}" for c in plans if c.kind == "plan") or "(none)"))
        parts.append("CANDIDATE ACTIONS (label -> program prediction):\n  " + "\n  ".join(c.line() for c in cands[:32] if c.kind != "info"))
        parts.append("Write the python call for this turn.")
        user = "\n\n".join(parts)
        self.calls["decide"] += 1
        msgs = [{"role": "system", "content": CODER_SYSTEM}, {"role": "user", "content": user}]
        try:
            r = self.client.chat(msgs, tools=[PYTHON_TOOL], tool_choice="auto", override={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1400})
        except Exception as e:
            self.log(f"[model:code] call failed: {e!r}"); self.calls["failed"] += 1; return "", ""
        self.seconds += r.latency
        msg = r.message; content = msg.get("content") or ""
        code = ""
        for call in msg.get("tool_calls") or []:
            raw = call.get("function", {}).get("arguments") or ""
            try:
                args = json.loads(raw or "{}")
                if isinstance(args, dict) and args.get("code"):
                    code = str(args["code"]); break
                if isinstance(args, str) and args.strip():
                    code = args; break
            except Exception:
                m = re.search(r'"code"\s*:\s*"(.*)"\s*}?\s*$', raw, re.S)   # unterminated / badly escaped JSON: salvage the code string
                if m:
                    code = m.group(1).encode().decode("unicode_escape", errors="ignore"); break
                if any(k in raw for k in ("act(", "click(", "press(", "print(")):
                    code = raw; break
                self.log(f"[model:code] unparsable tool arguments: {raw[:200]!r}")
        if not code:
            m = re.search(r"```(?:python)?\s*(.*?)```", content, re.S)
            if m:
                code = m.group(1)
            elif any(k in content for k in ("act(", "click(", "press(", "print(")) and not content.strip().startswith("{"):
                code = content
        self.log(f"[model:code] {r.latency:.0f}s finish={r.finish_reason} tool_calls={len(msg.get('tool_calls') or [])} code={len(code)} chars content={len(content)} chars")
        if not code:
            self.calls["failed"] += 1
        return code, content

    def decide(self, game, book: Rulebook, cands: list, outcomes: list[str], budget_text: str, extra: str = "") -> Optional[dict]:
        f = game.frame
        user = (f"{book.render()}\n\nGAME: level {game.level} of {game.levels_total or '?'}; {budget_text}; valid actions {game.valid_actions}.\n"
                + (extra + "\n" if extra else "")
                + (("RECENT ACTIONS (newest last):\n  " + "\n  ".join(outcomes[-10:]) + "\n") if outcomes else "")
                + f"\nENTITIES (areas P0.. and the objects in each; #id colour size @(row,col)):\n{entities_text(f)}\n\nBOARD:\n{f.ascii}\n\nCANDIDATES (label -> program prediction):\n  "
                + "\n  ".join(c.line() for c in cands[:40]) + "\n\nChoose one candidate label.")
        return self._call("decide", DECIDE_SYSTEM, user, think=False)

    def review(self, game, book: Rulebook, event: str, evidence_changes: list[str], outcomes: list[str], *, before: Optional[Frame], bbox=None, think: bool = True, extra: str = "") -> Optional[dict]:
        f = game.frame
        parts = [book.render(), f"GAME: level {game.level} of {game.levels_total or '?'}; valid actions {game.valid_actions}.", "EVENT: " + event]
        if extra:
            parts.append(extra)
        if evidence_changes:
            parts.append("PROGRAM-VERIFIED CHANGES TO THE RULEBOOK (facts):\n  " + "\n  ".join(evidence_changes[-12:]))
        if outcomes:
            parts.append("RECENT ACTIONS (newest last):\n  " + "\n  ".join(outcomes[-12:]))
        if before is not None and bbox is not None:
            parts.append("BEFORE (changed region):\n" + crop(before, bbox) + "\nAFTER (same region):\n" + crop(f, bbox))
        parts.append(f"ENTITIES NOW (areas P0.. and the objects in each):\n{entities_text(f)}\n\nBOARD NOW:\n{f.ascii}")
        parts.append("Revise the rulebook (edits) and the plan.")
        return self._call("review", REVIEW_SYSTEM, "\n\n".join(parts), think=think)
