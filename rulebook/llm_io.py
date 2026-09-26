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
[?] untested, [X] refuted), the board, and the CANDIDATE actions with the program's deterministic prediction of what each will do
('never'/'unknown' = untested). Pick ONE candidate label. Prefer: (1) an action the plan calls for that makes progress toward the win
condition, (2) an action that tests an unverified rule or reveals an unknown effect cheaply, (3) never a candidate marked [tried here]
or predicted to change nothing unless the rulebook explains why it would differ now. Actions cost score, so no wandering.
Answer with JSON only: {"choice": "<exact label>", "expect": "<what you expect, one line>", "edits": [<optional rulebook edits>]}
Edit ops: {"op":"add","section":"env|rules|win","text":..,"kind":..,"params":{}} {"op":"confirm|refute|remove","id":"R3","note":".."}
{"op":"edit","id":"R3","text":..}. Keep edits rare and factual."""

REVIEW_SYSTEM = """You maintain the RULEBOOK of a program playing a grid puzzle game. Something did not go as the rulebook predicted, or a
level ended. Revise the rulebook so that it explains ALL recorded evidence: refute or edit wrong entries, add the rule that explains the
surprise (machine-checkable kind when possible), update the win condition and the plan. Program-verified entries (source harness, with
for/against counts) are facts; do not contradict them. Be concrete and short.
""" + SCHEMA_HELP + """
Answer with JSON only: {"edits": [ {"op":"add","section":"rules","text":"..","kind":"..","params":{}}, {"op":"refute","id":"R2","note":".."},
{"op":"edit","id":"W1","text":".."}, {"op":"remove","id":"E4"} ], "plan": "<updated plan>"}"""


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
                f"OBJECTS (by colour; (row,col) centres):\n{objects_text(f)}\n\nBOARD (64x64 hex digits, row 0 at the top):\n{f.ascii}\n\n"
                "Write the initial rulebook (hypotheses) and the plan.")
        return self._call("init", INIT_SYSTEM, user, think=think)

    def decide(self, game, book: Rulebook, cands: list, outcomes: list[str], budget_text: str) -> Optional[dict]:
        f = game.frame
        user = (f"{book.render()}\n\nGAME: level {game.level} of {game.levels_total or '?'}; {budget_text}; valid actions {game.valid_actions}.\n"
                + (("RECENT ACTIONS (newest last):\n  " + "\n  ".join(outcomes[-10:]) + "\n") if outcomes else "")
                + f"\nOBJECTS:\n{objects_text(f)}\n\nBOARD:\n{f.ascii}\n\nCANDIDATES (label -> program prediction):\n  "
                + "\n  ".join(c.line() for c in cands[:40]) + "\n\nChoose one candidate label.")
        return self._call("decide", DECIDE_SYSTEM, user, think=False)

    def review(self, game, book: Rulebook, event: str, evidence_changes: list[str], outcomes: list[str], *, before: Optional[Frame], bbox=None, think: bool = True) -> Optional[dict]:
        f = game.frame
        parts = [book.render(), f"GAME: level {game.level} of {game.levels_total or '?'}; valid actions {game.valid_actions}.", "EVENT: " + event]
        if evidence_changes:
            parts.append("PROGRAM-VERIFIED CHANGES TO THE RULEBOOK (facts):\n  " + "\n  ".join(evidence_changes[-12:]))
        if outcomes:
            parts.append("RECENT ACTIONS (newest last):\n  " + "\n  ".join(outcomes[-12:]))
        if before is not None and bbox is not None:
            parts.append("BEFORE (changed region):\n" + crop(before, bbox) + "\nAFTER (same region):\n" + crop(f, bbox))
        parts.append(f"OBJECTS NOW:\n{objects_text(f)}\n\nBOARD NOW:\n{f.ascii}")
        parts.append("Revise the rulebook (edits) and the plan.")
        return self._call("review", REVIEW_SYSTEM, "\n\n".join(parts), think=think)
