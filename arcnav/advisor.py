"""Layer 2: the model as a macro planner, not a per-step actor.

The explorer (layer 0) supplies structured evidence: which macro actions exist, what each kind of touch did on earlier
levels (effect table), how the earlier levels were won, and the current board. The model answers with a short ordered
list of macro labels (JSON). The harness executes them, reports what changed, and asks again — a handful of rounds per
level, tens of actions per round, no free-form per-step reasoning.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional

from .frame import summarize_diff

SYSTEM = """You plan moves for a grid puzzle game played by a program. You never act step by step: you return an ordered list of
macro actions (by their exact labels) that the program executes for you, then you see what changed and plan again.
Answer with JSON only: {"hypothesis": "<one sentence: what the level requires>", "plan": ["<label>", "<label>", ...]}.
Use only labels from the AVAILABLE MACROS list, at most 12 per answer. Prefer macros whose kind/colour did something on earlier
levels. If a plan made no progress, try a genuinely different hypothesis instead of repeating it."""


def _board_ascii(frame) -> str:
    return frame.ascii


def _objects_by_colour(frame, limit: int = 12) -> str:
    rows = []
    by: dict[int, list] = {}
    for n in frame.segmentation["nodes"]:
        if n["hud"]:
            continue
        by.setdefault(n["color"], []).append(n)
    for c, ns in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ns.sort(key=lambda n: n["pixels"])
        pos = ", ".join(f"({n['center'][0]},{n['center'][1]}){'' if n['pixels'] == ns[0]['pixels'] else f'[{n['pixels']}px]'}" for n in ns[:limit])
        rows.append(f"  colour {c}: {len(ns)} object(s), sizes {sorted(set(n['pixels'] for n in ns))[:6]}, at {pos}{' ...' if len(ns) > limit else ''}")
    return "\n".join(rows)


def build_brief(explorer, macros: list) -> str:
    s = explorer.s
    parts = [f"GAME: level {s.level} of {s.levels_total}; valid actions {s.valid_actions}; actions used so far {s.actions_used}."]
    if explorer.solutions:
        for lvl, path in sorted(explorer.solutions.items()):
            parts.append(f"LEVEL {lvl} WAS WON by this macro sequence: {path[-16:]}")
    if s.level_recaps:
        parts.append("HARNESS RECAP OF THE LAST WIN: " + s.level_recaps[-1][:700])
    if s.goal_hypotheses:
        parts.append("GOAL HYPOTHESES FROM EARLIER LEVELS: " + " | ".join(h["text"][:160] for h in s.goal_hypotheses[:3]))
    if explorer.effect_stats:
        eff = sorted(((k, v) for k, v in explorer.effect_stats.items() if v[0]), key=lambda kv: -kv[1][1] / kv[1][0])
        parts.append("EFFECT TABLE (what touching each colour did so far; 1.0 = always changed the board, 0 = never): " +
                     ", ".join(f"{k[0]} colour {k[1]}: {v[1] / v[0]:.2f} over {v[0] // 2} tries" for k, v in eff[:14]))
    if getattr(s, "rules_text", ""):
        parts.append("INDUCED RULES: " + s.rules_text[:600])
    parts.append("CURRENT BOARD OBJECTS (row,col centres):\n" + _objects_by_colour(s.frame))
    parts.append("CURRENT BOARD (64x64, one hex digit per cell, HUD included):\n" + _board_ascii(s.frame))
    parts.append("AVAILABLE MACROS (use these exact labels):\n  " + "\n  ".join(m.label for m in macros[:60]))
    return "\n\n".join(parts)


def parse_plan(text: str) -> tuple[str, list[str]]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return "", []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        # tolerate trailing commas / single quotes
        try:
            obj = json.loads(re.sub(r",\s*([\]}])", r"\1", m.group(0)).replace("'", '"'))
        except Exception:
            return "", []
    plan = obj.get("plan") or []
    return str(obj.get("hypothesis", ""))[:300], [str(p) for p in plan][:12]


def advise_level(explorer, client, *, rounds: int = 6, max_actions: int = 150, log=print) -> dict:
    """Run up to `rounds` plan→execute→report cycles on the current level. Returns a summary dict."""
    s = explorer.s
    level0 = s.level; a0 = s.actions_used
    messages = [{"role": "system", "content": SYSTEM}]
    history: list[str] = []
    for rnd in range(rounds):
        if s.level != level0 or s.state == "WIN" or s.actions_used - a0 >= max_actions:
            break
        node = explorer.node()
        macros = explorer.macros(node)
        by_label = {m.label: m for m in macros}
        brief = build_brief(explorer, macros)
        if history:
            brief = "WHAT YOUR EARLIER PLANS DID ON THIS LEVEL:\n" + "\n".join(history[-6:]) + "\n\n" + brief
        messages = [messages[0], {"role": "user", "content": brief}]   # one fresh brief per round: the state, not the chat, is the memory
        try:
            r = client.chat(messages, tools=None, tool_choice="none")
        except Exception as e:
            log(f"[advisor] model call failed: {e!r}"); break
        hyp, plan = parse_plan(r.message.get("content", ""))
        log(f"[advisor] round {rnd + 1}: hypothesis={hyp!r} plan={plan}")
        explorer.trace.append(f"advisor r{rnd + 1}: {hyp[:80]} -> {plan}")
        if not plan:
            history.append(f"round {rnd + 1}: no valid plan returned"); continue
        outcomes = []
        for label in plan:
            m = by_label.get(label)
            if m is None:
                # labels go stale as the board changes: re-resolve by kind+colour when possible
                macros = explorer.macros(explorer.node()); by_label = {x.label: x for x in macros}
                m = by_label.get(label)
                if m is None:
                    col = explorer._macro_color_label(label) if hasattr(explorer, "_macro_color_label") else None
                    cands = [x for x in macros if x.kind == label.split("(")[0] and (col is None or explorer._macro_color(x) == col)]
                    m = cands[0] if cands else None
            if m is None:
                outcomes.append(f"{label}: not available now"); continue
            before = s.frame
            res = explorer._run_macro(explorer.node(), m)
            d = summarize_diff(before, s.frame) if s.frame is not None else {}
            outcomes.append(f"{label}: {d.get('changed_cells', 0)} cells changed" +
                            (f", vanished {[(x['color'], x['pixels']) for x in d.get('disappeared', [])][:3]}" if d.get("disappeared") else "") +
                            (f", appeared {[(x['color'], x['pixels']) for x in d.get('appeared', [])][:3]}" if d.get("appeared") else "") +
                            (" -> LEVEL COMPLETED" if res["level_completed"] else " -> GAME OVER (level reset)" if res["game_over"] else ""))
            if res["level_completed"] or res["game_over"] or s.actions_used - a0 >= max_actions:
                break
        history.append(f"round {rnd + 1} (hypothesis: {hyp[:120]}): " + "; ".join(outcomes))
        if s.level > level0:
            explorer.solutions[level0] = [f"advisor:{hyp[:60]}"] + plan
            return {"completed": True, "rounds": rnd + 1, "actions": s.actions_used - a0, "hypothesis": hyp}
    return {"completed": s.level > level0, "rounds": rounds, "actions": s.actions_used - a0}
