"""memory/promote.py — prior promotion (spec §11): a verified rule reused in >= 2 games becomes a prior after game-specific
constants are parameterised. The static filter rejects colour-index and coordinate hard-coding."""
from __future__ import annotations

import re

COLOR_LITERAL = re.compile(r"\b(color|colour)\s*==\s*\d+")
COORD_LITERAL = re.compile(r"\(\s*\d{1,2}\s*,\s*\d{1,2}\s*\)")


def parameterise(code: str) -> tuple[str, list[str]]:
    """Replace literal colours / coordinates with named parameters; returns (code, parameter names)."""
    params: list[str] = []
    def sub_color(m):
        params.append(f"COLOR_{len(params)}")
        return f"{m.group(1)} == {params[-1]}"
    code = COLOR_LITERAL.sub(sub_color, code)
    if COORD_LITERAL.search(code):
        params.append("COORDS")
    return code, params


def eligible(rule_meta: dict) -> bool:
    return len(set(rule_meta.get("games_verified", []))) >= 2 and rule_meta.get("confidence", 0) >= 1.0
