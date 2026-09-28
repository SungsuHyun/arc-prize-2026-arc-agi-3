#!/usr/bin/env python
"""tools/lint_no_game_id.py — CI lint (spec §13): no module may branch on a game id string.

Flags string literals that look like game ids (two letters + two digits, e.g. 'ls20') used in comparisons, `in` tests,
dict keys or conditionals anywhere under pbg/ (tests and data files excluded).

    .venv/bin/python pbg/tools/lint_no_game_id.py [paths...]   -> exit 1 on findings"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

GAME_ID = re.compile(r"^[a-z]{2}\d{2}$")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = [ROOT / "pbg"]
EXCLUDE = {"tests", "data", "tools"}


def check_file(path: Path) -> list[str]:
    out = []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return [f"{path}: syntax error {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Compare, ast.If, ast.IfExp, ast.Match if hasattr(ast, "Match") else ast.If, ast.Dict, ast.Subscript)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and GAME_ID.match(sub.value):
                    out.append(f"{path.relative_to(ROOT)}:{sub.lineno}: game id literal '{sub.value}' in a branch/lookup")
                    break
    return out


def main(argv=None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])] or DEFAULT
    findings = []
    for base in paths:
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for f in files:
            if any(part in EXCLUDE for part in f.relative_to(ROOT).parts):
                continue
            findings += check_file(f)
    for line in findings:
        print(line)
    print(f"no-game-id-branch: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
