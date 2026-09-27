"""Summarize a Duck run that used the advisor ensemble (ARC3_ADVISORS).

Usage: .venv/bin/python scripts/duck_advisor_report.py vendor/duck-runs/<run_dir> [...]

Prints per game: final score/levels/actions (from stdout.log), advisor turns,
advisor wall time per turn (mean / p50 / max), failures, output tokens, and how
often the actions executed after a turn started with an advisor's first
recommendation (agreement).  Advisor logs are artifacts/<game>_p<N>_tool_runtime_state_advisors.jsonl.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

_ACTION_RE = re.compile(r"(MOUSE\s*\(\s*\d+\s*,\s*\d+\s*\)|ACTION\d|UP|DOWN|LEFT|RIGHT|MOUSE|SPACE|ENTER|RESET)", re.I)


def _first_recommendation(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().upper().startswith("RECOMMEND"):
            m = _ACTION_RE.search(line.split(":", 1)[-1])
            return m.group(1).upper().replace(" ", "") if m else None
    return None


def _norm(action: str) -> str:
    a = action.upper().replace(" ", "")
    return "MOUSE" if a.startswith("MOUSE") else a


def _game_scores(stdout_log: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not stdout_log.exists():
        return out
    for line in stdout_log.read_text(errors="replace").splitlines():
        m = re.match(r"\s+([a-z0-9]+-[0-9a-f]+): (score=.*)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def report(run_dir: Path) -> None:
    print(f"== {run_dir.name}")
    scores = _game_scores(run_dir / "stdout.log")
    logs = sorted(run_dir.rglob("*_advisors.jsonl"))
    if not logs:
        print("  (no *_advisors.jsonl found)")
    for log in logs:
        game = next((g for g in scores if g.split("-")[0] in str(log)), log.parent.name)
        records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        walls = [r["wall_s"] for r in records]
        ops = [o for r in records for o in r["opinions"]]
        fails = sum(1 for o in ops if not o["ok"])
        out_tok = [o["output_tokens"] for o in ops if o.get("output_tokens")]
        agree = total = 0
        for prev, nxt in zip(records, records[1:]):
            executed = [_norm(a) for a in nxt.get("executed_since_last") or []]
            recs = [_first_recommendation(o["text"]) for o in prev["opinions"] if o["ok"]]
            recs = [_norm(r) for r in recs if r]
            if executed and recs:
                total += 1
                agree += int(executed[0] in recs)
        print(f"  {game}: {scores.get(game, '(score n/a)')}")
        print(f"    advisor turns={len(records)} wall/turn mean={statistics.mean(walls):.1f}s "
              f"p50={statistics.median(walls):.1f}s max={max(walls):.1f}s total={sum(walls)/60:.1f}min"
              if walls else "    advisor turns=0")
        print(f"    opinions={len(ops)} failed={fails} out_tokens/opinion={statistics.mean(out_tok):.0f}"
              if out_tok else f"    opinions={len(ops)} failed={fails}")
        if total:
            print(f"    first executed action matched an advisor's first recommendation: {agree}/{total} turns")
    for game, s in scores.items():
        if not any(game.split("-")[0] in str(l) for l in logs):
            print(f"  {game}: {s} (no advisor log)")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["."]:
        report(Path(arg))
