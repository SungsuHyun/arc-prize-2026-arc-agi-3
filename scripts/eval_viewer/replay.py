"""Rebuild the per-step boards of a recorded rulebook run.

A run leaves `run-<id>.json` (summary) and `logs/<id>/<game>.log` (one tagged line per action plus the agent's
reasoning). Boards are not stored, but the games are deterministic: replaying the recorded action sequence against
the offline engine reproduces every frame (verified: same level count, ~1 ms per step). Newer runs also leave
`<game>.actions.jsonl` (written by rulebook.env.Game), which records resets explicitly and a hash of every frame;
when it exists it is the action source and the replay is checked against the hashes.

    steps = load_steps(run_id, game_id)          # cached in results/cache/<run>/<game>.json.gz
    level_steps = [s for s in steps if s["level"] == 2]
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "experiments" / "rulebook" / "results"
CACHE = RESULTS / "cache"
CACHE_VERSION = 3

TAG_RE = re.compile(r"^\[\s*(\d+)s a\s*(\d+) L(\d+)\] (.*)$")
ACTION_RE = re.compile(r"^(OK |MISMATCH|obs): (MOUSE\((\d+),(\d+)\)|[A-Z0-9]+) \[([^\]]*)\] predicted: (.*?) \| actual: (.*)$", re.S)
EVENT_KINDS = ("plan candidate", "plan plan", "plan ", "goal state", "rulebook <- evidence", "win facts", "essential winning sequence",
               "level ", "DECIDE", "REVIEW", "INIT", "CODE OUTPUT", "CODE", "PLAN", "win predicates refuted", "start", "result", "CRASH",
               "review cap", "goal inference")

_arcade = None
_arcade_lock = threading.RLock()
_mem: dict = {}


def _kind(text: str) -> str:
    for k in EVENT_KINDS:
        if text.startswith(k):
            return k.strip().replace(" <- evidence", "")
    return "other"


def parse_log(path: Path) -> list[dict]:
    """Group the log into events: each tagged line starts one, untagged lines continue the previous one."""
    events: list[dict] = []
    for ln in path.read_text(errors="replace").splitlines():
        m = TAG_RE.match(ln)
        if m:
            events.append({"t": int(m.group(1)), "a": int(m.group(2)), "L": int(m.group(3)), "text": m.group(4)})
        elif events:
            events[-1]["text"] += "\n" + ln
    return events


def _action_from_label(label: str, row: Optional[str], col: Optional[str]):
    if row is not None:
        return {"action": "MOUSE", "row": int(row), "col": int(col)}
    return label


def _actions_from_log(events: list[dict]) -> list[dict]:
    """[{a, action, tag, label, predicted, actual}] from the action lines. Resets are not logged; the replay infers them."""
    out = []
    for e in events:
        m = ACTION_RE.match(e["text"])
        if m:
            out.append({"a": e["a"], "t": e["t"], "action": _action_from_label(m.group(2), m.group(3), m.group(4)), "tag": m.group(1).strip(),
                        "label": m.group(5), "predicted": m.group(6).strip(), "actual": m.group(7).strip(), "reset": False})
    return out


def _actions_from_jsonl(path: Path, log_actions: list[dict]) -> list[dict]:
    by_step = {x["a"]: x for x in log_actions}
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r["kind"] == "reset":
            out.append({"a": r["step"], "t": r.get("t"), "reset": True, "hash": r.get("hash")})
        else:
            base = by_step.get(r["step"], {})
            out.append({"a": r["step"], "t": r.get("t"), "action": r["action"], "reset": False, "hash": r.get("hash"),
                        "tag": base.get("tag", ""), "label": base.get("label", ""), "predicted": base.get("predicted", ""), "actual": base.get("actual", "")})
    return out


def _arc():
    global _arcade
    with _arcade_lock:
        if _arcade is None:
            import logging
            logging.disable(logging.CRITICAL)
            from arcnav.runner import make_arcade
            _arcade = make_arcade()
        return _arcade


def _hex(frame) -> list[str]:
    return ["".join(f"{v & 15:x}" for v in row) for row in frame.grid]


def _diff(a, b) -> int:
    return sum(1 for ra, rb in zip(a.grid, b.grid) for x, y in zip(ra, rb) if x != y)


def replay(run_id: str, game_id: str) -> dict:
    """Replay one game of a run. Returns {"steps": [...], "levels": {...}, "notes": [...]}."""
    from rulebook.env import Game, action_label
    log_dir = RESULTS / "logs" / run_id
    log = log_dir / f"{game_id}.log"
    events = parse_log(log) if log.exists() else []
    log_actions = _actions_from_log(events)
    jsonl = log_dir / f"{game_id}.actions.jsonl"
    actions = _actions_from_jsonl(jsonl, log_actions) if jsonl.exists() else log_actions
    source = "actions.jsonl" if jsonl.exists() else "log"

    with _arcade_lock:
        env = _arc().make(game_id)
    if env is None:
        raise RuntimeError(f"cannot create environment for {game_id}")
    g = Game(env, game_id)
    steps: list[dict] = []
    notes: list[str] = []
    hash_mismatch = 0

    def pseudo(kind: str, a: int, t):
        steps.append({"i": len(steps), "a": a, "t": t, "kind": kind, "level": g.level, "attempt": g.attempt, "action": kind.upper(),
                      "row": None, "col": None, "tag": "", "label": "", "predicted": "", "actual": "", "changed": 0,
                      "level_completed": False, "game_over": False, "won": False, "frame": _hex(g.frame), "events": []})

    g.reset(); pseudo("reset", 0, 0)
    if source == "actions.jsonl" and actions and actions[0]["reset"]:
        actions = actions[1:]   # the initial reset is already applied
    for x in actions:
        if x["reset"]:
            g.reset(); pseudo("reset", x["a"], x["t"])
            continue
        act = x["action"]
        before = g.frame
        res = g.step(act)
        if res.get("invalid"):
            notes.append(f"step {x['a']}: {action_label(act)} rejected as invalid during replay (skipped)")
            continue
        tr = g.transitions[-1]
        s = {"i": len(steps), "a": x["a"], "t": x.get("t"), "kind": "step", "level": tr.level, "attempt": tr.attempt, "action": action_label(act),
             "row": act["row"] if isinstance(act, dict) else None, "col": act["col"] if isinstance(act, dict) else None,
             "tag": x.get("tag", ""), "label": x.get("label", ""), "predicted": x.get("predicted", ""), "actual": x.get("actual", ""),
             "changed": _diff(before, g.frame), "level_completed": bool(res["level_completed"]), "game_over": bool(res["game_over"]),
             "won": bool(res["won"]), "frame": _hex(g.frame), "events": []}
        if x.get("hash") and g.frame is not None:
            import hashlib
            if hashlib.sha1(g.frame.ascii.encode()).hexdigest()[:12] != x["hash"]:
                hash_mismatch += 1; s["hash_mismatch"] = True
        steps.append(s)
        if res["level_completed"] and not res["won"]:
            pseudo("start", x["a"], x.get("t"))
        elif res["game_over"] and source == "log":
            g.reset(); pseudo("reset", x["a"], x.get("t"))
    if hash_mismatch:
        notes.append(f"{hash_mismatch} replayed frames differ from the recorded hashes")
    if source == "log" and len(log_actions) != len([s for s in steps if s["kind"] == "step"]):
        notes.append("some actions could not be replayed")

    # attach the log events: everything logged after action N and before action N+1 belongs to step N
    by_a: dict[int, list] = {}
    for e in events:
        if ACTION_RE.match(e["text"]):
            continue
        by_a.setdefault(e["a"], []).append({"kind": _kind(e["text"]), "text": e["text"]})
    last_for_a: dict[int, dict] = {}
    for s in steps:
        last_for_a[s["a"]] = s
    for a, evs in by_a.items():
        target = last_for_a.get(a) or (steps[0] if a <= steps[0]["a"] else None)
        if target is None:   # counter beyond the replayed steps: attach to the last step
            target = steps[-1]
        target["events"].extend(evs)

    levels: dict = {}
    for s in steps:
        lv = levels.setdefault(str(s["level"]), {"level": s["level"], "steps": 0, "attempts": set(), "completed": False, "first": s["i"], "last": s["i"]})
        lv["steps"] += s["kind"] == "step"; lv["attempts"].add(s["attempt"]); lv["last"] = s["i"]
        lv["completed"] = lv["completed"] or s["level_completed"] or s["won"]
    for lv in levels.values():
        lv["attempts"] = len(lv["attempts"])
    return {"run_id": run_id, "game_id": game_id, "source": source, "steps": steps, "levels": levels, "notes": notes,
            "levels_total": g.levels_total, "log_events": len(events), "log_actions": len(log_actions)}


def _cache_key(run_id: str, game_id: str) -> tuple:
    log_dir = RESULTS / "logs" / run_id
    parts = [CACHE_VERSION]
    for name in (f"{game_id}.log", f"{game_id}.actions.jsonl"):
        p = log_dir / name
        parts.append((name, p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None)
    return tuple(parts)


def load_steps(run_id: str, game_id: str) -> dict:
    """Replay with a two-level cache (memory, then results/cache/<run>/<game>.json.gz keyed by the log's mtime/size)."""
    key = _cache_key(run_id, game_id)
    hit = _mem.get((run_id, game_id))
    if hit and hit[0] == key:
        return hit[1]
    path = CACHE / run_id / f"{game_id}.json.gz"
    if path.exists():
        try:
            with gzip.open(path, "rt") as f:
                data = json.load(f)
            if data.get("cache_key") == list(map(lambda k: list(k) if isinstance(k, tuple) else k, key)):
                _mem[(run_id, game_id)] = (key, data); return data
        except Exception:
            pass
    data = replay(run_id, game_id)
    data["cache_key"] = [list(k) if isinstance(k, tuple) else k for k in key]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    _mem[(run_id, game_id)] = (key, data)
    return data


def list_runs() -> list[dict]:
    """All run-*.json summaries, newest first, with the games that have a log."""
    out = []
    for p in sorted(RESULTS.glob("run-*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        run_id = d.get("run_id") or p.stem[4:]
        log_dir = RESULTS / "logs" / run_id
        logged = sorted(x.stem for x in log_dir.glob("*.log")) if log_dir.exists() else []
        params = d.get("config", {}).get("params", {})
        out.append({"run_id": run_id, "tag": d.get("tag", ""), "started_at": d.get("started_at"), "finished_at": d.get("finished_at"),
                    "commit": d.get("git", {}).get("commit"), "mode": params.get("mode"), "no_model": bool(params.get("no_model")),
                    "minutes": params.get("max_minutes"), "aggregate": d.get("aggregate", {}), "games": len(d.get("games", [])),
                    "logged_games": logged})
    return out


def run_detail(run_id: str) -> dict:
    p = RESULTS / f"run-{run_id}.json"
    d = json.loads(p.read_text())
    log_dir = RESULTS / "logs" / run_id
    games = []
    for g in d.get("games", []):
        gid = g.get("game_id")
        levels = []
        stats = {int(l["level"]): l for l in g.get("levels", []) if isinstance(l, dict) and "level" in l}
        total = int(g.get("levels_total") or 0)
        n = max(total, max(stats) if stats else 0)
        for lv in range(1, n + 1):
            st = stats.get(lv, {})
            ls = (g.get("level_scores") or [])
            lb = (g.get("level_baseline_actions") or [])
            levels.append({"level": lv, "completed": bool(st.get("completed")), "played": lv in stats, "actions": st.get("actions"),
                           "reason": st.get("reason"), "score": ls[lv - 1] if lv - 1 < len(ls) else None,
                           "baseline": lb[lv - 1] if lv - 1 < len(lb) else None})
        games.append({"game_id": gid, "levels_completed": g.get("levels_completed"), "levels_total": total, "actions": g.get("actions"),
                      "score": g.get("score"), "stop_reason": g.get("stop_reason") or g.get("error"), "seconds": g.get("seconds"),
                      "mismatches": g.get("mismatches"), "reviews": g.get("reviews"), "model_calls": g.get("model_calls"),
                      "has_log": (log_dir / f"{gid}.log").exists(), "levels": levels})
    return {"run_id": run_id, "tag": d.get("tag", ""), "started_at": d.get("started_at"), "git": d.get("git"), "config": d.get("config"),
            "aggregate": d.get("aggregate"), "games": games}
