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

import datetime as dt
import gzip
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# every experiment line whose runner leaves run-<id>.json + logs/<id>/<game>.log (+ <game>.actions.jsonl); run ids are unique across lines
EXPERIMENTS = {"rulebook": ROOT / "experiments" / "rulebook" / "results", "pbg": ROOT / "experiments" / "pbg" / "results"}
RESULTS = EXPERIMENTS["rulebook"]
CACHE_VERSION = 4

TAG_RE = re.compile(r"^\[\s*(\d+)s a\s*(\d+) L(\d+)\] (.*)$")
CLOCK_RE = re.compile(r"^(\d\d):(\d\d):(\d\d) (.*)$")      # pbg logs: wall-clock prefix, no action counter
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
            continue
        m = CLOCK_RE.match(ln)
        if m:
            events.append({"clock": ln[:8], "a": None, "text": m.group(4)})
        elif events:
            events[-1]["text"] += "\n" + ln
    return events


def _kind_clock(text: str) -> str:
    """Event kind of a clock-format (pbg) line: the prefix before the first ':' when it is short, else the first word."""
    head = text.split("\n", 1)[0]
    i = head.find(":")
    if 0 < i <= 24:
        return head[:i].strip()
    return head.split(" ", 1)[0][:24] if head else "other"


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
    for r in read_record(path):
        if r["kind"] == "reset":
            out.append({"a": r["step"], "t": r.get("t"), "clock": r.get("clock"), "reset": True, "hash": r.get("hash")})
        else:
            base = by_step.get(r["step"], {})
            out.append({"a": r["step"], "t": r.get("t"), "clock": r.get("clock"), "action": r["action"], "reset": False, "hash": r.get("hash"),
                        "tag": base.get("tag", ""), "label": base.get("label", ""), "predicted": base.get("predicted", ""), "actual": base.get("actual", "")})
    return out


def read_record(path: Path) -> list[dict]:
    """Lines of <game>.actions.jsonl; a half-written last line (run still in progress) is skipped."""
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


# ── run lookup across experiment lines ────────────────────────────────────
def find_run(run_id: str) -> tuple[str, Path]:
    """(experiment, results dir) of a run id: the line whose summary or log dir has it."""
    for name, d in EXPERIMENTS.items():
        if (d / f"run-{run_id}.json").exists() or (d / "logs" / run_id).is_dir():
            return name, d
    raise FileNotFoundError(f"run {run_id}")


def log_dir(run_id: str) -> Path:
    return find_run(run_id)[1] / "logs" / run_id


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _run_meta(d: Path, run_id: str) -> dict:
    """logs/<run>/run.json written at run start (newer runners); older runs fall back to the id (utc time + pid)."""
    p = d / "logs" / run_id / "run.json"
    try:
        meta = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        meta = {}
    if "pid" not in meta:
        try:
            meta["pid"] = int(run_id.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            pass
    if "started_at" not in meta:
        try:
            meta["started_at"] = dt.datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S").replace(tzinfo=dt.timezone.utc).isoformat()
        except ValueError:
            pass
    return meta


def record_summary(path: Path) -> dict:
    """Per-level tally of an actions.jsonl: what a run in progress can show before its summary exists."""
    levels: dict[int, dict] = {}
    total = 0; completed = 0; levels_total = 0; state = None; last_level = None; resets = 0
    for r in read_record(path):
        levels_total = int(r.get("levels_total") or levels_total or 0); state = r.get("state", state)
        if r["kind"] == "reset":
            resets += 1; last_level = r.get("level"); continue
        lv = int(r.get("level") or 0); total += 1; last_level = lv
        st = levels.setdefault(lv, {"level": lv, "actions": 0, "completed": False})
        st["actions"] += 1
        if r.get("level_completed"):
            st["completed"] = True; completed += 1
    return {"levels": levels, "actions": total, "levels_completed": completed, "levels_total": levels_total, "state": state, "current_level": last_level, "resets": resets}


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
    ldir = log_dir(run_id)
    log = ldir / f"{game_id}.log"
    events = parse_log(log) if log.exists() else []
    log_actions = _actions_from_log(events)
    jsonl = ldir / f"{game_id}.actions.jsonl"
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

    def pseudo(kind: str, a: int, t, clock=None):
        steps.append({"i": len(steps), "a": a, "t": t, "clock": clock, "kind": kind, "level": g.level, "attempt": g.attempt, "action": kind.upper(),
                      "row": None, "col": None, "tag": "", "label": "", "predicted": "", "actual": "", "changed": 0,
                      "level_completed": False, "game_over": False, "won": False, "frame": _hex(g.frame), "events": []})

    g.reset(); pseudo("reset", 0, 0, actions[0].get("clock") if actions else None)
    if source == "actions.jsonl" and actions and actions[0]["reset"]:
        actions = actions[1:]   # the initial reset is already applied
    for x in actions:
        if x["reset"]:
            g.reset(); pseudo("reset", x["a"], x["t"], x.get("clock"))
            continue
        act = x["action"]
        before = g.frame
        res = g.step(act)
        if res.get("invalid"):
            notes.append(f"step {x['a']}: {action_label(act)} rejected as invalid during replay (skipped)")
            continue
        tr = g.transitions[-1]
        s = {"i": len(steps), "a": x["a"], "t": x.get("t"), "clock": x.get("clock"), "kind": "step", "level": tr.level, "attempt": tr.attempt, "action": action_label(act),
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
            pseudo("start", x["a"], x.get("t"), x.get("clock"))
        elif res["game_over"] and source == "log":
            g.reset(); pseudo("reset", x["a"], x.get("t"), x.get("clock"))
    if hash_mismatch:
        notes.append(f"{hash_mismatch} replayed frames differ from the recorded hashes")
    if source == "log" and len(log_actions) != len([s for s in steps if s["kind"] == "step"]):
        notes.append("some actions could not be replayed")
    if source == "log" and events and not log_actions and not jsonl.exists():
        notes.append("no action record for this game (run started before <game>.actions.jsonl existed): only the initial board")

    # attach the log events: everything logged after action N and before action N+1 belongs to step N
    # (rulebook logs carry the action counter; pbg logs carry a wall clock that is matched against the record's clock)
    by_a: dict[int, list] = {}
    clocked: list[dict] = []
    for e in events:
        if ACTION_RE.match(e["text"]):
            continue
        if e.get("a") is None:
            clocked.append({"kind": _kind_clock(e["text"]), "text": e["text"], "clock": e.get("clock")})
        else:
            by_a.setdefault(e["a"], []).append({"kind": _kind(e["text"]), "text": e["text"]})
    last_for_a: dict[int, dict] = {}
    for s in steps:
        last_for_a[s["a"]] = s
    for a, evs in by_a.items():
        target = last_for_a.get(a) or (steps[0] if a <= steps[0]["a"] else None)
        if target is None:   # counter beyond the replayed steps: attach to the last step
            target = steps[-1]
        target["events"].extend(evs)
    if clocked:
        stamped = [s for s in steps if s.get("clock")]
        for ev in clocked:
            target = steps[0]
            for s in stamped:
                if s["clock"] <= (ev["clock"] or ""):
                    target = s
                else:
                    break
            target["events"].append({"kind": ev["kind"], "text": ev["text"]})

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
    ldir = log_dir(run_id)
    parts = [CACHE_VERSION]
    for name in (f"{game_id}.log", f"{game_id}.actions.jsonl"):
        p = ldir / name
        parts.append((name, p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None)
    return tuple(parts)


def load_steps(run_id: str, game_id: str) -> dict:
    """Replay with a two-level cache (memory, then results/cache/<run>/<game>.json.gz keyed by the log's mtime/size)."""
    key = _cache_key(run_id, game_id)
    hit = _mem.get((run_id, game_id))
    if hit and hit[0] == key:
        return hit[1]
    path = find_run(run_id)[1] / "cache" / run_id / f"{game_id}.json.gz"
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


def _status(d: Path, run_id: str, meta: dict) -> str:
    """finished (summary exists) / running (runner pid alive) / aborted (log dir without summary, no live pid)."""
    if (d / f"run-{run_id}.json").exists():
        return "finished"
    return "running" if _pid_alive(meta.get("pid")) else "aborted"


def list_runs() -> list[dict]:
    """All runs of every experiment line, newest first: finished ones from run-*.json, unfinished ones from logs/<id>/ alone."""
    out = []
    for exp, d in EXPERIMENTS.items():
        seen = set()
        for p in d.glob("run-*.json"):
            try:
                doc = json.loads(p.read_text())
            except Exception:
                continue
            run_id = doc.get("run_id") or p.stem[4:]
            seen.add(run_id)
            ldir = d / "logs" / run_id
            logged = sorted(x.stem for x in ldir.glob("*.log")) if ldir.exists() else []
            params = doc.get("config", {}).get("params", {})
            out.append({"run_id": run_id, "experiment": exp, "status": "finished", "tag": doc.get("tag", ""), "started_at": doc.get("started_at"),
                        "finished_at": doc.get("finished_at"), "commit": doc.get("git", {}).get("commit"), "mode": params.get("mode"),
                        "no_model": bool(params.get("no_model") or params.get("no_llm")), "minutes": params.get("max_minutes"),
                        "aggregate": doc.get("aggregate", {}), "games": len(doc.get("games", [])), "logged_games": logged})
        logs = d / "logs"
        for ldir in (logs.iterdir() if logs.is_dir() else []):
            if not ldir.is_dir() or ldir.name in seen:
                continue
            meta = _run_meta(d, ldir.name)
            logged = sorted(x.stem for x in ldir.glob("*.log"))
            params = meta.get("params", {})
            recs = {g: record_summary(ldir / f"{g}.actions.jsonl") for g in logged if (ldir / f"{g}.actions.jsonl").exists()}
            out.append({"run_id": ldir.name, "experiment": exp, "status": _status(d, ldir.name, meta), "tag": meta.get("tag", ""),
                        "started_at": meta.get("started_at"), "finished_at": None, "commit": (meta.get("git") or {}).get("commit"), "mode": params.get("mode"),
                        "no_model": bool(params.get("no_model") or params.get("no_llm")), "minutes": params.get("max_minutes"),
                        "aggregate": {"levels_completed": sum(r["levels_completed"] for r in recs.values()), "actions": sum(r["actions"] for r in recs.values()),
                                      "games_played": len(logged)},
                        "games": len(meta.get("games") or logged), "logged_games": logged})
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out


def _levels_from_summary(g: dict) -> list[dict]:
    """Per-level rows of a finished game: rulebook summaries carry a levels list, pbg ones a level_actions dict."""
    stats = {int(l["level"]): l for l in g.get("levels", []) if isinstance(l, dict) and "level" in l}
    if not stats and isinstance(g.get("level_actions"), dict):
        done = int(g.get("levels_completed") or 0)
        stats = {int(k): {"level": int(k), "actions": v, "completed": int(k) <= done, "reason": g.get("stop_reason") if int(k) > done else None}
                 for k, v in g["level_actions"].items()}
    total = int(g.get("levels_total") or 0)
    n = max(total, max(stats) if stats else 0)
    ls = (g.get("level_scores") or []); lb = (g.get("level_baseline_actions") or [])
    return [{"level": lv, "completed": bool(stats.get(lv, {}).get("completed")), "played": lv in stats, "actions": stats.get(lv, {}).get("actions"),
             "reason": stats.get(lv, {}).get("reason"), "score": ls[lv - 1] if lv - 1 < len(ls) else None, "baseline": lb[lv - 1] if lv - 1 < len(lb) else None}
            for lv in range(1, n + 1)]


def run_detail(run_id: str) -> dict:
    exp, d = find_run(run_id)
    ldir = d / "logs" / run_id
    p = d / f"run-{run_id}.json"
    if p.exists():
        doc = json.loads(p.read_text()); status = "finished"
        raw_games = doc.get("games", [])
    else:
        meta = _run_meta(d, run_id); status = _status(d, run_id, meta)
        doc = {"tag": meta.get("tag", ""), "started_at": meta.get("started_at"), "git": meta.get("git"), "config": {"games": meta.get("games"), "params": meta.get("params", {})},
               "aggregate": None}
        logged = sorted(x.stem for x in ldir.glob("*.log"))
        raw_games = [{"game_id": g} for g in (meta.get("games") or logged)]
        for g in raw_games:            # in progress: everything comes from the action record
            rp = ldir / f"{g['game_id']}.actions.jsonl"
            if rp.exists():
                r = record_summary(rp)
                g.update({"levels_completed": r["levels_completed"], "levels_total": r["levels_total"], "actions": r["actions"], "state": r["state"],
                          "levels": [{"level": lv, "actions": v["actions"], "completed": v["completed"], "reason": None if v["completed"] else ("in progress" if status == "running" else "unfinished")}
                                     for lv, v in sorted(r["levels"].items())], "stop_reason": None if status == "running" else "aborted"})
            else:
                g.update({"stop_reason": ("waiting" if not (ldir / f"{g['game_id']}.log").exists() else "no action record") if status == "running" else "no record"})
        doc["aggregate"] = {"levels_completed": sum(g.get("levels_completed") or 0 for g in raw_games), "actions": sum(g.get("actions") or 0 for g in raw_games),
                            "games_played": len(raw_games), "games_level2plus": sum(1 for g in raw_games if (g.get("levels_completed") or 0) >= 2)}
    games = []
    for g in raw_games:
        gid = g.get("game_id")
        games.append({"game_id": gid, "levels_completed": g.get("levels_completed"), "levels_total": int(g.get("levels_total") or 0), "actions": g.get("actions"),
                      "score": g.get("score"), "stop_reason": g.get("stop_reason") or g.get("error"), "seconds": g.get("seconds") or g.get("elapsed"),
                      "mismatches": g.get("mismatches"), "reviews": g.get("reviews"), "model_calls": g.get("model_calls") or g.get("llm_calls"),
                      "has_log": (ldir / f"{gid}.log").exists(), "has_record": (ldir / f"{gid}.actions.jsonl").exists(),
                      "has_rulebook": (ldir / f"{gid}.rulebook.json").exists(), "levels": _levels_from_summary(g)})
    return {"run_id": run_id, "experiment": exp, "status": status, "tag": doc.get("tag", ""), "started_at": doc.get("started_at"), "git": doc.get("git"),
            "config": doc.get("config"), "aggregate": doc.get("aggregate"), "games": games}
