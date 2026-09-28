#!/usr/bin/env python
"""Local evaluation viewer: browse recorded rulebook runs game by game and level by level, replaying every board.

    make eval-site            # http://0.0.0.0:8090/ (all interfaces)
    .venv/bin/python scripts/serve_eval.py [--port 8090] [--host 0.0.0.0]

Data: experiments/rulebook/results/run-*.json + logs/<run>/<game>.log (+ <game>.actions.jsonl for newer runs).
Boards are rebuilt by replaying the recorded actions against the offline engine (scripts/eval_viewer/replay.py)
and cached under results/cache/. Nothing is published; this is a local tool only.

API (all JSON):
    GET /api/runs                               runs, newest first
    GET /api/runs/<run>                         games + per-level summary of one run
    GET /api/runs/<run>/games/<game>            replayed steps of one game (?level=N keeps one level)
    GET /api/runs/<run>/games/<game>/rulebook   final rulebook of that game
    GET /api/runs/<run>/games/<game>/log        raw log (text/plain)
    GET /api/games                              playable game ids (environment_files)
    POST /api/play/new {game_id}                start an interactive session (no timing, nothing recorded)
    POST /api/play/<sid>/step {action}          action = "UP"|"DOWN"|"LEFT"|"RIGHT"|"SPACE"|"ACTION7" | {"action":"MOUSE","row":r,"col":c}
    POST /api/play/<sid>/reset
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_viewer import replay as R  # noqa: E402

HERE = Path(__file__).resolve().parent / "eval_viewer"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")

# ── interactive play sessions ──────────────────────────────────────────────
_sessions: dict = {}
_sessions_lock = threading.Lock()
MAX_SESSIONS = 32


def list_games() -> list[str]:
    d = R.ROOT / "environment_files"
    return sorted(x.name for x in d.iterdir() if x.is_dir()) if d.exists() else []


def _hex(frame) -> list[str]:
    return ["".join(f"{v & 15:x}" for v in row) for row in frame.grid]


def _state(sess: dict, last: dict | None = None) -> dict:
    g = sess["game"]
    return {"sid": sess["sid"], "game_id": g.game_id, "frame": _hex(g.frame) if g.frame else None, "level": g.level, "levels_total": g.levels_total,
            "state": g.state, "valid_actions": g.valid_actions, "actions_used": g.actions_used, "level_actions": g.level_actions,
            "attempt": g.attempt, "level_action_log": g.level_action_log, "last": last}


def play_new(game_id: str) -> dict:
    from rulebook.env import Game
    if not SAFE.match(game_id) or game_id not in list_games():
        raise ValueError(f"unknown game {game_id}")
    with R._arcade_lock:
        env = R._arc().make(game_id)
    if env is None:
        raise RuntimeError(f"cannot create environment for {game_id}")
    g = Game(env, game_id); g.reset()
    sess = {"sid": uuid.uuid4().hex[:12], "game": g, "lock": threading.Lock(), "created": time.time()}
    with _sessions_lock:
        if len(_sessions) >= MAX_SESSIONS:   # drop the oldest
            oldest = min(_sessions.values(), key=lambda x: x["created"])
            _sessions.pop(oldest["sid"], None)
        _sessions[sess["sid"]] = sess
    return _state(sess)


def play_step(sid: str, action) -> dict:
    from rulebook.env import action_label
    sess = _sessions.get(sid)
    if sess is None:
        raise KeyError("session expired: start a new game")
    with sess["lock"]:
        g = sess["game"]
        if isinstance(action, dict):
            act = {"action": "MOUSE", "row": max(0, min(63, int(action.get("row", 0)))), "col": max(0, min(63, int(action.get("col", 0))))}
        else:
            act = str(action)
            if act not in ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "ACTION7"):
                raise ValueError(f"bad action {act}")
        before = g.frame
        res = g.step(act)
        changed = 0 if res.get("invalid") else sum(1 for ra, rb in zip(before.grid, g.frame.grid) for x, y in zip(ra, rb) if x != y)
        last = {**res, "action": action_label(act), "row": act["row"] if isinstance(act, dict) else None, "col": act["col"] if isinstance(act, dict) else None,
                "changed": changed}
        return _state(sess, last)


def play_reset(sid: str) -> dict:
    sess = _sessions.get(sid)
    if sess is None:
        raise KeyError("session expired: start a new game")
    with sess["lock"]:
        sess["game"].reset()
        return _state(sess, {"action": "RESET", "row": None, "col": None, "changed": 0})


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode())

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        parts = [p for p in u.path.split("/") if p]
        try:
            if not parts or parts == ["index.html"]:
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] != "api":
                return self._json({"error": "not found"}, 404)
            if parts[1:] == ["runs"]:
                return self._json(R.list_runs())
            if parts[1:] == ["games"]:
                return self._json(list_games())
            if len(parts) >= 3 and parts[1] == "runs":
                run_id = parts[2]
                if not SAFE.match(run_id):
                    return self._json({"error": "bad run id"}, 400)
                if len(parts) == 3:
                    return self._json(R.run_detail(run_id))
                if len(parts) >= 5 and parts[3] == "games":
                    game = parts[4]
                    if not SAFE.match(game):
                        return self._json({"error": "bad game id"}, 400)
                    sub = parts[5] if len(parts) > 5 else ""
                    if sub == "rulebook":
                        p = R.RESULTS / "logs" / run_id / f"{game}.rulebook.json"
                        return self._send(200, p.read_bytes()) if p.exists() else self._json({"error": "no rulebook"}, 404)
                    if sub == "log":
                        p = R.RESULTS / "logs" / run_id / f"{game}.log"
                        return self._send(200, p.read_bytes(), "text/plain; charset=utf-8") if p.exists() else self._json({"error": "no log"}, 404)
                    data = R.load_steps(run_id, game)
                    if "level" in q:
                        lv = int(q["level"][0])
                        data = {**data, "steps": [s for s in data["steps"] if s["level"] == lv], "level": lv}
                    return self._json(data)
            return self._json({"error": "not found"}, 404)
        except FileNotFoundError as e:
            return self._json({"error": f"not found: {e}"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if parts[:2] == ["api", "play"]:
                if parts[2:] == ["new"]:
                    return self._json(play_new(str(body.get("game_id", ""))))
                if len(parts) == 4 and parts[3] == "step":
                    return self._json(play_step(parts[2], body.get("action")))
                if len(parts) == 4 and parts[3] == "reset":
                    return self._json(play_reset(parts[2]))
            return self._json({"error": "not found"}, 404)
        except (KeyError, ValueError) as e:
            return self._json({"error": e.args[0] if e.args else str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):
        print(f"[serve_eval] {fmt % args}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"평가 뷰어: http://{args.host}:{args.port}/   (종료: Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
