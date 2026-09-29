#!/usr/bin/env python
"""Replay: browse recorded runs (rulebook and pbg lines, finished or still in progress) game by game and level by level, replaying every board.

    make eval-site            # http://0.0.0.0:8090/ (all interfaces)
    .venv/bin/python scripts/serve_eval.py [--port 8090] [--host 0.0.0.0] [--no-reload]

Auto-reload: by default a supervisor process runs the server as a child and restarts it whenever a watched
source file changes (this file, scripts/eval_viewer/, rulebook/, arcnav/). index.html is read per request, so
page edits need only a browser refresh; the page also polls /api/health and reloads itself after a restart.
Replay caches on disk survive a restart.

Data: experiments/{rulebook,pbg}/results/run-*.json + logs/<run>/<game>.log + <game>.actions.jsonl (+ logs/<run>/run.json written at
run start, which is how a run still in progress shows up before its summary exists). Boards are rebuilt by replaying the recorded
actions against the offline engine (scripts/eval_viewer/replay.py) and cached under results/cache/. Nothing is published; local only.

API (all JSON):
    GET /api/runs                               runs of every line, newest first (status: finished | running | aborted)
    GET /api/runs/<run>                         games + per-level summary of one run
    GET /api/runs/<run>/games/<game>            replayed steps of one game (?level=N keeps one level)
    GET /api/runs/<run>/games/<game>/rulebook   final rulebook of that game
    GET /api/runs/<run>/games/<game>/log        raw log (text/plain)
    GET /api/find/<hash>                        {run_id, game_id, ...} of one game execution (hash = scripts/eval_viewer/ids.game_hash)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_viewer import replay as R  # noqa: E402

HERE = Path(__file__).resolve().parent / "eval_viewer"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
STARTED = time.time()
WATCH_DIRS = [HERE, ROOT / "rulebook", ROOT / "arcnav"]
WATCH_FILES = [Path(__file__).resolve()]

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
            if parts[1:] == ["health"]:
                return self._json({"started": STARTED, "pid": os.getpid()})
            if len(parts) == 3 and parts[1] == "find":
                if not SAFE.match(parts[2]):
                    return self._json({"error": "bad hash"}, 400)
                hit = R.find_game(parts[2])
                return self._json(hit) if hit else self._json({"error": f"no game with hash {parts[2]}"}, 404)
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
                        p = R.log_dir(run_id) / f"{game}.rulebook.json"
                        return self._send(200, p.read_bytes()) if p.exists() else self._json({"error": "no rulebook"}, 404)
                    if sub == "log":
                        p = R.log_dir(run_id) / f"{game}.log"
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

    def log_message(self, fmt, *args):
        print(f"[serve_eval] {fmt % args}")


# ── auto-reload supervisor ─────────────────────────────────────────────────
def _snapshot() -> dict:
    files = list(WATCH_FILES)
    for d in WATCH_DIRS:
        if d.exists():
            files += [f for f in d.rglob("*") if f.suffix in (".py", ".html") and "__pycache__" not in f.parts]
    out = {}
    for f in files:
        try:
            out[str(f)] = f.stat().st_mtime_ns
        except FileNotFoundError:
            pass
    return out


def supervise(argv: list[str]) -> int:
    """Run the server as a child process; restart it when a watched file changes. Ctrl+C stops both."""
    cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--no-reload"]
    env = {**os.environ, "EVAL_VIEWER_CHILD": "1"}
    child = None
    try:
        while True:
            snap = _snapshot()
            child = subprocess.Popen(cmd, env=env)
            crashed = False
            while True:
                time.sleep(0.5)
                if child.poll() is not None:
                    if child.returncode == 0:
                        return 0
                    if not crashed:
                        print(f"[reload] server exited with code {child.returncode}; waiting for a file change", flush=True)
                    crashed = True
                if _snapshot() != snap:
                    changed = [k for k in set(snap) | set(_snapshot()) if snap.get(k) != _snapshot().get(k)]
                    print(f"[reload] {', '.join(Path(c).name for c in changed[:4])} changed -> restarting", flush=True)
                    break
                if crashed:
                    snap = _snapshot(); time.sleep(1.0)
                    continue
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                child.kill()
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--no-reload", action="store_true", help="run a single process without the file watcher")
    args = p.parse_args()
    if not args.no_reload:
        sys.exit(supervise([a for a in sys.argv[1:] if a != "--no-reload"]))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    tag = " (auto-reload on)" if os.environ.get("EVAL_VIEWER_CHILD") else ""
    print(f"Replay: http://{args.host}:{args.port}/{tag}   (종료: Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
