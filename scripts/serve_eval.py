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
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_viewer import replay as R  # noqa: E402

HERE = Path(__file__).resolve().parent / "eval_viewer"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


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
