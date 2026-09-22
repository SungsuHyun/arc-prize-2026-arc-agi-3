"""Serve the benchmark site at http://localhost:8080 (default).

Every .html request rebuilds summary.json and the site pages first, so the
browser always shows the latest results — run `make bench` or `make exp-run`
in another terminal and just refresh the page.

Usage:
    .venv/bin/python scripts/serve_site.py [--port 8080] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "experiments" / "site"
PY = ROOT / ".venv" / "bin" / "python"


def rebuild() -> None:
    for script in ("exp_summary.py", "build_dashboard.py"):
        r = subprocess.run([str(PY), str(ROOT / "scripts" / script)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[serve_site] {script} failed:\n{r.stderr}")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.endswith(".html"):
            rebuild()
        super().do_GET()

    def log_message(self, fmt, *args):
        print(f"[serve_site] {self.address_string()} {fmt % args}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    rebuild()
    handler = partial(Handler, directory=str(SITE))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"벤치마크 사이트: http://{args.host}:{args.port}/")
    print("페이지를 새로고침하면 항상 최신 결과가 반영됩니다. 종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
