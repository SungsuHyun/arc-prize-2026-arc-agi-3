"""Host the ARC-AGI-3 game engine as a local REST server.

Replicates the official three.arcprize.org API (same routes: /api/games,
/api/scorecard/*, /api/cmd/*) using the `arc-agi` package's built-in Flask
app, backed by the same game engine `make play-local` uses. Games are
downloaded on first use and cached in ./environment_files/.

Point the vendored ARC-AGI-3-Agents framework at it with its defaults
(SCHEME=http HOST=localhost PORT=8001), or talk to it with plain HTTP:

    curl http://localhost:8001/api/games
    curl -X POST http://localhost:8001/api/cmd/RESET \
         -H 'Content-Type: application/json' -d '{"game_id": "ls20"}'

Usage:
    .venv/bin/python scripts/serve_local.py [--host 127.0.0.1] [--port 8001]
"""
from __future__ import annotations

import argparse

import arc_agi
from arc_agi import OperationMode
from arc_agi.server import create_app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--competition-mode", action="store_true",
                   help="Mirror Kaggle's competition gateway behaviour")
    args = p.parse_args()

    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    app, _api = create_app(arcade, competition_mode=args.competition_mode)
    print(f"ARC-AGI-3 local server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
