"""Stable ids for recorded games.

game_hash(run_id, game_id): 10 hex chars, sha1 of "<run id>/<game id>". One value per game execution — a run id is unique
(utc timestamp + pid) and a game is played once per run — so it can be computed for any run, old or new, from the
same two strings. Runners write it into the run summary and logs/<run>/run.json; the Replay viewer resolves it with
GET /api/find/<hash> (or the #h/<hash> URL) and shows it next to every game. Stdlib only: imported by the runners.
"""
from __future__ import annotations

import hashlib

HASH_LEN = 10


def game_hash(run_id: str, game_id: str) -> str:
    return hashlib.sha1(f"{run_id}/{game_id}".encode()).hexdigest()[:HASH_LEN]
