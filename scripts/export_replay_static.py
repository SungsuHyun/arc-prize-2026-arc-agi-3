#!/usr/bin/env python
"""Export the Replay viewer as static files (GitHub Pages): the same page, with every API answer written out as a file.

    .venv/bin/python scripts/export_replay_static.py --out <dir> [--experiments pbg,rulebook] [--since 20260928] [--tag x] [--no-logs]

<dir>/index.html                          the viewer with window.REPLAY_STATIC = true (reads data/ instead of /api)
<dir>/data/runs.json                      GET /api/runs (finished runs only)
<dir>/data/find.json                      {hash: {run_id, game_id, experiment, status}} for the #h/<hash> lookup
<dir>/data/runs/<run>.json                GET /api/runs/<run>
<dir>/data/runs/<run>/<game>.json.gz      GET /api/runs/<run>/games/<game> (boards replayed with the offline engine; gunzipped in the page)
<dir>/data/runs/<run>/<game>.log.txt      raw log (omitted with --no-logs)
<dir>/data/runs/<run>/<game>.rulebook.json

Runs still in progress are skipped (a static page cannot follow them). Key-like tokens are redacted from logs."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_viewer import replay as R  # noqa: E402

PAGE = ROOT / "scripts" / "eval_viewer" / "index.html"
SECRET = re.compile(r"(sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]+)")


def _write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, default=str, separators=(",", ":"))
    path.write_text(body)
    return len(body)


def _write_gz(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = gzip.compress(json.dumps(obj, default=str, separators=(",", ":")).encode(), 9)
    path.write_bytes(body)
    return len(body)


def export(out: Path, *, experiments: list[str], since: str = "", tag: str = "", logs: bool = True, log=print) -> dict:
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    runs = [r for r in R.list_runs() if r["status"] == "finished" and r["experiment"] in experiments and r["logged_games"]
            and r["run_id"] >= since and (not tag or tag in (r.get("tag") or ""))]
    find, size, games, failed = {}, 0, 0, []
    for r in runs:
        run_id = r["run_id"]
        size += _write(data / "runs" / f"{run_id}.json", R.run_detail(run_id))
        ldir = R.log_dir(run_id)
        for g in r["logged_games"]:
            try:
                size += _write_gz(data / "runs" / run_id / f"{g}.json.gz", R.load_steps(run_id, g))
                games += 1
            except Exception as e:      # an unreplayable game still lists; its board view shows the error
                failed.append(f"{run_id}/{g}: {type(e).__name__}: {e}")
                _write_gz(data / "runs" / run_id / f"{g}.json.gz", {"error": f"replay failed: {e}"})
            if logs and (ldir / f"{g}.log").exists():
                p = data / "runs" / run_id / f"{g}.log.txt"
                p.write_text(SECRET.sub("<redacted>", (ldir / f"{g}.log").read_text(errors="replace"))); size += p.stat().st_size
            if (ldir / f"{g}.rulebook.json").exists():
                shutil.copy(ldir / f"{g}.rulebook.json", data / "runs" / run_id / f"{g}.rulebook.json")
        for gid, h in r["hashes"].items():
            find[h] = {"run_id": run_id, "game_id": gid, "experiment": r["experiment"], "status": r["status"]}
    size += _write(data / "runs.json", runs) + _write(data / "find.json", find)
    html = PAGE.read_text().replace("<script>\n", "<script>window.REPLAY_STATIC = true;</script>\n<script>\n", 1)
    (out / "index.html").write_text(html)
    summary = {"runs": len(runs), "games": games, "failed": failed, "mb": round(size / 1e6, 1)}
    log(f"replay export: {summary['runs']} runs, {games} games, {summary['mb']} MB -> {out}" + (f" ({len(failed)} replay failures)" if failed else ""))
    for f in failed[:10]:
        log("  failed: " + f)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", default="pbg,rulebook")
    ap.add_argument("--since", default="", help="only runs whose id (UTC yyyymmdd-hhmmss) is >= this")
    ap.add_argument("--tag", default="", help="only runs whose tag contains this")
    ap.add_argument("--no-logs", action="store_true", help="leave the raw logs out")
    a = ap.parse_args(argv)
    export(Path(a.out), experiments=[x.strip() for x in a.experiments.split(",") if x.strip()], since=a.since, tag=a.tag, logs=not a.no_logs)


if __name__ == "__main__":
    main()
