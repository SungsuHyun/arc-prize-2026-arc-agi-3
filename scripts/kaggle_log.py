"""Print the status, output files and execution log of a Kaggle kernel.

The kaggle CLI's `kernels output` downloads every output file (GBs for the
wheels kernel); this hits the same REST endpoint but prints only the log.

Usage:
    .venv/bin/python scripts/kaggle_log.py                 # submission kernel (notebooks/)
    .venv/bin/python scripts/kaggle_log.py notebooks/wheels
    .venv/bin/python scripts/kaggle_log.py owner/kernel-slug --tail 80 --grep vllm
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kernel_ref(arg: str) -> str:
    p = ROOT / arg
    if p.is_dir() and (p / "kernel-metadata.json").exists():
        return json.loads((p / "kernel-metadata.json").read_text())["id"]
    return arg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kernel", nargs="?", default="notebooks",
                    help="notebook dir (with kernel-metadata.json) or owner/slug")
    ap.add_argument("--tail", type=int, default=60, help="print last N log lines")
    ap.add_argument("--grep", default=None, help="regex filter for log lines")
    args = ap.parse_args()

    token = (ROOT / ".kaggle" / "access_token").read_text().strip()
    owner, slug = kernel_ref(args.kernel).split("/", 1)
    hdr = {"Authorization": f"Bearer {token}"}

    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/kernels/status?userName={owner}&kernelSlug={slug}",
        headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as r:
        st = json.load(r)
    print(f"[{owner}/{slug}] status={st.get('status')}"
          + (f"  failure={st.get('failureMessage')}" if st.get("failureMessage") else ""))

    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/kernels/output?userName={owner}&kernelSlug={slug}",
        headers=hdr)
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    files = out.get("files") or []
    if files:
        total = sum(f.get("fileSize", 0) for f in files)
        print(f"output files: {len(files)} ({total / 2**20:.1f} MiB)")
        for f in files[:8]:
            print(f"  {f.get('fileSize', 0) / 2**20:8.1f} MiB  {f.get('fileName')}")
    log = out.get("log") or ""
    try:
        entries = json.loads(log)
        text = "".join(e.get("data", "") for e in entries if isinstance(e, dict))
    except Exception:
        text = log
    lines = text.splitlines()
    if args.grep:
        rx = re.compile(args.grep, re.I)
        lines = [ln for ln in lines if rx.search(ln)]
    print(f"--- log ({len(lines)} lines, showing last {args.tail}) ---")
    for ln in lines[-args.tail:]:
        print(ln[:300])


if __name__ == "__main__":
    main()
