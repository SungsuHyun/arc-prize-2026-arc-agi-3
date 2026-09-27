#!/usr/bin/env python
"""Build notebooks/rulebook/rulebook_submission.ipynb: the rulebook agent (rulebook/ + arcnav/ as its library) with an
in-notebook vLLM server (Qwen3.6-27B-FP8) for the ARC-AGI-3 competition rerun.

Attached inputs (kernel-metadata.json): the competition (arc-agi wheels + environment_files), the vLLM wheelhouse dataset and
the HF snapshot of the model. Sources are embedded, so no source dataset is needed.

Commit ("Save & Run All") = quota-safe smoke: 2 games x 12 min offline. A competition rerun (KAGGLE_IS_COMPETITION_RERUN)
plays every gateway game with a global deadline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_arcnav_notebook as base   # setup / vLLM cells and constants are shared with the arcnav notebook

KERNEL_ID = "sungsuhyun/arc3-rulebook"
KERNEL_TITLE = "arc3-rulebook"
OUT_DIR = ROOT / "notebooks" / "rulebook"
NOTEBOOK_PATH = OUT_DIR / "rulebook_submission.ipynb"
METADATA_PATH = OUT_DIR / "kernel-metadata.json"
SMOKE_GAMES = ["tn36", "lp85"]
SMOKE_MINUTES = 12
RERUN_JOBS = 12
RERUN_TOTAL_MINUTES = 470
RERUN_MAX_MINUTES_PER_GAME = 120
PRESET = base.PRESETS["qwen27b"]

PACKAGES = {"arcnav": sorted(p.name for p in (ROOT / "arcnav").glob("*.py")),
            "rulebook": sorted(p.name for p in (ROOT / "rulebook").glob("*.py"))}


def build() -> dict:
    sources = {pkg: {name: (ROOT / pkg / name).read_text() for name in names} for pkg, names in PACKAGES.items()}
    setup_cell = base.code_cell(dedent(f"""\
        import json, os, subprocess, sys, time
        T0 = time.time()
        RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
        WORK = '/kaggle/working'
        print('competition rerun:', RERUN)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', '{base.COMP}/arc_agi_3_wheels', 'arc-agi'], check=True)
        SOURCES = json.loads({json.dumps(json.dumps(sources))})
        for pkg, files in SOURCES.items():
            os.makedirs(f'{{WORK}}/{{pkg}}', exist_ok=True)
            for name, body in files.items():
                open(f'{{WORK}}/{{pkg}}/{{name}}', 'w').write(body)
        sys.path.insert(0, WORK)
        import arc_agi, rulebook.run, rulebook.agent
        print('rulebook agent embedded | arc_agi ok |', f'{{time.time()-T0:.0f}}s')
        """))
    # the vLLM cell is identical to the arcnav notebook's (same wheelhouse / model / flags)
    vllm_cell = base.build()["cells"][2]
    play_cell = base.code_cell(dedent(f"""\
        import math, os, sys, time, urllib.request, logging
        from pathlib import Path
        sys.path.insert(0, '/kaggle/working')
        logging.basicConfig(level=logging.WARNING)
        from arcnav.runner import make_arcade
        from rulebook.run import run, DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG, model='{base.SERVED_MODEL}', base_url='http://127.0.0.1:1234/v1', level_actions=200, max_actions=4000, max_levels=20)
        out_dir = Path('/kaggle/working/rulebook-results')
        if RERUN:
            os.environ['ARC_API_KEY'] = 'test-key-123'
            deadline = time.time() + 60
            while True:  # wait for the gateway sidecar
                try:
                    urllib.request.urlopen('http://gateway:8001/api/games', timeout=10).read(); break
                except Exception as e:
                    if time.time() > deadline:
                        raise RuntimeError(f'gateway not ready: {{e!r}}')
                    time.sleep(5)
            arc = make_arcade(competition=True, base_url='http://gateway:8001/')
            games = [e.game_id for e in arc.get_environments()]
            waves = max(1, math.ceil(len(games) / {RERUN_JOBS}))
            total_left = {RERUN_TOTAL_MINUTES} - (time.time() - T0) / 60
            cfg.update(jobs={RERUN_JOBS}, max_minutes=max(20, min({RERUN_MAX_MINUTES_PER_GAME}, total_left / waves)))
            print(f'rerun: {{len(games)}} games, {{cfg["jobs"]}} concurrent, {{cfg["max_minutes"]:.0f}} min/game, waves={{waves}}')
            run(games, cfg, out_dir=out_dir, tag='kaggle-rerun', arc=arc, deadline=T0 + {RERUN_TOTAL_MINUTES} * 60)   # the gateway records actions and emits submission.parquet
        else:
            arc = make_arcade(environments_dir='{base.COMP}/environment_files')
            cfg.update(jobs=2, max_minutes={SMOKE_MINUTES})
            run({SMOKE_GAMES!r}, cfg, out_dir=out_dir, tag='kaggle-smoke', arc=arc)
            import pandas as pd   # commit mode: dummy submission so the commit succeeds
            pd.DataFrame(data=[['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet('/kaggle/working/submission.parquet', index=False)
        try:
            VLLM.terminate()
        except Exception:
            pass
        print(f'done in {{(time.time()-T0)/60:.1f}} min')
        """))
    nb = base.build()
    nb["cells"] = [base.markdown_cell("# ARC-AGI-3 — rulebook agent submission (SungsuHyun)\n\n"
                                      "Our solver: an explicit rulebook of hypotheses (environment / rules / win conditions) written by Qwen3.6-27B-FP8 (vLLM in the notebook) "
                                      "at level start; a deterministic predictor (entities and regions, click statistics by colour and region, button slot permutations, "
                                      "marker coupling) says what every candidate action will do; the model only chooses; every action is verified and mismatches "
                                      "trigger a review. Win predicates (gone / inside / equal / pose / pressed) are learned from completed levels and program-made plans "
                                      "(match / press BFS / marker) are offered as candidates. Sources are embedded from `rulebook/` and `arcnav/` by "
                                      "`scripts/build_rulebook_notebook.py`; do not edit cells here.\n\n"
                                      "Commit = 2-game offline smoke (12 min); the competition rerun plays every gateway game."),
                   setup_cell, vllm_cell, play_cell]
    return nb


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    METADATA_PATH.write_text(json.dumps({
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": NOTEBOOK_PATH.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [base.WHEELHOUSE_REF, PRESET["dataset_model"]], "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"], "model_sources": [], "machine_shape": base.MACHINE_SHAPE}, indent=2) + "\n")
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} ({NOTEBOOK_PATH.stat().st_size // 1024} KB) and {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
