#!/usr/bin/env python
"""Build notebooks/arcnav/arcnav_submission.ipynb: our arcnav agent + an in-notebook vLLM
server (Qwen3.6-27B-FP8) for the ARC-AGI-3 competition rerun.

Attached inputs (kernel-metadata.json): the competition (arc-agi wheels + environment_files),
a vLLM wheelhouse dataset and an HF snapshot of the model. The arcnav sources are embedded
in the notebook, so no source dataset is needed.

Commit ("Save & Run All") = quota-safe smoke: 2 games x 15 min offline. A competition rerun
(KAGGLE_IS_COMPETITION_RERUN) plays every gateway game with a global deadline.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "arcnav"

import argparse, sys
_ap = argparse.ArgumentParser(); _ap.add_argument("--preset", default="qwen27b"); _ap.add_argument("--out", default=None)
_ARGS, _ = _ap.parse_known_args(sys.argv[1:])
PRESETS = {
    # model source (dataset_sources / model_sources), vLLM flags, client extra_body
    "qwen27b": {"kernel": "sungsuhyun/arc3-arcnav", "title": "arc3-arcnav", "out": "arcnav",
                "dataset_model": "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot", "model_source": None,
                "vllm_flags": ["--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", '{"preserve_thinking": true}'],
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "max_tokens": 4096},
    "gptoss120b": {"kernel": "sungsuhyun/arc3-arcnav-gptoss", "title": "arc3-arcnav-gptoss", "out": "arcnav-gptoss",
                   "dataset_model": None, "model_source": "danielhanchen/gpt-oss-120b/transformers/default/1",   # 65 GB MXFP4, Apache 2.0
                   "vllm_flags": ["--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss"],
                   "extra_datasets": ["sungsuhyun/tiktoken-o200k-cache"],   # o200k_base vocab under its sha1 name: harmony loads it offline
                   "env": {"TIKTOKEN_RS_CACHE_DIR": "/kaggle/input/datasets/sungsuhyun/tiktoken-o200k-cache"},
                   "extra_body": {"reasoning_effort": "low"}, "max_tokens": 6144},
}
PRESET = PRESETS[_ARGS.preset]
KERNEL_ID = PRESET["kernel"]
KERNEL_TITLE = PRESET["title"]
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"       # vllm 0.19 / torch 2.10 / flashinfer 0.6.6, requirements.lock
MODEL_REF = PRESET["dataset_model"] or PRESET["model_source"]
SERVED_MODEL = "local-model"
OUT_DIR = ROOT / "notebooks" / (_ARGS.out or PRESET["out"])
NOTEBOOK_PATH = OUT_DIR / "arcnav_submission.ipynb"
METADATA_PATH = OUT_DIR / "kernel-metadata.json"
MACHINE_SHAPE = "NvidiaRtxPro6000"

SMOKE_GAMES = ["ls20", "vc33"]
SMOKE_MINUTES = 15
RERUN_JOBS = 12               # concurrent games in the competition rerun
RERUN_TOTAL_MINUTES = 470     # global cap (Kaggle rerun limit is 540 min incl. model load)
RERUN_MAX_MINUTES_PER_GAME = 150

SOURCE_FILES = ["__init__.py", "frame.py", "nav.py", "sandbox.py", "llm.py", "prompts.py", "solver.py", "agent.py", "runner.py"]


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def markdown_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def build() -> dict:
    sources = {name: (PKG / name).read_text() for name in SOURCE_FILES}

    setup_cell = code_cell(dedent(f"""\
        import json, os, subprocess, sys, time
        T0 = time.time()
        RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
        WORK = '/kaggle/working'
        print('competition rerun:', RERUN)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', '{COMP}/arc_agi_3_wheels', 'arc-agi'], check=True)
        # arcnav package (embedded sources)
        SOURCES = json.loads({json.dumps(json.dumps(sources))})
        os.makedirs(f'{{WORK}}/arcnav', exist_ok=True)
        for name, body in SOURCES.items():
            open(f'{{WORK}}/arcnav/{{name}}', 'w').write(body)
        sys.path.insert(0, WORK)
        import arcnav, arc_agi
        print('arcnav', arcnav.__version__, '| arc_agi ok |', f'{{time.time()-T0:.0f}}s')
        """))

    vllm_cell = code_cell(dedent(f"""\
        # In-notebook vLLM OpenAI server from the attached wheelhouse (installed into a private target dir so the
        # notebook kernel's own packages stay untouched) serving the attached HF snapshot.
        import glob, os, shutil, subprocess, sys, time, urllib.request, json
        WORK = '/kaggle/working'
        def _find(ref):
            owner, slug = ref.split('/', 1)
            for p in (f'/kaggle/input/{{slug}}', f'/kaggle/input/datasets/{{owner}}/{{slug}}'):
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(ref)
        WHEELHOUSE = _find('{WHEELHOUSE_REF}')
        _mp = glob.glob('/kaggle/input/models/**/config.json', recursive=True)
        MODEL_PATH = os.path.dirname(_mp[0]) if ({PRESET["model_source"] is not None!r} and _mp) else _find('{MODEL_REF}')
        cfgs = glob.glob(MODEL_PATH + '/**/config.json', recursive=True)
        MODEL_PATH = os.path.dirname(cfgs[0]) if cfgs else MODEL_PATH
        SITE = '/tmp/vllm-site-packages'   # outside /kaggle/working so the kernel output stays small
        print('wheelhouse:', WHEELHOUSE, '| model:', MODEL_PATH)
        print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
        if not os.path.exists(SITE + '/vllm'):
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', WHEELHOUSE, '--requirement', WHEELHOUSE + '/requirements.lock',
                            '--target', SITE, '--upgrade', '--ignore-installed', '--only-binary', ':all:', '--no-compile', '--disable-pip-version-check',
                            '--no-warn-conflicts', '-q'], check=True)
        # FlashInfer JIT-compiles sm120 kernels and links -lcuda: the driver stub lives in /usr/local/nvidia/lib64 on Kaggle
        env = dict(os.environ, PYTHONPATH=SITE, USE_TF='0', TRANSFORMERS_NO_TF='1', TRANSFORMERS_NO_TORCHVISION='1', VLLM_NO_USAGE_STATS='1', **{PRESET.get("env", {})!r},
                   LIBRARY_PATH='/usr/local/nvidia/lib64:' + os.environ.get('LIBRARY_PATH', ''),
                   LD_LIBRARY_PATH='/usr/local/nvidia/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''))
        cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL_PATH, '--served-model-name', '{SERVED_MODEL}',
               '--host', '127.0.0.1', '--port', '1234', '--max-model-len', '65536', '--gpu-memory-utilization', '0.92',
               '--enable-auto-tool-choice', '--enable-prefix-caching', '--generation-config', 'vllm'] + {PRESET["vllm_flags"]!r}
        if env.get('TIKTOKEN_RS_CACHE_DIR') and not os.path.isdir(env['TIKTOKEN_RS_CACHE_DIR']):
            _alt = '/kaggle/input/' + env['TIKTOKEN_RS_CACHE_DIR'].rstrip('/').split('/')[-1]
            env['TIKTOKEN_RS_CACHE_DIR'] = _alt if os.path.isdir(_alt) else env['TIKTOKEN_RS_CACHE_DIR']
            print('tiktoken cache dir:', env['TIKTOKEN_RS_CACHE_DIR'], os.listdir(env['TIKTOKEN_RS_CACHE_DIR']) if os.path.isdir(env['TIKTOKEN_RS_CACHE_DIR']) else 'MISSING')
        log = open(f'{{WORK}}/vllm-server.log', 'w')
        VLLM = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if VLLM.poll() is not None:
                _log = open(f'{{WORK}}/vllm-server.log').read()
                _err = [l for l in _log.splitlines() if any(k in l for k in ('FAILED', 'error:', 'Error', 'cannot find', 'RuntimeError', 'assert'))]
                raise RuntimeError('vLLM exited. Error lines:\\n' + '\\n'.join(_err[:40]) + '\\n--- tail ---\\n' + _log[-3000:])
            try:
                urllib.request.urlopen('http://127.0.0.1:1234/v1/models', timeout=5).read(); break
            except Exception:
                if time.time() - t0 > 1500:
                    raise TimeoutError(open(f'{{WORK}}/vllm-server.log').read()[-4000:])
                time.sleep(5)
        print(f'vLLM ready after {{time.time()-t0:.0f}}s')
        req = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions', data=json.dumps({{'model': '{SERVED_MODEL}', 'max_tokens': 400,
              'messages': [{{'role': 'user', 'content': 'Say hello in five words.'}}], **{PRESET["extra_body"]!r}}}).encode(),
              headers={{'Content-Type': 'application/json'}})
        try:
            _m = json.loads(urllib.request.urlopen(req, timeout=300).read())['choices'][0]['message']
            print('smoke:', repr((_m.get('content') or '')[:200]), '| reasoning:', repr((_m.get('reasoning_content') or _m.get('reasoning') or '')[:120]))
        except Exception as _e:   # the smoke chat is informational; the game run below is the real test
            print('smoke chat failed:', repr(_e)[:300])
        """))

    play_cell = code_cell(dedent(f"""\
        import math, os, sys, time, urllib.request, logging
        from pathlib import Path
        sys.path.insert(0, '/kaggle/working')
        logging.basicConfig(level=logging.WARNING)
        from arcnav.runner import run, make_arcade, DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG, model='{SERVED_MODEL}', base_url='http://127.0.0.1:1234/v1', verbose=True,
                   extra_body={PRESET["extra_body"]!r}, max_tokens={PRESET["max_tokens"]})
        out_dir = Path('/kaggle/working/arcnav-results')
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
            cfg.update(jobs={RERUN_JOBS}, max_minutes=max(20, min({RERUN_MAX_MINUTES_PER_GAME}, total_left / waves)), deadline=T0 + {RERUN_TOTAL_MINUTES} * 60)
            print(f'rerun: {{len(games)}} games, {{cfg["jobs"]}} concurrent, {{cfg["max_minutes"]:.0f}} min/game, waves={{waves}}')
            run(games, cfg, out_dir=out_dir, tag='kaggle-rerun', arc=arc)   # the gateway records actions and emits submission.parquet
        else:
            arc = make_arcade(environments_dir='{COMP}/environment_files')
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

    return {
        "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py", "pygments_lexer": "ipython3"},
                     "kaggle": {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False, "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}},
        "nbformat_minor": 4, "nbformat": 4,
        "cells": [markdown_cell("# ARC-AGI-3 — arcnav submission (SungsuHyun)\n\n"
                                "Our solver: a tool-using LLM agent (Qwen3.6-27B-FP8 served by vLLM inside the notebook) with a python sandbox, "
                                "a navigation helper (avatar/move learning, coarse map, BFS routing, targets, action gauge), solver synthesis "
                                "(`propose_solver`: the model writes a persistent `solve()` that the harness verifies and runs without further model calls) "
                                "and failure policies. Sources are embedded from `arcnav/` by `scripts/build_arcnav_notebook.py`; do not edit cells here.\n\n"
                                "Commit = 2-game offline smoke (15 min); the competition rerun plays every gateway game."),
                  setup_cell, vllm_cell, play_cell],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    METADATA_PATH.write_text(json.dumps({
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": NOTEBOOK_PATH.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [WHEELHOUSE_REF] + ([PRESET["dataset_model"]] if PRESET["dataset_model"] else []) + PRESET.get("extra_datasets", []), "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [PRESET["model_source"]] if PRESET["model_source"] else [], "machine_shape": MACHINE_SHAPE}, indent=2) + "\n")
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} ({NOTEBOOK_PATH.stat().st_size // 1024} KB) and {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
