#!/usr/bin/env python
"""Build notebooks/pbg/pbg_submission.ipynb: the pbg hypothesis policy (look -> hypothesise -> test -> verify at pixel level ->
revise -> plan, with level-1 exploration) and an in-notebook vLLM server (Qwen3.8-Flash-Next NVFP4) for the ARC-AGI-3 competition rerun.

Attached inputs (kernel-metadata.json): the competition (arc-agi wheels + environment_files), the vLLM wheelhouse dataset and the
HF snapshot of the model. The pbg package is embedded (sources, yaml configs, priors), so no source dataset is needed.

Commit ("Save & Run All") = quota-safe smoke: SMOKE_GAMES x SMOKE_MINUTES offline. A competition rerun (KAGGLE_IS_COMPETITION_RERUN)
plays every gateway game, RERUN_JOBS at a time, within RERUN_TOTAL_MINUTES.

    .venv/bin/python scripts/build_pbg_notebook.py [--explore 20]      # make pbg-notebook / make pbg-submit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_arcnav_notebook as base   # setup / vLLM cells and constants are shared with the arcnav notebook

KERNEL_ID = "sungsuhyun/arc3-pbg"
KERNEL_TITLE = "arc3-pbg"
OUT_DIR = ROOT / "notebooks" / "pbg"
NOTEBOOK_PATH = OUT_DIR / "pbg_submission.ipynb"
METADATA_PATH = OUT_DIR / "kernel-metadata.json"
SMOKE_GAMES = ["r11l", "vc33"]
SMOKE_MINUTES = 12
RERUN_JOBS = 12
RERUN_TOTAL_MINUTES = 470          # Kaggle rerun limit is 540 min including the model load
RERUN_MAX_MINUTES_PER_GAME = 120
LLM_CALLS_PER_GAME = 600           # the local run used ~6 calls/min/game (r11l: 58 calls in 10 min)
# Qwen3.8-Flash-Next (qwen4_exp, NVFP4) on the 96 GB RTX Pro 6000. The vLLM cell comes from base.build(), which reads
# base's module globals, so point them all at this preset before calling it (the preset also pins a vLLM 0.27.1 wheelhouse,
# since qwen4_exp needs a vLLM newer than the qwen27b wheelhouse's 0.19).
PRESET = base.PRESETS["qwen38fn"]
base.PRESET = PRESET
base.WHEELHOUSE_REF = PRESET.get("wheelhouse") or base.WHEELHOUSE_REF
base.MODEL_REF = PRESET["model_source"] or PRESET["dataset_model"]


def sources() -> dict[str, str]:
    """Every file the package needs at run time, keyed by its path relative to the repository root."""
    out = {}
    for p in sorted((ROOT / "pbg").rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or "__pycache__" in rel.parts or rel.parts[1] in ("tests", "data") or p.suffix not in (".py", ".yaml", ".json"):
            continue
        if p.name == "llm-opus.yaml":
            continue
        out[str(rel)] = p.read_text()
    for name in ("__init__.py", "ids.py"):          # the runner stamps a per-game hash (scripts/eval_viewer/ids.py)
        out[f"scripts/eval_viewer/{name}"] = (ROOT / "scripts" / "eval_viewer" / name).read_text()
    return out


def image_vllm_cell() -> dict:
    """vLLM launched from an exported Docker image (vllm/vllm-openai:qwen38-flash-next), the only runtime that knows the
    qwen4_exp architecture. The image layers are attached as blobs; extract them, then run the image's vLLM (its own
    Transformers/vLLM) against the separately attached NVFP4 weights. Serves local-model on 127.0.0.1:1234 like the
    wheelhouse cell, so the play cell and the gateway config are unchanged."""
    runtime_ref = PRESET["wheelhouse"]
    model_ref = PRESET["model_source"]
    return base.code_cell(dedent(f"""\
        import glob, json, os, subprocess, sys, tarfile, time, urllib.request
        WORK = '/kaggle/working'
        def _find(ref):
            owner, slug = ref.split('/', 1)
            for p in (f'/kaggle/input/{{slug}}', f'/kaggle/input/datasets/{{owner}}/{{slug}}'):
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(ref)
        RUNTIME = _find('{runtime_ref}')
        _mp = glob.glob('/kaggle/input/models/**/config.json', recursive=True)
        MODEL_PATH = os.path.dirname(_mp[0]) if _mp else _find('{model_ref}')
        cfgs = glob.glob(MODEL_PATH + '/**/config.json', recursive=True)
        MODEL_PATH = os.path.dirname(cfgs[0]) if cfgs else MODEL_PATH
        print('runtime:', RUNTIME, '| model:', MODEL_PATH)
        print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())

        # Extract the exported image layers (manifest order; later layers overlay earlier ones) into a scratch root.
        ROOT = '/tmp/vllm-image'
        if not os.path.exists(ROOT + '/.done'):
            man = json.load(open(RUNTIME + '/runtime-manifest.json'))
            layers = sorted(man['selected_layers'], key=lambda l: l['index'])
            os.makedirs(ROOT, exist_ok=True)
            for l in layers:
                cands = [RUNTIME + '/' + l['file'] + '.blob', RUNTIME + '/' + l['file']] + glob.glob(RUNTIME + '/layer-%d-*' % l['index'])
                blob = next(p for p in cands if os.path.exists(p))
                print('extract', os.path.basename(blob), f"({{l['size']//(1<<20)}} MiB) ...", flush=True)
                with tarfile.open(blob, 'r:gz') as tf:
                    for _mem in tf:   # per-member so one odd entry (device, absolute symlink, .wh. whiteout) never kills the layer
                        try:
                            tf.extract(_mem, ROOT, filter='tar')
                        except Exception:
                            pass
            open(ROOT + '/.done', 'w').write('ok')
        print('image root size MiB:', subprocess.run(['du', '-sm', ROOT], capture_output=True, text=True).stdout.split()[0])

        # Locate the image's python site-packages (the dir that holds vllm). Its native extensions must match this kernel's
        # python ABI (both cp312 for vllm/vllm-openai + Kaggle); warn otherwise.
        _v = (glob.glob(ROOT + '/**/site-packages/vllm/__init__.py', recursive=True) or
              glob.glob(ROOT + '/**/dist-packages/vllm/__init__.py', recursive=True) or   # vllm/vllm-openai installs into dist-packages
              glob.glob(ROOT + '/**/vllm/__init__.py', recursive=True))
        _v = [p for p in _v if os.path.isdir(os.path.dirname(p) + '/entrypoints')] or _v   # a real install, not a nested copy
        if not _v:
            raise RuntimeError('no vllm in the extracted image. python dirs: ' + str(glob.glob(ROOT + '/**/python3*/', recursive=True)[:20]))
        SITE = os.path.dirname(os.path.dirname(_v[0]))
        import re as _re
        _m = _re.search(r'python3\\.(\\d+)', SITE)
        if _m and int(_m.group(1)) != sys.version_info.minor:
            print(f'WARNING: image python 3.{{_m.group(1)}} != kernel python 3.{{sys.version_info.minor}} (ABI mismatch likely)')
        print('image vllm site-packages:', SITE)
        # CUDA libs: torch ships its own under site-packages/nvidia/*/lib; add them plus the Kaggle driver stub.
        _nvlibs = ':'.join(sorted(set(os.path.dirname(p) for p in glob.glob(SITE + '/nvidia/**/lib', recursive=True)) |
                                  set(glob.glob(SITE + '/nvidia/*/lib'))))
        _ld = '/usr/local/nvidia/lib64:' + _nvlibs + ':' + SITE + '/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        _ram_gb = round(int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal'))) / (1 << 20), 1)
        print(f'host RAM: {{_ram_gb}} GiB (the FP8 PLE n-gram table offloaded to CPU needs ~48 GiB + headroom; recipe wants >= 64)')
        # Qwen3.8-Flash-Next has a 51B-param PLE n-gram embedding. On one 96 GB GPU it must be offloaded to host RAM
        # (VLLM_PLE_CPU_OFFLOAD=1): NVFP4 experts then use ~89 GiB VRAM. Recipe: max-model-len 32768, max-num-seqs 36.
        env = dict(os.environ, PYTHONPATH=SITE, USE_TF='0', TRANSFORMERS_NO_TF='1', TRANSFORMERS_NO_TORCHVISION='1',
                   VLLM_NO_USAGE_STATS='1', VLLM_PLE_CPU_OFFLOAD='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   LIBRARY_PATH='/usr/local/nvidia/lib64:' + os.environ.get('LIBRARY_PATH', ''), LD_LIBRARY_PATH=_ld)
        cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL_PATH, '--served-model-name', '{base.SERVED_MODEL}',
               '--host', '127.0.0.1', '--port', '1234', '--max-model-len', '32768', '--max-num-seqs', '36', '--gpu-memory-utilization', '0.93',
               '--enable-auto-tool-choice', '--enable-prefix-caching', '--generation-config', 'vllm'] + {PRESET["vllm_flags"]!r}
        log = open(f'{{WORK}}/vllm-server.log', 'w')
        VLLM = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if VLLM.poll() is not None:
                _log = open(f'{{WORK}}/vllm-server.log').read()
                _err = [l for l in _log.splitlines() if any(k in l for k in ('FAILED', 'error:', 'Error', 'cannot find', 'RuntimeError', 'assert', 'not recognize'))]
                raise RuntimeError('vLLM exited. Error lines:\\n' + '\\n'.join(_err[:40]) + '\\n--- tail ---\\n' + _log[-3000:])
            try:
                urllib.request.urlopen('http://127.0.0.1:1234/v1/models', timeout=5).read(); break
            except Exception:
                if time.time() - t0 > 1800:
                    raise TimeoutError(open(f'{{WORK}}/vllm-server.log').read()[-4000:])
                time.sleep(5)
        print(f'vLLM ready after {{time.time()-t0:.0f}}s')
        req = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions', data=json.dumps({{'model': '{base.SERVED_MODEL}', 'max_tokens': 400,
              'messages': [{{'role': 'user', 'content': 'Say hello in five words.'}}], **{PRESET["extra_body"]!r}}}).encode(),
              headers={{'Content-Type': 'application/json'}})
        try:
            _msg = json.loads(urllib.request.urlopen(req, timeout=300).read())['choices'][0]['message']
            print('smoke:', repr((_msg.get('content') or '')[:200]))
        except Exception as _e:
            print('smoke chat failed:', repr(_e)[:300])
        """))


def build(explore: int) -> dict:
    files = sources()
    setup_cell = base.code_cell(dedent(f"""\
        import json, os, subprocess, sys, time
        T0 = time.time()
        RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
        WORK = '/kaggle/working'
        print('competition rerun:', RERUN)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', '{base.COMP}/arc_agi_3_wheels', 'arc-agi'], check=True)
        FILES = json.loads({json.dumps(json.dumps(files))})
        for rel, body in FILES.items():
            os.makedirs(os.path.dirname(f'{{WORK}}/{{rel}}'), exist_ok=True)
            open(f'{{WORK}}/{{rel}}', 'w').write(body)
        sys.path.insert(0, WORK)
        os.environ['PBG_LLM_CONFIG'] = f'{{WORK}}/pbg/llm/llm-kaggle.yaml'
        import arc_agi, yaml, pbg.harness.online_runner, pbg.hypothesis.policy
        print(f'pbg embedded ({{len(FILES)}} files) | arc_agi ok |', f'{{time.time()-T0:.0f}}s')
        """))
    vllm_cell = image_vllm_cell() if PRESET.get("runtime") == "image" else base.build()["cells"][2]
    play_cell = base.code_cell(dedent(f"""\
        import math, os, sys, time, urllib.request, logging
        from pathlib import Path
        sys.path.insert(0, '/kaggle/working')
        logging.basicConfig(level=logging.WARNING)
        from pbg.harness.online_runner import make_arcade, run
        cfg = {{'policy': 'hypothesis', 'explore': {explore}, 'no_llm': False, 'verbose': False, 'max_levels': 20, 'budget': None,
               'llm_calls_per_game': {LLM_CALLS_PER_GAME}}}
        out_dir = Path('/kaggle/working/pbg-results'); mem = Path('/kaggle/working/pbg-memory')
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
            # per game: its wave's share of the time left, minus the 5-min hang allowance of the runner
            cfg.update(jobs={RERUN_JOBS}, max_minutes=max(10, min({RERUN_MAX_MINUTES_PER_GAME}, total_left / waves - 6)))
            print(f'rerun: {{len(games)}} games, {{cfg["jobs"]}} concurrent, {{cfg["max_minutes"]:.0f}} min/game, waves={{waves}}')
            run(games, cfg, out_dir=out_dir, memory_root=mem, tag='kaggle-rerun', arc=arc)   # the gateway records actions and emits submission.parquet
        else:
            arc = make_arcade(environments_dir='{base.COMP}/environment_files')
            cfg.update(jobs=2, max_minutes={SMOKE_MINUTES})
            res = run({SMOKE_GAMES!r}, cfg, out_dir=out_dir, memory_root=mem, tag='kaggle-smoke', arc=arc)
            print('smoke errors:', [(g['game_id'], g.get('error')) for g in res['games'] if g.get('error')])
            import pandas as pd   # commit mode: dummy submission so the commit succeeds
            pd.DataFrame(data=[['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet('/kaggle/working/submission.parquet', index=False)
        try:
            VLLM.terminate()
        except Exception:
            pass
        print(f'done in {{(time.time()-T0)/60:.1f}} min')
        """))
    nb = base.build()
    nb["cells"] = [base.markdown_cell("# ARC-AGI-3 — pbg hypothesis agent submission (SungsuHyun)\n\n"
                                      "pbg treats an unseen game as a science problem: explore level 1 briefly (random buttons that never repeat the last one, "
                                      f"one click per object class; at most {explore} actions), then Qwen3.8-Flash-Next (NVFP4, vLLM in the notebook) writes a hypothesis of the "
                                      "whole game as executable code (roles, rules, win condition, tests); the prediction is checked against the real next frame pixel "
                                      "by pixel, counter-examples go back to the model, and A* plans only on a verified hypothesis. Sources are embedded from `pbg/` "
                                      "by `scripts/build_pbg_notebook.py`; do not edit cells here.\n\n"
                                      f"Commit = {len(SMOKE_GAMES)}-game offline smoke ({SMOKE_MINUTES} min); the competition rerun plays every gateway game."),
                   setup_cell, vllm_cell, play_cell]
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--explore", type=int, default=20, help="level-1 exploration cap (0 = off)")
    a, _ = ap.parse_known_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(a.explore), indent=1))
    METADATA_PATH.write_text(json.dumps({
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": NOTEBOOK_PATH.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [base.WHEELHOUSE_REF] + ([PRESET["dataset_model"]] if PRESET["dataset_model"] else []) + PRESET.get("extra_datasets", []), "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [PRESET["model_source"]] if PRESET.get("model_source") else [], "machine_shape": base.MACHINE_SHAPE}, indent=2) + "\n")
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} ({NOTEBOOK_PATH.stat().st_size // 1024} KB, explore={a.explore}) and {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
