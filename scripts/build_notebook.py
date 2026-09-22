"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"), extended (v005) with an
offline LLM stack:

  Cell 1: install the `arc-agi` wheel from the offline competition dataset,
          plus vLLM from the attached wheels kernel output (if present).
  Cell 2: write `my_agent.py` to /tmp/ — its body is THIS file.
  Cell 3: locate the attached Kaggle Model (QWEN_MODEL_PATH) and export env.
  Cell 4: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 5: otherwise (commit / save-and-run-all): smoke-test the planner on
          the real hardware (load model, one plan call, print timing) and
          write a dummy submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THESE TO PICK YOUR KAGGLE ACCELERATOR + ATTACHED RESOURCES
# Accelerator options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — required for the 35B FP8 planner.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "rtx6000"

# Kaggle Models to attach: "owner/model/framework/variation/version".
# The agent auto-discovers the checkpoint under /kaggle/input/models/.
# GPTQ-int4 mirror (22.8GB): the path verified locally on sm_120 Blackwell
# (Marlin kernels, no JIT). The FP8 mirror
# ("wukeneth/qwen3-5-35b-a3b-fp8/transformers/qwen3-5-35b-a3b-fp8/1", 37.5GB)
# hit vLLM's DeepGEMM nvcc>=12.9 requirement on Kaggle (kernel v5).
MODEL_SOURCES = [
    "awooooo/qwen3-5-35b-a3b-gptq-int4/other/gptq-int4/1",
]
# Notebook outputs to attach (our vLLM wheel cache, built by notebooks/wheels/).
KERNEL_SOURCES = [
    "sungsuhyun/arc3-wheels-vllm",
]
DATASET_SOURCES: list[str] = []

# Smoke test at commit time (not rerun): load the model and run one plan.
# Set False to make commits fast/cheap once the stack is known-good.
SMOKE_TEST = False

# Planner backend forced in the rerun: "" = auto (vllm if a model is attached),
# "none" = L2 off (python layers only). Three 25-game sweeps (v006/v009/v010)
# showed the LLM as a small net loss, so the leaderboard runs with it off;
# the model stays attached so this is a one-line flip.
PLANNER_BACKEND = "none"

# `shape` is the kernel-metadata `machine_shape` the Kaggle API actually honours
# (the notebook-level "accelerator" alone is ignored and you silently get T4x2).
_ACCELERATORS = {
    "cpu":     {"name": "none",             "gpu": False, "shape": None},
    "t4":      {"name": "nvidiaTeslaT4",    "gpu": True,  "shape": "NvidiaTeslaT4"},
    "p100":    {"name": "nvidiaTeslaP100",  "gpu": True,  "shape": "NvidiaTeslaP100"},
    "rtx6000": {"name": "nvidiaRtxPro6000", "gpu": True,  "shape": "NvidiaRtxPro6000"},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"

FRAMEWORK_INIT = '''from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    'random': Random,
    'myagent': MyAgent,
}
'''


SMOKE_SCRIPT = """import json, sys, time
sys.path.insert(0, sys.argv[1])
from agents.templates.my_agent import QwenPlanner, MyAgent
p = QwenPlanner(MyAgent.MODEL)
print('backend =', p.backend, '| model =', p.model, '| path =', p.model_path,
      ('| note: ' + p.last_error) if p.last_error else '')
obs = ('[PROGRESS] level 0/7 state=NOT_FINISHED\\n[AVAILABLE] 1,2,3,4\\n'
       '[OBJECTS] color/size/bbox/center\\n c9 s15 (34,47)-(38,49) ctr(36,48)\\n'
       ' c12 s10 (34,45)-(38,46) ctr(36,45)\\n c11 s84 (13,61)-(54,62) ctr(33,61)\\n'
       '[RECENT]\\n a1 ACTION1: 52 cells; moved c9s15@36,43 (0,-5); resized c11 s84->82 @33,61\\n'
       ' a2 ACTION3: 52 cells; moved c9s15@31,43 (-5,0); resized c11 s82->80 @33,61\\n'
       '[ACTION_STATS changed/tried] ACTION1:1/1 ACTION3:1/1')
for i in range(3):
    t0 = time.time(); plan = p.plan(obs); dt = time.time() - t0
    print(f'call {i}: {dt:.1f}s (load {p.load_s:.0f}s) plan={plan}')
    print('   hypothesis:', p.last_hypothesis, '| error:', p.last_error or '-')
print(json.dumps({'backend': p.backend, 'calls': p.calls, 'errors': p.errors,
                  'load_s': round(p.load_s, 1), 'total_s': round(p.total_s, 1)}))
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(dedent(f"""\
        import glob, os, subprocess, sys, time
        t0 = time.time()
        if {PLANNER_BACKEND!r}:
            os.environ['QWEN_BACKEND'] = {PLANNER_BACKEND!r}
        !pip install -q --no-index --find-links {COMP}/arc_agi_3_wheels arc-agi python-dotenv

        # v005: offline vLLM from the attached wheels-kernel output (optional).
        wheel_dirs = [os.path.dirname(m) for m in glob.glob('/kaggle/input/**/MANIFEST.json', recursive=True)]
        if wheel_dirs and os.getenv('QWEN_BACKEND') != 'hf':
            print('wheels:', wheel_dirs[0], len(os.listdir(wheel_dirs[0])), 'files')
            r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
                                '--find-links', wheel_dirs[0], 'vllm'],
                               capture_output=True, text=True)
            print('pip install vllm ->', r.returncode, (r.stderr or r.stdout)[-1500:])
            if r.returncode == 0:
                # The Kaggle image ships a pillow whose leftover files break the
                # upgraded one ("cannot import name '_Ink' from 'PIL._typing'"):
                # purge and reinstall pillow cleanly from the wheel cache.
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'pillow'], capture_output=True)
                subprocess.run(['rm', '-rf'] + glob.glob('/usr/local/lib/python3.12/dist-packages/PIL')
                               + glob.glob('/usr/local/lib/python3.12/dist-packages/pillow-*'), capture_output=True)
                r2 = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
                                     '--find-links', wheel_dirs[0], 'pillow'], capture_output=True, text=True)
                print('pip reinstall pillow ->', r2.returncode, (r2.stderr or r2.stdout)[-300:])
            if r.returncode != 0:
                # Fallback: at least a recent transformers for the hf backend.
                r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
                                    '--find-links', wheel_dirs[0], 'transformers'],
                                   capture_output=True, text=True)
                print('pip install transformers ->', r.returncode, (r.stderr or r.stdout)[-800:])
        else:
            print('no wheels attached; planner will use transformers (hf) backend')
        r = subprocess.run([sys.executable, '-c',
                            'import vllm, transformers, torch, PIL.ImageDraw; from vllm import LLM; '
                            'print("vllm", vllm.__version__, "| transformers", transformers.__version__, '
                            '"| torch", torch.__version__, "| cuda", torch.cuda.is_available())'],
                           capture_output=True, text=True)
        print('import check ->', r.returncode, r.stdout.strip(), r.stderr.strip()[-600:])
        print(f'[install] {{time.time()-t0:.0f}}s')
        """))

    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    env_cell = code_cell(dedent(f"""\
        import glob, os, subprocess
        # Locate the attached Kaggle Model checkpoint (HF format) and export it
        # for the agent. `!` shell commands below inherit os.environ.
        cands = sorted(os.path.dirname(c) for c in glob.glob('/kaggle/input/models/**/config.json', recursive=True)
                       if glob.glob(os.path.join(os.path.dirname(c), '*.safetensors')))
        if cands:
            os.environ['QWEN_MODEL_PATH'] = cands[0]
        if {PLANNER_BACKEND!r}:
            os.environ['QWEN_BACKEND'] = {PLANNER_BACKEND!r}
        print('QWEN_MODEL_PATH =', os.environ.get('QWEN_MODEL_PATH'), '| QWEN_BACKEND =', os.environ.get('QWEN_BACKEND', 'auto'))
        # JIT kernels (DeepGEMM/FlashInfer) need a recent nvcc; prefer the one
        # shipped by the nvidia-cuda-nvcc wheel from our cache over the image's.
        nvccs = sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/cu*/bin/nvcc'))
        if nvccs:
            os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(nvccs[-1]))
            os.environ['PATH'] = os.path.dirname(nvccs[-1]) + ':' + os.environ.get('PATH', '')
        print('CUDA_HOME =', os.environ.get('CUDA_HOME'))
        print(subprocess.run(['bash', '-lc', 'nvcc --version | tail -2'], capture_output=True, text=True).stdout)
        print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                             capture_output=True, text=True).stdout)
        """))

    run_cell = code_cell(dedent(f"""\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r {COMP}/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write({FRAMEWORK_INIT!r})

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """))

    # The smoke test runs in a fresh `python` subprocess (like main.py in the
    # rerun): the IPython kernel already has the image's old PIL modules
    # imported, and the vllm import chain would hit them after the reinstall.
    smoke_cell = code_cell(dedent(f"""\
        import os, sys, shutil, subprocess
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Save-and-run-all (commit) mode: emit a dummy submission so the
            # commit succeeds. The real submission.parquet is produced by the
            # gateway during competition rerun.
            import pandas as pd
            submission = pd.DataFrame(
                data=[['1_0', '1', True, 1]],
                columns=['row_id', 'game_id', 'end_of_game', 'score'])
            submission.to_parquet('/kaggle/working/submission.parquet', index=False)

            if {SMOKE_TEST!r}:
                # v005 smoke test: import the agent through the framework and
                # run a few planner calls on a canned observation, on real hardware.
                fw = '/kaggle/working/fw'
                if not os.path.isdir(fw):
                    shutil.copytree('{COMP}/ARC-AGI-3-Agents', fw)
                shutil.copy('/tmp/my_agent.py', fw + '/agents/templates/my_agent.py')
                open(fw + '/agents/__init__.py', 'w').write({FRAMEWORK_INIT!r})
                open('/tmp/smoke.py', 'w').write({SMOKE_SCRIPT!r})
                r = subprocess.run([sys.executable, '/tmp/smoke.py', fw], capture_output=True, text=True)
                print(r.stdout[-6000:])
                print(r.stderr[-3000:])
        """))

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: {sorted(_ACCELERATORS)}")
    accel = _ACCELERATORS[ACCELERATOR]

    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python",
                              "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": accel["name"], "isInternetEnabled": False,
                       "isGpuEnabled": accel["gpu"], "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            write_agent_cell,
            env_cell,
            run_cell,
            smoke_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR}, models: {len(MODEL_SOURCES)}, "
          f"kernels: {len(KERNEL_SOURCES)}, smoke_test: {SMOKE_TEST}, "
          f"planner: {PLANNER_BACKEND or 'auto'})")

    # Keep notebooks/kernel-metadata.json in sync (GPU flag + attached sources).
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = {
            "enable_gpu": _ACCELERATORS[ACCELERATOR]["gpu"],
            "machine_shape": _ACCELERATORS[ACCELERATOR]["shape"],
            "model_sources": MODEL_SOURCES,
            "kernel_sources": KERNEL_SOURCES,
            "dataset_sources": DATASET_SOURCES,
        }
        changed = [k for k, v in wanted.items() if meta.get(k) != v]
        if changed:
            meta.update({k: v for k, v in wanted.items() if v is not None})
            if wanted["machine_shape"] is None:
                meta.pop("machine_shape", None)
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced {', '.join(changed)} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
