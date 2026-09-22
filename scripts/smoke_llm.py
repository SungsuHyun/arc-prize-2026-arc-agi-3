"""Local GPU smoke test for the in-process planner backends (vllm / hf).

Mirrors the commit-time smoke cell of the Kaggle notebook, but runs on the
local GPU: import the agent through the vendored framework, force the
in-process backend via QWEN_MODEL_PATH, run a few planner calls on a canned
observation and print load time / latency / parsed plan.

Usage (needs .venv-llm, see `make llm-venv`):
    .venv-llm/bin/python scripts/smoke_llm.py --model-path ~/models/qwen3.5-35b-a3b-gptq-int4
    .venv-llm/bin/python scripts/smoke_llm.py --model-path ... --backend hf --agent experiments/v006-nav-memory/agent.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"

OBS = """[PROGRESS] level 0/7 state=NOT_FINISHED
[AVAILABLE] 1,2,3,4
[PLAYER] c9+c12 s25 bbox(34,45)-(38,49) ctr(36,47) — your avatar
[MOVES] ACTION1=(+0,-5)  ACTION2=(+0,+5)  ACTION3=(-5,+0)  ACTION4=(+5,+0)
[MAP] cell=5 origin=(4,0); char=(col,row) block majority color hex, P=player; pixel x=col*5+4, y=row*5+0
  0 444444444444
  1 444444544444
  2 444444544444
  3 444444344444
  4 444444344444
  5 443333333344
  6 443334333344
  7 443334333344
  8 444344333344
  9 444333P33344
 10 544444444444
 11 554444444444
 12 553bbbbbbb55
[NAV] avatar block (col 6,row 9); floor colors: [3]; background c4; walls found: 0
[TARGETS] id color size pixel block path-length
 T1 c0 s3 @(21,31) block(3,6) 6 moves
 T2 c5 s43 @(35,12) block(6,2) 7 moves
[GAUGE] shrinking/growing bar (likely energy/timer)
 c11 bar 84→80 cells, -2 per action (~40 actions until empty)
[OBJECTS] color/size/bbox/center (largest first, background omitted)
 c3 s892 (14,8)-(53,49) ctr(35,34)
 c5 s43 (33,9)-(39,15) ctr(35,12)
 c9 s15 (34,47)-(38,49) ctr(36,48)
 c12 s10 (34,45)-(38,46) ctr(36,45)
 c0 s3 (21,31)-(22,32) ctr(21,31)
[RECENT]
 a1 ACTION1: 52 cells; moved c9s15@36,43 (0,-5); resized c11 s84→82 @33,61
 a2 ACTION2: 52 cells; moved c9s15@36,48 (0,+5); resized c11 s82→80 @33,61
[ACTION_STATS changed/tried] ACTION1:1/1 ACTION2:1/1"""


def force_cuda_platform_without_nvml() -> None:
    """Local-dev workaround: after a driver upgrade without reboot, NVML fails
    ("Driver/library version mismatch") while CUDA itself works. vLLM detects
    the CUDA platform through NVML and would end up with no device; its
    cuda.py already falls back to a non-NVML platform class, so we only need
    to force the detection step. No-op when NVML works or torch/vllm are absent."""
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return
    except Exception:
        pass
    try:
        import torch
        if not torch.cuda.is_available():
            return
        import vllm.platforms as vp
        vp.builtin_platform_plugins["cuda"] = lambda: "vllm.platforms.cuda.CudaPlatform"
        vp._current_platform = None          # drop the already-resolved UnspecifiedPlatform
        assert vp.current_platform.device_type == "cuda"
        # The engine core normally runs in a subprocess where this patch would
        # not apply; keep it in-process for the smoke test.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        print("note: NVML unavailable (driver/library mismatch) — forcing vLLM non-NVML CUDA platform")
    except Exception as e:
        print("note: could not force CUDA platform:", type(e).__name__, e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="HF checkpoint directory")
    ap.add_argument("--backend", default=None, choices=[None, "vllm", "hf"],
                    help="force a backend (default: auto = vllm if importable)")
    ap.add_argument("--agent", default=str(ROOT / "agent" / "my_agent.py"))
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--gpu-util", default=None, help="QWEN_GPU_UTIL for vllm (default 0.88)")
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path):
        raise SystemExit(f"model path not found: {model_path}")
    os.environ["QWEN_MODEL_PATH"] = model_path
    if args.backend:
        os.environ["QWEN_BACKEND"] = args.backend
    if args.gpu_util:
        os.environ["QWEN_GPU_UTIL"] = args.gpu_util

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print("GPU:", (smi.stdout or smi.stderr).strip())
    weights = sum(os.path.getsize(f) for f in Path(model_path).glob("*.safetensors"))
    print(f"model: {model_path} ({weights / 2**30:.1f} GiB safetensors)")

    force_cuda_platform_without_nvml()
    # venv-local tools (ninja for JIT builds) must be on PATH for subprocess calls
    os.environ["PATH"] = str(Path(sys.prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
    # JIT kernels (flashinfer etc.) want nvcc; the venv ships one via the
    # nvidia-cuda-nvcc wheel when there is no system CUDA toolkit.
    if not os.getenv("CUDA_HOME") and not os.path.isdir("/usr/local/cuda"):
        cands = sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))
        if cands:
            os.environ["CUDA_HOME"] = str(cands[-1].parent.parent)
            print("note: CUDA_HOME =", os.environ["CUDA_HOME"])
    sys.path.insert(0, str(VENDOR))
    spec = importlib.util.spec_from_file_location("smoke_agent", args.agent)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.QwenPlanner(mod.MyAgent.MODEL)
    print(f"backend = {p.backend} | model = {p.model} | enabled = {p.enabled}"
          + (f" | note: {p.last_error}" if p.last_error else ""))
    for i in range(args.calls):
        t0 = time.time()
        plan = p.plan(OBS)
        dt = time.time() - t0
        print(f"call {i}: {dt:.1f}s (load {p.load_s:.0f}s) plan={plan}")
        print("   hypothesis:", p.last_hypothesis or "-", "| error:", p.last_error or "-")
    print(json.dumps({"backend": p.backend, "calls": p.calls, "errors": p.errors,
                      "load_s": round(p.load_s, 1), "total_s": round(p.total_s, 1)}))


if __name__ == "__main__":
    main()
