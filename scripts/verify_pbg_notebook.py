#!/usr/bin/env python
"""Gate for `make pbg-submit`: refuse to push notebooks/pbg/ unless the built notebook is internally consistent.

Checks invariants derived from the builder's PRESET (not literal strings that drift):
  1. every code cell parses as Python (the vLLM cell is generated from an f-string: a bad escape would only fail on Kaggle);
  2. kernel-metadata.json attaches exactly the preset's model_source / dataset_model and wheelhouse (runtime) dataset;
  3. a runtime == "image" cell writes a sitecustomize.py that compiles, and launches vLLM with the single-GPU PLE offload
     requirements (--distributed-executor-backend mp, VLLM_PLE_CPU_OFFLOAD) -- without them the engine deadlocks or the
     PLE worker never spawns (vllm#53960);
  4. no active PYTORCH_CUDA_ALLOC_CONF=expandable_segments kwarg: it routes CUDA IPC through fd handles that the PLE
     worker cannot import in the Kaggle sandbox (pidfd_getfd EPERM, pytorch#165685).

    .venv/bin/python scripts/verify_pbg_notebook.py      # exit 0 = ok, 1 = refuse
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_pbg_notebook as bp   # noqa: E402  (import has no side effects beyond argparse.parse_known_args)

NB = bp.NOTEBOOK_PATH
META = bp.METADATA_PATH
PRESET = bp.PRESET


def fail(msg: str) -> None:
    print(f"REFUSE: {msg}")
    sys.exit(1)


def main() -> None:
    nb = json.loads(NB.read_text())
    meta = json.loads(META.read_text())
    code_cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]

    # 1. every code cell parses
    trees = []
    for i, src in enumerate(code_cells):
        try:
            trees.append(ast.parse(src))
        except SyntaxError as e:
            fail(f"code cell {i} does not parse: {e}")
    print(f"ok  {len(code_cells)} code cells parse")

    # 2. metadata matches the preset
    want_models = [PRESET["model_source"]] if PRESET.get("model_source") else []
    want_datasets = [PRESET.get("wheelhouse") or bp.base.WHEELHOUSE_REF] + ([PRESET["dataset_model"]] if PRESET.get("dataset_model") else []) + PRESET.get("extra_datasets", [])
    if meta.get("model_sources") != want_models:
        fail(f"model_sources {meta.get('model_sources')} != preset {want_models}")
    if meta.get("dataset_sources") != want_datasets:
        fail(f"dataset_sources {meta.get('dataset_sources')} != preset {want_datasets}")
    if meta.get("id") != bp.KERNEL_ID:
        fail(f"kernel id {meta.get('id')} != {bp.KERNEL_ID}")
    print(f"ok  metadata attaches {want_models + want_datasets}")

    vllm_src = code_cells[1]   # markdown, setup, vLLM, play -> code cells: setup, vLLM, play
    if PRESET.get("runtime") == "image":
        # 3a. sitecustomize text compiles
        body = None
        for n in ast.walk(trees[1]):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "write"
                    and isinstance(n.func.value, ast.Call) and "sitecustomize" in ast.unparse(n.func.value)):
                if not (len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                    fail("sitecustomize write() argument is not a single string literal")
                body = n.args[0].value
        if body is None:
            fail("image runtime cell writes no sitecustomize.py (ptrace opt-in for the PLE worker)")
        try:
            compile(body, "sitecustomize.py", "exec")
        except SyntaxError as e:
            fail(f"generated sitecustomize.py does not compile: {e}")
        if "0x59616d61" not in body:
            fail("sitecustomize.py does not call prctl(PR_SET_PTRACER, ...)")
        print("ok  sitecustomize.py compiles and sets PR_SET_PTRACER")
        # 3b. single-GPU PLE offload launch requirements
        for needle, why in (("'--distributed-executor-backend', 'mp'", "the default executor never spawns the PLE offload worker (vllm#53960)"),
                            ("VLLM_PLE_CPU_OFFLOAD='1'", "the 51B PLE table must be offloaded to host RAM on one GPU")):
            if needle not in vllm_src:
                fail(f"vLLM cell lacks {needle}: {why}")
        print("ok  vLLM launch has the mp executor and PLE CPU offload")

    # 4. no active expandable_segments kwarg (comments may mention it)
    active = [l.strip() for l in vllm_src.splitlines() if re.search(r"PYTORCH_CUDA_ALLOC_CONF\s*=\s*['\"]", l)]
    if active:
        fail(f"active PYTORCH_CUDA_ALLOC_CONF kwarg (fd-based CUDA IPC -> pidfd_getfd EPERM): {active}")
    print("ok  no active PYTORCH_CUDA_ALLOC_CONF kwarg")
    print(f"PASS {NB.relative_to(ROOT)} is consistent with preset {bp.PRESET.get('kernel', '?')} (model {want_models or PRESET.get('dataset_model')})")


if __name__ == "__main__":
    main()
