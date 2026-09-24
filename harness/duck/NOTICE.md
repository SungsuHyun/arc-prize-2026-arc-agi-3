# NOTICE — origin and attribution

This directory vendors the **Duck harness** and the **Tufa ARC-AGI Framework (TAAF)**
published by Tufa Labs (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit,
Michal Tesnar, Stefano Viel) as their open-sourced ARC-AGI-3 Milestone 1 solution:

- Source: https://github.com/Tufalabs/duck-harness (snapshot 2026-09-23)
- License basis: the same code was released by the authors on Kaggle as dataset
  `jeroencottaar/taaf-kaggle-source-share` under the **MIT License**, and
  `ARC3-Inference/pyproject.toml` declares `License :: OSI Approved :: MIT License`.
  The GitHub repository itself carries no LICENSE file (see upstream issue #6);
  we rely on the Kaggle MIT release and the authors' public open-source statement.
- Copyright (c) Tufa Labs. Permission is granted under the MIT License; this notice
  and the MIT permission notice must be retained in copies or substantial portions.

## Our modifications (SungsuHyun / arc-prize-2026-arc-agi-3)

Everything below is our own work on top of the harness (see git history of this repo):

- `ARC3-Inference/inference/agent/nav_helpers.py` — navigation helper (`nav`):
  avatar detection, learned movement deltas, cell-aligned coarse map, floor/wall/
  closed-target learning, BFS `path_to`, targets/frontier/gauge summaries.
- `python_tool_sandbox.py` — ships the helper source into the isolated sandbox
  (`python -I` blocked package imports, so `nav` used to be None); `propose_solver()`
  with dry-run + `predict()` verification; `solver` host message.
- `tool_agent.py` — per-turn navigation summary; stored-solver auto-run with
  failure policies (empty result, exceptions, no board change, no level progress,
  cycling, verification gate); solver status lines and proposal demands.
- `prompts.py` — `NAV_HELPER_ADDENDUM`, `SOLVER_ADDENDUM` with navigation and
  click-game solver templates.
- `tufa-arc-agi-framework/src/taaf/kaggle/taaf_kaggle_run.ipynb` — commit-mode
  smoke guard (2 games / 15 min against the competition environment files) to
  protect the weekly GPU quota; competition reruns keep the full configuration.
- `configs/local5090.json` — local single-GPU configuration.

Third-party datasets attached at deployment (public on Kaggle, each under its own
terms): `driessmit1/arc3-vllm-h100-wheelhouse-v3` (vLLM wheelhouse) and
`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (Qwen3.6-27B-FP8 weights, Qwen license).
