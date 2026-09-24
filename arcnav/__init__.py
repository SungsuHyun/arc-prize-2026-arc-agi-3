"""arcnav — our ARC-AGI-3 solver: a tool-using LLM harness built around
navigation helpers, solver synthesis and programmatic safety policies.

Modules: frame (observation encoding), nav (navigation helper), sandbox
(python tool), llm (OpenAI-compatible tool-calling client), prompts, solver
(persistent solver policies), agent (per-game loop), runner (local/Kaggle play).
"""
__version__ = "0.1.0"
