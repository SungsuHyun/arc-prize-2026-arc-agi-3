# arcnav — our ARC-AGI-3 agent

A tool-using LLM harness built from scratch around three ideas we found to work in
v006–v012: a **navigation helper** (learn what the movement keys do, map the floor,
route with BFS), **solver synthesis** (the model writes a persistent `solve()` that the
harness verifies and runs without further model calls) and **programmatic failure
policies** (no-change / no-progress / cycle detection, level-completion hints).

```
arcnav/
  frame.py    Frame: grid -> ascii, connected-component segmentation (nodes, adjacency, containment, HUD strips), diff summary
  nav.py      NavHelper: avatar from co-moving objects, move deltas, cell-size map, floor/wall votes, path_to, targets, frontier, gauge
  sandbox.py  persistent `python -I` child per game; JSON-lines protocol; action(), propose_solver(), nav, transitions
  llm.py      dependency-free OpenAI-compatible chat client (tool calls, reasoning field, context-length errors)
  prompts.py  system prompt + solver templates (navigation / click)
  solver.py   auto-run policy for stored solvers (12 actions per turn, 2 no-change turns, 80 no-progress actions, cycles)
  agent.py    GameSession: env stepping, model turns, context trimming, solver turns, transcripts (<game>.log / .jsonl)
  runner.py   thread-pool runner, experiments-style result JSON (experiments/arcnav/results/)
```

Run locally against a vLLM OpenAI server on :1234 (`make arcnav GAME=ls20,vc33 MINUTES=20 JOBS=2 TAG=x`).
Kaggle: `python scripts/build_arcnav_notebook.py` embeds these sources into
`notebooks/arcnav/arcnav_submission.ipynb` (in-notebook vLLM server + gateway play).

Lineage: the REPL-tool idea and the transitions-based observation follow the published
Milestone-1 harness by Tufa Labs (MIT); no code from it is used here.
