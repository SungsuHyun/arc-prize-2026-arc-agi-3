"""pbg — pixel board game autonomous reasoning system (docs/027, SW spec).

Layers: env -> perception -> {probe, wml, goal} -> planner, driven by the orchestrator state machine.
The LLM is used only inside wml/goal (HYPOTHESIZE state); everything else is deterministic code.
"""
__version__ = "0.1.0"
