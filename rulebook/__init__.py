"""rulebook — a rulebook-driven ARC-AGI-3 agent (new line, 2026-09-27).

The agent keeps an explicit rulebook of hypotheses (environment facts, rules, win conditions). A deterministic predictor
derived from the recorded transitions tells, for every candidate action, what should happen; the model only chooses among
the candidates; every executed action is checked against its prediction and a mismatch triggers a rulebook review.
"""
