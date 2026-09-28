"""orchestrator/states.py — the four states and the events that move between them (spec §12)."""
STATES = ("PROBE", "HYPOTHESIZE", "PLAN", "EXECUTE")
TERMINAL = ("WIN", "GAME_OVER", "BUDGET", "UNRESOLVED", "TIMEOUT")
