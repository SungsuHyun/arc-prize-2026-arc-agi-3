"""hypothesis/guard.py — a time limit on LLM-written model code (docs/029).

An LLM-written rule with an unbounded loop froze a game thread for 50 minutes (r11l: `slide_movers_toward_click` inside
verify()); Python threads cannot be killed, and the sandbox's CPU-limited subprocess only covers the validation run,
not the in-process calls the verifier and the planner make. This module runs every model/goal call under a per-thread
trace function that raises ModelTimeout once the deadline has passed. sys.settrace is per thread, so games running in
parallel threads do not affect each other; pure-Python loops trigger 'line' events and are interrupted, C calls (numpy)
are bounded anyway.

    GuardedModel(model, seconds)     predict / with_roles / assign_roles under the deadline; a timeout is an UNKNOWN
                                     prediction (None), counted in .timeouts; after MAX_TIMEOUTS the model is dead
                                     and every call returns None at once (no more time is spent on it)
    call_with_deadline(fn, seconds)  the same guard for any function (build_model, build_goal, is_goal, ...)"""
from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional

MAX_TIMEOUTS = 2
CHECK_EVERY = 32          # line events between clock reads (the clock is the expensive part of the tracer)


class ModelTimeout(BaseException):
    """BaseException on purpose: LLM code and RuleModel.predict catch Exception; the deadline must pass through them."""


def call_with_deadline(fn: Callable, *args: Any, seconds: float = 2.0, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs); raise ModelTimeout if its Python code runs past `seconds`."""
    deadline = time.monotonic() + seconds
    count = [0]

    def local_tracer(frame, event, arg):
        if event == "line":
            count[0] += 1
            if count[0] % CHECK_EVERY == 0 and time.monotonic() > deadline:
                raise ModelTimeout(f"model code exceeded {seconds:.1f}s in {frame.f_code.co_name} (line {frame.f_lineno})")
        return local_tracer

    def global_tracer(frame, event, arg):
        if event == "call":
            return local_tracer
        return None

    prev = sys.gettrace()
    sys.settrace(global_tracer)
    try:
        return fn(*args, **kwargs)
    finally:
        sys.settrace(prev)


class GuardedModel:
    """Proxy around an LLM-written WorldModel: every call is deadline-limited; timeouts are recorded, never raised."""

    def __init__(self, model: Any, seconds: float = 2.0):
        self._model = model
        self.seconds = float(seconds)
        self.timeouts = 0
        self.last_timeout: str = ""

    @property
    def dead(self) -> bool:
        return self.timeouts >= MAX_TIMEOUTS

    def _guarded(self, name: str, *args: Any, default: Any = None) -> Any:
        if self.dead:
            return default
        fn = getattr(self._model, name, None)
        if fn is None:
            return default
        try:
            return call_with_deadline(fn, *args, seconds=self.seconds)
        except ModelTimeout as e:
            self.timeouts += 1
            self.last_timeout = str(e)
            return default

    def predict(self, scene, action):
        return self._guarded("predict", scene, action, default=None)

    def with_roles(self, scene):
        return self._guarded("with_roles", scene, default=scene)

    def assign_roles(self, scene) -> dict:
        return self._guarded("assign_roles", scene, default={}) or {}

    def rules(self) -> list:
        try:
            return list(self._model.rules())
        except Exception:
            return []

    def __getattr__(self, name: str):      # anything else (level_scoped, verified, name, ...) passes through
        return getattr(self._model, name)
