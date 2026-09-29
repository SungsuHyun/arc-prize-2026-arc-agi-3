"""docs/029 P0: LLM model code is deadline-limited (an unbounded loop must not freeze a game thread) and the sandbox
accepts the mechanism library under the names the prompt suggests."""
import threading
import time

from pbg.core.types import Action
from pbg.hypothesis.guard import GuardedModel, ModelTimeout, call_with_deadline
from pbg.hypothesis.policy import Hypothesis
from pbg.perception import Perception
from pbg.tests.unit.synthetic import frame
from pbg.wml.sandbox import Sandbox

LOOPING = '''
HYPOTHESIS = {"summary": "loops forever"}
def spin(scene, action):
    i = 0
    while True:
        i += 1
    return scene
def build_model():
    return RuleModel([Rule("spin", lambda s, a: True, spin)], None, default="noop")
def build_goal():
    class G:
        name = "never"
        def is_goal(self, s): return False
        def progress(self, s): return 0.0
    return G()
'''


def test_call_with_deadline_interrupts_python_loop():
    def spin():
        while True:
            pass
    t0 = time.monotonic()
    try:
        call_with_deadline(spin, seconds=0.3)
        assert False, "no timeout"
    except ModelTimeout:
        pass
    assert time.monotonic() - t0 < 2.0


def test_guarded_model_returns_none_and_dies_after_two_timeouts():
    sb = Sandbox(allow_getattr=True)
    ns = sb.load_namespace(LOOPING)
    assert ns is not None, sb.last_error
    h = Hypothesis(LOOPING, ns, 1)
    h.model.seconds = 0.3
    scene = Perception().parse(frame(0), None)
    t0 = time.monotonic()
    assert h.model.predict(scene, Action.button(1)) is None
    assert h.model.predict(scene, Action.button(2)) is None
    assert h.model.timeouts == 2 and h.model.dead
    assert h.model.predict(scene, Action.button(3)) is None     # dead: returns at once
    assert time.monotonic() - t0 < 3.0
    assert "unbounded loop" in h.timeout_note()
    assert "TIMEOUTS=2" in h.summary()


def test_deadline_is_per_thread():
    """A spinning model in one thread must not trip the deadline of another thread's call."""
    results = {}

    def spin():
        while True:
            pass

    def fine():
        x = 0
        for _ in range(20000):
            x += 1
        return "ok"

    def spinner():
        try:
            call_with_deadline(spin, seconds=0.3)
        except ModelTimeout:
            results["a"] = "timeout"

    def worker():
        time.sleep(0.05)
        try:
            results["b"] = call_with_deadline(fine, seconds=5.0)
        except ModelTimeout:
            results["b"] = "timeout"

    ta, tb = threading.Thread(target=spinner), threading.Thread(target=worker)
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    assert results.get("a") == "timeout" and results.get("b") == "ok"


def test_sandbox_accepts_mechanism_imports():
    code = '''from mechanisms import move_role
import mechanism
def build_model():
    return RuleModel([Rule("m", lambda s, a: a.type == "BUTTON", move_role("agent"))], None, default="noop")
'''
    sb = Sandbox(allow_getattr=True)
    ns = sb.load_namespace(code)
    assert ns is not None and "build_model" in ns, sb.last_error
    assert ns["build_model"]() is not None
