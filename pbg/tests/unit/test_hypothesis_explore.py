"""Hypothesis policy pieces: the prompt survives a RESET in the level log (GAME_OVER mid-level), and the level-1
exploration never presses the same button twice in a row and respects its cap."""
from pbg.core.types import Action
from pbg.hypothesis.contract import hypothesis_prompt
from pbg.hypothesis.explore import explore_level
from pbg.perception import Perception

from pbg.tests.unit.synthetic import frame


def _transition(p, before, action, after, tid, before_frame=None, after_frame=None):
    return p.transition(before, action, after, before_frame=before_frame, after_frame=after_frame, status_change=None, tid=tid, level=1, step_idx=0)


def test_prompt_renders_reset_transition():
    p = Perception()
    f0, f1 = frame(0), frame(1, agent=(30, 34))
    s0 = p.parse(f0, None); s1 = p.parse(f1, s0)
    moved = _transition(p, s0, Action.button(4), s1, "g:1:1", f0, f1)
    reset = _transition(p, s1, Action.reset(), p.parse(f0, None), "g:1:2", None, f0)
    msgs = hypothesis_prompt(scene=s0, grid=f0.grid, log=[moved, reset], available=[Action.button(i) for i in (1, 2, 3, 4)],
                             prior_signatures="", previous_code=None, counterexamples=[], level=1)
    assert "RESET -> the level restarted" in msgs[1]["content"]


class _Status:
    state = "NOT_FINISHED"


class _Env:
    def status(self):
        return _Status()


class _FakeSession:
    """Buttons only; every press 'changes' nothing, so the stale stop must not fire before the cap here (stale > cap)."""
    def __init__(self):
        self.level = 1; self.env = _Env(); self.scene = Perception().parse(frame(0), None)

    def finished(self):
        return False

    def timed_out(self):
        return False

    def available_actions(self):
        return [Action.button(i) for i in (1, 2, 3, 4)]


def test_explore_no_repeated_button_and_cap():
    s = _FakeSession(); pressed = []

    def act(a, kind):
        pressed.append(a)
        f = frame(len(pressed))
        return Perception().transition(s.scene, a, s.scene, before_frame=f, after_frame=f, status_change=None, tid=f"g:1:{len(pressed)}", level=1, step_idx=len(pressed))
    out = explore_level(s, act, cap=10, stale=99, seed=3)
    assert out["actions"] == 10 and len(pressed) == 10
    assert all(pressed[i].id != pressed[i - 1].id for i in range(1, len(pressed)))
