"""docs/029 task 8: the win condition inferred from the level-up evidence must be true right after the win, false right
before it and on earlier boards, and false on the next level's board; only such predicates become goals."""
import numpy as np

from pbg.core.types import Action, Frame
from pbg.hypothesis.goals import WinEvidence, goal_prompt, propose_goals, verify_goal
from pbg.perception import Perception
from pbg.tests.unit.synthetic import frame
from pbg.wml.sandbox import Sandbox


def _evidence():
    """The agent walks right; the level ends when it reaches column 42 (the win = the agent's left edge >= 42)."""
    p = Perception()
    frames = [frame(i, agent=(30, 30 + 4 * i)) for i in range(4)]         # cols 30, 34, 38, 42
    scenes = [p.parse(frames[0], None)]
    for i in range(1, 4):
        scenes.append(p.parse(frames[i], scenes[-1]))
    ev = WinEvidence(1, "ACTION4", scenes[2], scenes[3], frames[2].grid, frames[3].grid, earlier=scenes[:2])
    nxt = p.parse(frame(9, agent=(30, 20)), None)                          # next level: the agent starts elsewhere
    return ev, nxt


class _Goal:
    def __init__(self, name, fn, prog=None):
        self.name = name; self.is_goal = fn; self.progress = prog or (lambda s: 0.5)


def _agent_col(s):
    ag = [o for o in s.objects if o.color == 12]
    return ag[0].bbox[1] if ag else -1


def test_verify_goal_accepts_the_true_condition_and_rejects_others():
    ev, nxt = _evidence(); roled = lambda s: s
    ok, why = verify_goal(_Goal("reach_right", lambda s: _agent_col(s) >= 42, lambda s: min(1.0, _agent_col(s) / 42)), roled, ev, nxt)
    assert ok, why
    ok, why = verify_goal(_Goal("always", lambda s: True), roled, ev, nxt)
    assert not ok and "before" in why
    ok, why = verify_goal(_Goal("never", lambda s: False), roled, ev, nxt)
    assert not ok and "after" in why
    ok, why = verify_goal(_Goal("too_early", lambda s: _agent_col(s) >= 42 or _agent_col(s) == 30), roled, ev, nxt)   # also true on the level's first board
    assert not ok and "earlier" in why
    ok, why = verify_goal(_Goal("current", lambda s: _agent_col(s) >= 42 or _agent_col(s) == 20), roled, ev, nxt)
    assert not ok and "current board" in why


def test_goal_prompt_carries_the_evidence():
    ev, nxt = _evidence()
    msgs = goal_prompt(ev, lambda s: s, nxt, nxt.render(), rejected=["old(x)"], feedback=["'x' is FALSE on the board right after"])
    u = msgs[1]["content"]
    assert "Level 1 was completed by the action ACTION4" in u and "Board right BEFORE" in u and "Board right AFTER" in u
    assert "NOT yet won" in u and "Current board (level 2" in u and "old(x)" in u and "failed verification" in u
    assert "def build_goal" in msgs[0]["content"]


GOOD = '''
def build_goal():
    class Goal:
        name = "agent_reaches_right"
        def is_goal(self, scene):
            ag = [o for o in scene.objects if o.color == 12]
            return bool(ag) and ag[0].bbox[1] >= 42
        def progress(self, scene):
            ag = [o for o in scene.objects if o.color == 12]
            return min(1.0, ag[0].bbox[1] / 42) if ag else 0.0
    return Goal()
'''
BAD = '''
def build_goal():
    class Goal:
        name = "anything"
        def is_goal(self, scene): return True
        def progress(self, scene): return 1.0
    return Goal()
'''


class _LLM:
    cfg = {"timeout": 5}
    def __init__(self): self.n = 0
    def chat(self, msgs, **kw):
        self.n += 1
        return "```python\n" + (GOOD if "Alternative" not in msgs[1]["content"] else BAD) + "\n```"
    def extract_code(self, text):
        from pbg.llm.gateway import extract_code
        return extract_code(text)


def test_propose_goals_keeps_only_verified_predicates():
    ev, nxt = _evidence()
    accepted, reasons = propose_goals(_LLM(), Sandbox(), ev, lambda s: s, nxt, nxt.render(), K=2)
    assert [g.name for g in accepted] == ["agent_reaches_right"]
    assert accepted[0].is_goal(ev.after) and not accepted[0].is_goal(nxt) and accepted[0].template.startswith("win:")
    assert len(reasons) == 1 and "before" in reasons[0]


def test_goal_code_without_build_goal_is_wrapped():
    from pbg.hypothesis.goals import normalise_goal_code
    cls = "class Goal:\n    name = 'x'\n    def is_goal(self, scene): return False\n    def progress(self, scene): return 0.0\n"
    fns = "name = 'reach'\ndef is_goal(scene):\n    return False\ndef progress(scene):\n    return 0.2\n"
    assert "def build_goal" in normalise_goal_code(cls) and Sandbox().load_goal(normalise_goal_code(cls)) is not None
    g = Sandbox().load_goal(normalise_goal_code(fns))
    assert g is not None and g.name == "reach" and g.progress(None) == 0.2
    assert normalise_goal_code(GOOD).strip() == GOOD.strip()   # already complete: unchanged apart from stray blank lines


def test_indented_goal_code_is_dedented_and_loads():
    from pbg.hypothesis.goals import normalise_goal_code
    indented = "\n".join("    " + l if l else l for l in GOOD.strip("\n").splitlines())
    code = normalise_goal_code(indented)
    assert code.startswith("def build_goal") and Sandbox().load_goal(code) is not None
