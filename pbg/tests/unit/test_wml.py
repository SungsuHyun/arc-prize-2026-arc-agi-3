import numpy as np

from pbg.core.contracts import Rule, RuleModel, scene_equal
from pbg.core.types import Action
from pbg.memory.priors.mechanisms import cancel_move_if_off_floor, cancel_move_if_overlap, move_role, DIRS4
from pbg.perception import Perception
from pbg.probe.semantics import classify_actions
from pbg.wml.evaluate import evaluate, promotable
from pbg.wml.experiment import most_informative_action
from pbg.wml.induce import induce_hypotheses
from pbg.wml.sandbox import Sandbox, StaticCheckError, static_check
from pbg.tests.unit.synthetic import frame


def _walk_log(n=24):
    """Agent walks around a wall; a scripted simulation with 4-px steps, blocked by walls (5) and the board edge."""
    p = Perception()
    pos = [30, 30]
    prev = p.parse(frame(0, agent=tuple(pos)), None)
    ts = []
    seq = [1, 1, 1, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 4, 2, 2, 2, 2, 1, 3, 3, 1, 1, 4] * 2
    for i, b in enumerate(seq[:n], 1):
        dr, dc = DIRS4[b]
        nr, nc = pos[0] + 4 * dr, pos[1] + 4 * dc
        blocked = not (8 <= nr and nr + 4 <= 56 and 8 <= nc and nc + 4 <= 56) or (16 <= nr + 3 and nr <= 23 and 16 <= nc + 3 and nc <= 23)
        if not blocked:
            pos = [nr, nc]
        cur = p.parse(frame(i, agent=tuple(pos)), prev)
        t = p.transition(prev, Action.button(b), cur); t.id = f"g:1:{i}"; ts.append(t); prev = t.after
    if p.merge_confirmed or p.merge_log():
        p.refit(ts)
    return ts


def test_rule_model_predicts_and_evaluate_scores():
    ts = _walk_log()
    sem = classify_actions(ts)
    cands = induce_hypotheses(ts, sem, [Action.button(i) for i in (1, 2, 3, 4)])
    assert cands, "induction must produce candidates for a movement game"
    best = max((evaluate(m, ts).score, name) for name, m in cands)
    assert best[0] >= 0.9, f"induced model should explain the walk, got {best}"


def test_promotion_condition():
    ts = _walk_log(32)
    sem = classify_actions(ts)
    name, m = max(induce_hypotheses(ts, sem, [Action.button(i) for i in (1, 2, 3, 4)]), key=lambda nm: evaluate(nm[1], ts).score)
    res = evaluate(m, ts)
    assert res.n == 32 and (promotable(res) == (res.score == 1.0 and all(t >= 3 for _, t in res.per_class.values())))


def test_information_gain_picks_disagreement():
    ts = _walk_log()
    scene = ts[-1].after
    sem = classify_actions(ts)
    cands = induce_hypotheses(ts, sem, [Action.button(i) for i in (1, 2, 3, 4)])
    from pbg.wml.evaluate import make_hypothesis
    H = [make_hypothesis(m, evaluate(m, ts), name=n) for n, m in cands]
    # add a deliberately wrong hypothesis (agent never moves) so predictions disagree
    H.append(make_hypothesis(RuleModel([Rule("still", lambda s, a: True, lambda s, a: s)], H[0].model._role_fn, default="noop"), evaluate(H[0].model, ts), name="still"))
    a = most_informative_action(H, scene, [Action.button(i) for i in (1, 2, 3, 4)], tau=0.1)
    assert a is not None and a.type == "BUTTON"


def test_static_checks_reject_bad_code():
    for bad in ("import os\ndef build_model():\n    return None", "def build_model():\n    return eval('1')",
                "def build_model():\n    if game == 'ls20':\n        pass", "x = 1"):
        try:
            static_check(bad); assert False, bad
        except StaticCheckError:
            pass


def test_sandbox_loads_and_validates_generated_code():
    code = '''
def role_fn(scene):
    roles = {}
    for o in scene.objects:
        if len(o.colors) > 1 or o.color in (9, 12):
            roles[o.id] = "agent"
        elif o.color == 5:
            roles[o.id] = "wall"
        else:
            roles[o.id] = "unknown"
    return roles

def build_model():
    rules = [Rule("move", lambda s, a: a.type == "BUTTON" and a.id in DIRS4, move_role("agent", DIRS4, 4)),
             Rule("wall", lambda s, a: a.type == "BUTTON", cancel_move_if_overlap("agent", "wall")),
             Rule("floor", lambda s, a: a.type == "BUTTON", cancel_move_if_off_floor("agent", (3,)))]
    return RuleModel(rules, role_fn, default="unknown")
'''
    ts = _walk_log(16)
    sb = Sandbox()
    log_json = [{"id": t.id, "level": t.level, "step_idx": t.step_idx, "action": t.action.to_json(), "before": t.before.to_json(), "after": t.after.to_json()} for t in ts]
    v = sb.validate(code, log_json)
    assert v.get("ok"), v
    assert v["score"] >= 0.9
    m = sb.load(code)
    assert m is not None and abs(evaluate(m, ts).score - v["score"]) < 1e-9


def test_sandbox_worker_enforces_cpu_limit():
    code = "def build_model():\n    while True:\n        pass\n"
    v = Sandbox(cpu_seconds=1).validate(code, [])
    assert not v.get("ok")
