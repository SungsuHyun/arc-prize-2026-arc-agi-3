import json
from pathlib import Path

import numpy as np

from pbg.core.types import Action, Frame, grid_to_rows
from pbg.env.replay import ReplayEnv
from pbg.env.wrapper import stable_frames
from pbg.memory.store import Memory
from pbg.perception import Perception
from pbg.tests.unit.synthetic import board, frame


def test_stable_frames_picks_last_repeated():
    a, b, c = board(agent=(20, 20)), board(agent=(24, 20)), board(agent=(28, 20))
    after, inter = stable_frames([a, b, c, c])
    assert np.array_equal(after, c) and len(inter) == 3
    after, inter = stable_frames([a, a, b])
    assert np.array_equal(after, b) and len(inter) == 2


def test_replay_env_serves_log_and_rejects_other_actions(tmp_path: Path):
    recs = [{"kind": "reset", "action": {"type": "RESET"}, "frames": [grid_to_rows(board())], "state": "NOT_FINISHED", "level": 1, "levels_total": 3, "available_actions": [1, 2, 3, 4]},
            {"kind": "step", "action": {"type": "BUTTON", "id": 2}, "frames": [grid_to_rows(board(agent=(34, 30)))], "state": "NOT_FINISHED", "level": 1, "levels_total": 3, "available_actions": [1, 2, 3, 4], "status_change": None}]
    p = tmp_path / "raw.jsonl"; p.write_text("\n".join(json.dumps(r) for r in recs))
    env = ReplayEnv(p, "g")
    f = env.reset(); assert isinstance(f, Frame)
    bad = env.step(Action.button(1)); assert bad.error
    ok = env.step(Action.button(2)); assert not ok.error and ok.after.hash != f.hash
    assert env.exhausted() and env.step(Action.button(2)).error


def test_memory_roundtrip(tmp_path: Path):
    m = Memory(tmp_path)
    p = Perception()
    s0 = p.parse(frame(0), None); s1 = p.parse(frame(1, agent=(34, 30)), s0)
    t = p.transition(s0, Action.button(2), s1, after_frame=frame(1, agent=(34, 30)), tid="g:1:1", level=1, step_idx=1)
    tid = m.append(t, "g")
    assert tid == "g:1:1" and (tmp_path / "episodic" / "g" / "transitions.jsonl").exists()
    m.promote_model("g", "def build_model():\n    return None\n", {"score": 1.0})
    m.promote_model("g", "def build_model():\n    return 1\n", {"score": 1.0})
    assert (tmp_path / "semantic" / "g" / "world_model.v1.py").exists()
    m.record_level_note("g", 1, {"novelty": 0})
    m.save_semantics("g", {"ACTION1": {"class": "MOVE"}})
    m2 = Memory(tmp_path); gm = m2.load("g")
    assert gm.world_model_code.strip().endswith("return 1") and gm.level_notes["1"]["novelty"] == 0 and gm.action_semantics["ACTION1"]["class"] == "MOVE"
    assert not m2.promote_prior("def f(): pass", ["g"])          # needs two games
    assert m2.promote_prior("def build_model():\n    return None\n", ["g", "h"])
    m2.record_usage("reach", used=True, verified=True)
    assert m2.usage_stats()["reach"]["games_verified"] == 1


def test_lint_no_game_id_passes_on_package():
    from pbg.tools.lint_no_game_id import main
    assert main([]) == 0
