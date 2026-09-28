import json
from pathlib import Path

import numpy as np

from pbg.core.budget import Budget
from pbg.core.events import EventLog
from pbg.core.types import Action, Frame, Object, Scene, mask_to_rle, rle_to_mask, shape_signature, shape_signature_d4


def test_action_roundtrip():
    for a in (Action.button(3), Action.click(5, 7), Action.reset()):
        assert Action.from_json(a.to_json()) == a
    assert Action.button(6).key == "ACTION6" and Action.click(1, 2).key == "CLICK" and Action.click(1, 2).label() == "CLICK(1,2)"


def test_frame_hash_and_json():
    g = np.zeros((4, 4), dtype=np.int8); g[1, 2] = 7
    f = Frame(g, 1, 0)
    f2 = Frame.from_json(f.to_json())
    assert f2.hash == f.hash and np.array_equal(f2.grid, g)


def test_mask_rle_and_signatures():
    m = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
    assert np.array_equal(rle_to_mask(mask_to_rle(m), m.shape), m)
    shifted = np.zeros((4, 5), dtype=bool); shifted[1:3, 1:4] = m
    assert shape_signature(m) == shape_signature(shifted)          # translation invariant
    assert shape_signature_d4(m) == shape_signature_d4(np.rot90(m))  # D4 invariant
    assert shape_signature(m) != shape_signature(np.rot90(m))


def test_object_scene_json_roundtrip():
    m = np.ones((2, 3), dtype=bool)
    o = Object(1, 2, (2,), (5, 6, 7, 9), m, 6, shape_signature(m), "R0")
    s = Scene("h", (10, 10), [], [o], {"k": 1})
    s2 = Scene.from_json(json.loads(json.dumps(s.to_json())))
    assert s2.objects[0].identity() == o.identity() and s2.key() == s.key()
    assert o.moved(1, 1).bbox == (6, 7, 8, 10) and o.overlaps(o.moved(0, 2)) and not o.overlaps(o.moved(0, 3))


def test_budget_caps():
    b = Budget(1000, {"total": 1000, "initial_probe": 0.12, "experiment": 0.25, "level_probe": 0.05, "reprobe": 0.08, "low_budget_fraction": 0.2, "llm_calls_max": 60})
    assert b.cap("initial_probe") == 120
    b.charge("initial_probe", 100)
    assert b.cap("initial_probe") == 20 and b.remaining() == 900 and not b.low()
    b.charge("level_probe", 50, level=1)
    assert b.cap("level_probe", 1) == 0 and b.cap("level_probe", 2) == 50
    b.charge("plan", 700)
    assert b.low() and b.allows("experiment") and b.cap("experiment") == 150


def test_event_log_requires_reason(tmp_path: Path):
    ev = EventLog(tmp_path / "e.jsonl")
    ev.emit("A", "B", "because", budget_used=1)
    try:
        ev.emit("B", "C", "")
        assert False, "empty reason must be rejected"
    except ValueError:
        pass
    assert len(EventLog.read(tmp_path / "e.jsonl")) == 1


def test_budget_per_level_caps():
    b = Budget(2000, {"total": 2000, "reprobe": 0.08, "experiment": 0.25, "per_level": {"reprobe": 32, "experiment": 12}, "low_budget_fraction": 0.2, "llm_calls_max": 60})
    assert b.cap("reprobe", 1) == 32 and b.cap("reprobe") == 160
    b.charge("reprobe", 32, level=1)
    assert b.cap("reprobe", 1) == 0 and b.cap("reprobe", 2) == 32
    b.reset_level(1, "reprobe")
    assert b.cap("reprobe", 1) == 32
    assert b.allows("experiment", level=1) and b.cap("experiment", 1) == 12
