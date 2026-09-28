"""Integration tests against the offline arc_agi engine (skipped when environment_files/ is absent)."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(not (ROOT / "environment_files").exists(), reason="environment_files not present")


@pytest.fixture(scope="module")
def arcade():
    os.environ.setdefault("ARC_API_KEY", "local-dev")
    from pbg.harness.online_runner import make_arcade
    return make_arcade()


def test_first_frames_parse_fast_on_every_game(arcade):
    from pbg.env.wrapper import ArcadeEnv
    from pbg.perception import Perception
    games = sorted(p.name for p in (ROOT / "environment_files").iterdir() if p.is_dir())
    slow = []
    for g in games:
        env = ArcadeEnv(arcade.make(g), g)
        f = env.reset()
        p = Perception(); s = p.parse(f, None)
        assert s.objects and s.regions
        if p.timings[-1] > 0.02:
            slow.append((g, p.timings[-1]))
    assert not slow, slow


def test_orchestrator_runs_without_llm(arcade, tmp_path):
    from pbg.env.wrapper import ArcadeEnv
    from pbg.memory.store import Memory
    from pbg.orchestrator.loop import Orchestrator
    env = ArcadeEnv(arcade.make("ls20"), "ls20")
    orch = Orchestrator(memory=Memory(tmp_path / "mem"), use_llm=False, events_dir=tmp_path, max_seconds=60)
    r = orch.play(env, "ls20", budget_total=60)
    assert r.actions <= 60 and r.events > 3
    events = (tmp_path / "ls20.events.jsonl").read_text().splitlines()
    assert events and all('"reason": ""' not in e for e in events)
