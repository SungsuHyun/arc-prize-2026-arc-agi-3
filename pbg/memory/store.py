"""memory/store.py — three-layer memory (spec §11): episodic (raw logs), semantic (per-game verified knowledge),
priors (cross-game mechanisms / goal templates + usage stats). Everything stored is machine-readable and re-verifiable.

memory_root/
  episodic/{game_id}/raw.jsonl, transitions.jsonl, frames/{step_idx}.npy
  semantic/{game_id}/action_semantics.json, world_model.py, world_model.meta.json, goal.py, hypotheses/, level_notes.json
  priors/ -> the package library (pbg/memory/priors) + usage_stats.json + roles.json + promoted/ (auto-added)"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..core.types import Transition

PRIORS_PKG = Path(__file__).resolve().parent / "priors"
ROLES = ["agent", "wall", "target", "key", "door", "pushable", "collectible", "hazard", "indicator", "decoration"]
MAX_TRANSITIONS = 50_000


@dataclass
class GameMemory:
    game_id: str
    transitions: list[Transition] = field(default_factory=list)
    action_semantics: dict = field(default_factory=dict)
    world_model_code: Optional[str] = None
    world_model_meta: dict = field(default_factory=dict)
    goal_code: Optional[str] = None
    goal_meta: dict = field(default_factory=dict)
    hypotheses: list[dict] = field(default_factory=list)     # unpromoted: [{code, score, coverage, name}]
    level_notes: dict = field(default_factory=dict)
    aux: dict = field(default_factory=dict)

    @property
    def world_model(self) -> Optional[str]:
        return self.world_model_code


class Memory:
    def __init__(self, root: Path):
        self.root = Path(root)
        for sub in ("episodic", "semantic", "priors"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._games: dict[str, GameMemory] = {}
        self._step_counter: dict[str, int] = {}

    # ── paths ──
    def episodic_dir(self, game_id: str) -> Path:
        p = self.root / "episodic" / game_id; p.mkdir(parents=True, exist_ok=True); (p / "frames").mkdir(exist_ok=True)
        return p

    def semantic_dir(self, game_id: str) -> Path:
        p = self.root / "semantic" / game_id; p.mkdir(parents=True, exist_ok=True); (p / "hypotheses").mkdir(exist_ok=True)
        return p

    # ── load / save ──
    def load(self, game_id: str) -> GameMemory:
        if game_id in self._games:
            return self._games[game_id]
        gm = GameMemory(game_id)
        sd = self.semantic_dir(game_id)
        if (sd / "action_semantics.json").exists():
            gm.action_semantics = json.loads((sd / "action_semantics.json").read_text())
        if (sd / "world_model.py").exists():
            gm.world_model_code = (sd / "world_model.py").read_text()
            if (sd / "world_model.meta.json").exists():
                gm.world_model_meta = json.loads((sd / "world_model.meta.json").read_text())
        if (sd / "goal.py").exists():
            gm.goal_code = (sd / "goal.py").read_text()
            if (sd / "goal.meta.json").exists():
                gm.goal_meta = json.loads((sd / "goal.meta.json").read_text())
        for f in sorted((sd / "hypotheses").glob("*.json")):
            try:
                gm.hypotheses.append(json.loads(f.read_text()))
            except Exception:
                pass
        if (sd / "level_notes.json").exists():
            gm.level_notes = json.loads((sd / "level_notes.json").read_text())
        self._games[game_id] = gm
        return gm

    def append(self, t: Transition, game_id: Optional[str] = None) -> str:
        gid = game_id or t.id.split(":")[0]
        gm = self.load(gid)
        if not t.id:
            n = self._step_counter.get(gid, len(gm.transitions)); t.id = f"{gid}:{t.level}:{t.step_idx or n}"
        gm.transitions.append(t)
        ed = self.episodic_dir(gid)
        with open(ed / "transitions.jsonl", "a") as f:
            f.write(json.dumps(t.to_json()) + "\n")
        if t.after_frame is not None:
            np.save(ed / "frames" / f"{t.step_idx}.npy", t.after_frame.grid)
        if len(gm.transitions) > MAX_TRANSITIONS:
            # keep transitions, drop the oldest frame files only (spec §11)
            for f in sorted((ed / "frames").glob("*.npy"), key=lambda p: int(p.stem))[: len(gm.transitions) - MAX_TRANSITIONS]:
                f.unlink(missing_ok=True)
        return t.id

    def extend(self, ts: list[Transition], game_id: Optional[str] = None) -> None:
        for t in ts:
            self.append(t, game_id)

    def save_semantics(self, game_id: str, semantics: dict) -> None:
        gm = self.load(game_id); gm.action_semantics = dict(semantics)
        (self.semantic_dir(game_id) / "action_semantics.json").write_text(json.dumps(gm.action_semantics, indent=1, default=str))

    def promote_model(self, game_id: str, code: str, meta: dict) -> None:
        sd = self.semantic_dir(game_id); gm = self.load(game_id)
        if gm.world_model_code and gm.world_model_code != code:
            n = len(list(sd.glob("world_model.v*.py"))) + 1
            (sd / f"world_model.v{n}.py").write_text(gm.world_model_code)
        gm.world_model_code = code; gm.world_model_meta = {**meta, "promoted_at": time.time()}
        (sd / "world_model.py").write_text(code)
        (sd / "world_model.meta.json").write_text(json.dumps(gm.world_model_meta, indent=1, default=str))

    def promote_goal(self, game_id: str, code: str, meta: dict) -> None:
        sd = self.semantic_dir(game_id); gm = self.load(game_id)
        gm.goal_code = code; gm.goal_meta = {**meta, "promoted_at": time.time()}
        (sd / "goal.py").write_text(code); (sd / "goal.meta.json").write_text(json.dumps(gm.goal_meta, indent=1, default=str))

    def save_hypotheses(self, game_id: str, hyps: list[dict]) -> None:
        sd = self.semantic_dir(game_id); gm = self.load(game_id)
        for f in (sd / "hypotheses").glob("*.json"):
            f.unlink()
        gm.hypotheses = []
        for i, h in enumerate(hyps[:10]):
            (sd / "hypotheses" / f"h{i:02d}.json").write_text(json.dumps(h, indent=1, default=str)); gm.hypotheses.append(h)

    def record_level_note(self, game_id: str, level: int, note: dict) -> None:
        gm = self.load(game_id); gm.level_notes[str(level)] = note
        (self.semantic_dir(game_id) / "level_notes.json").write_text(json.dumps(gm.level_notes, indent=1, default=str))

    def save(self) -> None:
        for gid, gm in self._games.items():
            if gm.action_semantics:
                (self.semantic_dir(gid) / "action_semantics.json").write_text(json.dumps(gm.action_semantics, indent=1, default=str))
            if gm.aux:
                (self.semantic_dir(gid) / "aux.json").write_text(json.dumps(gm.aux, indent=1, default=str))

    # ── priors ──
    def priors(self, kind: Literal["mechanisms", "goals"]) -> dict:
        if kind == "mechanisms":
            from .priors.mechanisms import MECHANISMS, mechanism_signatures
            promoted = {p.stem: p.read_text() for p in (self.root / "priors" / "promoted").glob("*.py")} if (self.root / "priors" / "promoted").exists() else {}
            return {"mechanisms": MECHANISMS, "signatures": mechanism_signatures(), "promoted": promoted, "usage_stats": self.usage_stats()}
        from ..goal.templates import TEMPLATES
        return {"templates": TEMPLATES, "usage_stats": self.usage_stats()}

    def usage_stats(self) -> dict:
        p = self.root / "priors" / "usage_stats.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def record_usage(self, name: str, *, used: bool = True, verified: bool = False, failed: bool = False) -> None:
        stats = self.usage_stats(); s = stats.setdefault(name, {"games_used": 0, "games_verified": 0, "games_failed": 0})
        if used: s["games_used"] += 1
        if verified: s["games_verified"] += 1
        if failed: s["games_failed"] += 1
        (self.root / "priors" / "usage_stats.json").write_text(json.dumps(stats, indent=1))

    def promote_prior(self, code: str, evidence_games: list[str], name: Optional[str] = None) -> bool:
        """Auto-add a rule to the prior library only when verified in >= 2 games (spec §11) and it passes static checks."""
        if len(set(evidence_games)) < 2:
            return False
        from ..wml.sandbox import static_check, StaticCheckError
        try:
            static_check(code if "def build_model" in code else code + "\n\ndef build_model():\n    return None\n")
        except StaticCheckError:
            return False
        d = self.root / "priors" / "promoted"; d.mkdir(parents=True, exist_ok=True)
        n = name or f"prior_{len(list(d.glob('*.py'))) + 1:03d}"
        (d / f"{n}.py").write_text(code)
        (d / f"{n}.meta.json").write_text(json.dumps({"evidence_games": sorted(set(evidence_games)), "promoted_at": time.time()}))
        return True

    # ── cross-game transfer helpers (spec §13) ──
    def semantic_games(self) -> list[str]:
        return sorted(p.name for p in (self.root / "semantic").iterdir() if (p / "world_model.py").exists())

    def structure_vector(self, game_id: str) -> Optional[list[float]]:
        p = self.semantic_dir(game_id) / "structure.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save_structure(self, game_id: str, vec: list[float]) -> None:
        (self.semantic_dir(game_id) / "structure.json").write_text(json.dumps(vec))

    def wipe_game(self, game_id: str) -> None:
        for sub in ("episodic", "semantic"):
            shutil.rmtree(self.root / sub / game_id, ignore_errors=True)
        self._games.pop(game_id, None)
