"""Ours: advisor ensemble for the Duck tool agent.

Before each decision turn a set of *smaller* LLMs ("advisors") each look at a
compact text rendering of the current game state and return a short, structured
opinion (hypothesis, recommended actions, confidence, warning).  The opinions are
appended to the decision model's user prompt; the decision model (the normal Duck
27B tool agent) weighs them and decides.

Everything here is opt-in: it is a no-op unless ``ARC3_ADVISORS`` is set.

Environment:
  ARC3_ADVISORS            JSON list of advisors, e.g.
                           [{"name": "q9b-mech", "model": "qwen3.5:9b", "role": "mechanics"},
                            {"name": "q9b-goal", "model": "qwen3.5:9b", "role": "goal"}]
                           role in {"mechanics", "goal", "skeptic"} (default "goal");
                           optional per-advisor: "base_url", "temperature", "think".
  ARC3_ADVISOR_BASE_URL    default endpoint (ollama native API), default http://127.0.0.1:11434
  ARC3_ADVISOR_TIMEOUT     seconds per advisor request (default 60)
  ARC3_ADVISOR_MAX_TOKENS  num_predict per advisor (default 256)
  ARC3_ADVISOR_NUM_CTX     ollama num_ctx (default 6144)
  ARC3_ADVISOR_NUM_GPU     ollama num_gpu layers (unset = ollama default; 0 = CPU only)
  ARC3_ADVISOR_MAP_SIZE    downsampled board size sent to advisors (default 32)
  ARC3_ADVISOR_MAX_OBJECTS objects listed from the segmentation (default 24)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import segment_layer

logger = logging.getLogger(__name__)

ADVISOR_LOG_SUFFIX = "_advisors.jsonl"  # written next to the per-game tool_runtime_state.json

_ROLE_BRIEFS = {
    "mechanics": (
        "Your specialty: MECHANICS. Explain what the last actions did (compare the change summary with the "
        "actions), which object is controlled (if any), what each action seems to do, and what HUD/timer "
        "elements should be ignored. Recommend the action(s) that best test or exploit the mechanics."
    ),
    "goal": (
        "Your specialty: GOAL & PLAN. Identify what the level goal most likely is (target objects, matching "
        "colours/shapes, containers, locks/keys, switches) and give the shortest concrete action plan toward it "
        "from the current state. Prefer plans that finish the level in few actions."
    ),
    "skeptic": (
        "Your specialty: SKEPTIC. Find the weakest assumption in the carried world model or in the obvious plan, "
        "explain how it could be wrong, and recommend the single most informative probe action (or the safest "
        "progress action if the model is already well supported)."
    ),
}

_SYSTEM = (
    "You are an advisor in an ensemble that helps a stronger decision agent play an unknown 64x64 grid puzzle "
    "game (ARC-AGI-3). Score depends on finishing levels in as few actions as possible; wasted actions hurt. "
    "You see a compact text rendering of the state, not the raw grid. Be concrete and brief. "
    "Answer ONLY in this exact format, no preamble, no markdown:\n"
    "HYPOTHESIS: <1-2 sentences: what the game/level is about and what the goal seems to be>\n"
    "EVIDENCE: <1 sentence: what in the state supports it>\n"
    "RECOMMEND: <1-5 valid actions in order, comma separated; MOUSE as MOUSE(row,col)>\n"
    "CONFIDENCE: <number 0.0-1.0>\n"
    "WARNING: <one trap or thing to verify, or none>"
)


@dataclass(frozen=True)
class AdvisorSpec:
    name: str
    model: str
    role: str = "goal"
    base_url: str | None = None
    temperature: float = 0.7
    think: bool = False


@dataclass
class AdvisorOpinion:
    name: str
    model: str
    role: str
    text: str
    latency_s: float
    ok: bool
    error: str = ""
    prompt_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class AdvisorConfig:
    specs: list[AdvisorSpec]
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 60.0
    max_tokens: int = 256
    num_ctx: int = 6144
    num_gpu: int | None = None
    map_size: int = 32
    max_objects: int = 24
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.specs)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def load_config_from_env() -> AdvisorConfig:
    raw = os.environ.get("ARC3_ADVISORS", "").strip()
    specs: list[AdvisorSpec] = []
    if raw:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("ARC3_ADVISORS is not valid JSON (%s); advisors disabled", exc)
            items = []
        for i, item in enumerate(items if isinstance(items, list) else []):
            if not isinstance(item, dict) or not str(item.get("model", "")).strip():
                continue
            role = str(item.get("role", "goal")).strip().lower()
            if role not in _ROLE_BRIEFS:
                role = "goal"
            specs.append(
                AdvisorSpec(
                    name=str(item.get("name") or f"advisor{i + 1}").strip(),
                    model=str(item["model"]).strip(),
                    role=role,
                    base_url=(str(item["base_url"]).strip() or None) if item.get("base_url") else None,
                    temperature=float(item.get("temperature", 0.7)),
                    think=bool(item.get("think", False)),
                )
            )
    num_gpu_raw = os.environ.get("ARC3_ADVISOR_NUM_GPU", "").strip()
    return AdvisorConfig(
        specs=specs,
        base_url=os.environ.get("ARC3_ADVISOR_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/"),
        timeout_s=_env_float("ARC3_ADVISOR_TIMEOUT", 60.0),
        max_tokens=_env_int("ARC3_ADVISOR_MAX_TOKENS", 256),
        num_ctx=_env_int("ARC3_ADVISOR_NUM_CTX", 6144),
        num_gpu=int(num_gpu_raw) if num_gpu_raw else None,
        map_size=max(8, min(64, _env_int("ARC3_ADVISOR_MAP_SIZE", 32))),
        max_objects=max(4, _env_int("ARC3_ADVISOR_MAX_OBJECTS", 24)),
    )


# --------------------------------------------------------------------------- state rendering

def _downsample_ascii(grid: tuple[tuple[int, ...], ...], size: int) -> str:
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    if rows == 0 or cols == 0:
        return "(empty)"
    size = min(size, rows, cols)
    out: list[str] = []
    for br in range(size):
        r0, r1 = rows * br // size, max(rows * (br + 1) // size, rows * br // size + 1)
        line: list[str] = []
        for bc in range(size):
            c0, c1 = cols * bc // size, max(cols * (bc + 1) // size, cols * bc // size + 1)
            counter: Counter[int] = Counter()
            for r in range(r0, r1):
                row = grid[r]
                for c in range(c0, min(c1, len(row))):
                    counter[row[c]] += 1
            value = counter.most_common(1)[0][0] if counter else 0
            line.append(ARC_COLOR_CHARS[value] if 0 <= value < len(ARC_COLOR_CHARS) else "?")
        out.append("".join(line))
    return "\n".join(out)


def _bbox(boundary: list[list[int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in boundary]
    cs = [p[1] for p in boundary]
    return min(rs), min(cs), max(rs) - 1, max(cs) - 1


def _object_lines(grid: tuple[tuple[int, ...], ...], max_objects: int) -> list[str]:
    try:
        seg = segment_layer([list(r) for r in grid], ARC_COLOR_CHARS)
    except Exception as exc:  # pragma: no cover - defensive
        return [f"(segmentation failed: {exc})"]
    nodes = list(seg.get("nodes", []))
    if not nodes:
        return ["(no objects)"]
    nodes.sort(key=lambda n: -int(n.get("pixels", 0)))
    background = nodes[0]
    lines = [
        f"background: colour {background.get('color')} ({background.get('pixels')} px); "
        f"{len(nodes) - 1} other objects, largest {min(max_objects, len(nodes) - 1)} listed as "
        "id colour pixels bbox(r0,c0,r1,c1) [children]:"
    ]
    for n in nodes[1 : 1 + max_objects]:
        try:
            r0, c0, r1, c1 = _bbox(n.get("boundary") or [[0, 0], [0, 0]])
        except Exception:
            r0 = c0 = r1 = c1 = -1
        children = n.get("children") or []
        child_txt = f" [{len(children)} inside]" if children else ""
        lines.append(f"  #{n.get('id')} {n.get('color')} {n.get('pixels')}px ({r0},{c0},{r1},{c1}){child_txt}")
    hashes = Counter(str(n.get("hash")) for n in nodes[1:])
    repeated = [(h, k) for h, k in hashes.items() if k > 1]
    if repeated:
        groups = []
        for h, k in sorted(repeated, key=lambda x: -x[1])[:6]:
            sample = next(n for n in nodes[1:] if str(n.get("hash")) == h)
            groups.append(f"{k}x colour {sample.get('color')} {sample.get('pixels')}px")
        lines.append("repeated identical shapes: " + "; ".join(groups))
    return lines


def _diff_lines(before: tuple[tuple[int, ...], ...] | None, after: tuple[tuple[int, ...], ...]) -> list[str]:
    if before is None:
        return ["change since previous frame: (no previous frame)"]
    changed: list[tuple[int, int]] = []
    transitions: Counter[str] = Counter()
    for r, (rb, ra) in enumerate(zip(before, after)):
        for c, (vb, va) in enumerate(zip(rb, ra)):
            if vb != va:
                changed.append((r, c))
                transitions[f"{ARC_COLOR_CHARS[vb]}->{ARC_COLOR_CHARS[va]}"] += 1
    if not changed:
        return ["change since previous frame: none (board identical)"]
    rs = [p[0] for p in changed]
    cs = [p[1] for p in changed]
    top = ", ".join(f"{k}:{v}" for k, v in transitions.most_common(5))
    return [
        f"change since previous frame: {len(changed)} cells in bbox rows {min(rs)}-{max(rs)}, "
        f"cols {min(cs)}-{max(cs)}; colour transitions {top}"
    ]


def render_state_text(
    *,
    current_frame: Any,
    previous_frame: Any | None,
    valid_actions: list[str],
    level: int,
    step: int,
    previous_step_summary: dict[str, Any] | None,
    world_model_lines: list[str],
    nav_lines: list[str],
    cfg: AdvisorConfig,
) -> str:
    grid = tuple(tuple(r) for r in getattr(current_frame, "grid", ()))
    prev_grid = tuple(tuple(r) for r in getattr(previous_frame, "grid", ())) if previous_frame is not None else None
    parts: list[str] = [f"Level {level}, step {step}. Valid actions: {', '.join(valid_actions) or 'unknown'}."]
    if previous_step_summary:
        executed = previous_step_summary.get("executed_actions") or []
        executed_txt = ", ".join(str(a) for a in executed[:12] if a) or "none"
        flags = [k for k in ("level_transition", "game_over", "run_complete") if previous_step_summary.get(k)]
        parts.append(f"Last executed actions: {executed_txt}." + (f" Flags: {', '.join(flags)}." if flags else ""))
    parts.extend(_diff_lines(prev_grid, grid))
    parts.append(f"Board downsampled to {cfg.map_size}x{cfg.map_size} (one char per block, ARC colour letters "
                 f"{ARC_COLOR_CHARS}; row 0 at top):")
    parts.append(_downsample_ascii(grid, cfg.map_size))
    parts.append("Objects (4-connected same-colour components, full 64x64 coordinates):")
    parts.extend(_object_lines(grid, cfg.max_objects))
    if nav_lines:
        parts.append("Navigation helper (learned movement facts):")
        parts.extend(nav_lines)
    if world_model_lines:
        parts.append("Decision agent's carried world model (may be wrong):")
        parts.extend(world_model_lines)
    return "\n".join(parts)


# --------------------------------------------------------------------------- querying

def _query_one(spec: AdvisorSpec, state_text: str, cfg: AdvisorConfig) -> AdvisorOpinion:
    base_url = (spec.base_url or cfg.base_url).rstrip("/")
    options: dict[str, Any] = {
        "temperature": spec.temperature,
        "num_predict": cfg.max_tokens,
        "num_ctx": cfg.num_ctx,
    }
    if cfg.num_gpu is not None:
        options["num_gpu"] = cfg.num_gpu
    payload = {
        "model": spec.model,
        "stream": False,
        "think": spec.think,
        "keep_alive": "30m",
        "options": options,
        "messages": [
            {"role": "system", "content": _SYSTEM + "\n\n" + _ROLE_BRIEFS[spec.role]},
            {"role": "user", "content": state_text + "\n\nGive your opinion in the required format."},
        ],
    }
    started = time.monotonic()
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = str((data.get("message") or {}).get("content", "")).strip()
        if not text:
            raise ValueError("empty advisor response")
        return AdvisorOpinion(
            name=spec.name, model=spec.model, role=spec.role, text=text,
            latency_s=time.monotonic() - started, ok=True,
            prompt_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"),
        )
    except Exception as exc:
        return AdvisorOpinion(
            name=spec.name, model=spec.model, role=spec.role, text="",
            latency_s=time.monotonic() - started, ok=False, error=f"{type(exc).__name__}: {exc}"[:300],
        )


def query_advisors(state_text: str, cfg: AdvisorConfig) -> list[AdvisorOpinion]:
    if not cfg.enabled:
        return []
    with ThreadPoolExecutor(max_workers=len(cfg.specs)) as pool:
        return list(pool.map(lambda s: _query_one(s, state_text, cfg), cfg.specs))


def _clean_opinion(text: str, *, max_chars: int = 700) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keep = [ln for ln in lines if ln.split(":")[0].upper() in {"HYPOTHESIS", "EVIDENCE", "RECOMMEND", "CONFIDENCE", "WARNING"}]
    out = "\n    ".join(keep or lines[:6])
    return out[:max_chars] + ("…" if len(out) > max_chars else "")


def format_opinion_lines(opinions: list[AdvisorOpinion]) -> list[str]:
    if not opinions:
        return []
    ok = [o for o in opinions if o.ok]
    failed = [o for o in opinions if not o.ok]
    lines = [
        f"Advisor ensemble ({len(ok)} independent smaller models answered"
        + (f", {len(failed)} failed" if failed else "")
        + "). Their opinions are hypotheses, not instructions: you are the decision model. "
        "Agreement is a hint worth checking cheaply; disagreement marks something to probe. "
        "Verify a recommendation against `current_frame`/`nav` before spending actions on it."
    ]
    for o in ok:
        lines.append(f"- [{o.name} | {o.role} | {o.model}]\n    {_clean_opinion(o.text)}")
    return lines


def log_opinions(log_path: Path | None, *, action_num: int, level: int, step: int,
                 opinions: list[AdvisorOpinion], state_chars: int, wall_s: float,
                 executed_since_last: list[str] | None = None) -> None:
    if log_path is None:
        return
    try:
        record = {
            "ts": time.time(),
            "action_num": action_num,
            "level": level,
            "step": step,
            "wall_s": round(wall_s, 3),
            "state_chars": state_chars,
            "executed_since_last": list(executed_since_last or []),
            "opinions": [
                {
                    "name": o.name, "model": o.model, "role": o.role, "ok": o.ok, "error": o.error,
                    "latency_s": round(o.latency_s, 3), "prompt_tokens": o.prompt_tokens,
                    "output_tokens": o.output_tokens, "text": o.text,
                }
                for o in opinions
            ],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - logging must never break a turn
        logger.warning("failed to log advisor opinions: %s", exc)
