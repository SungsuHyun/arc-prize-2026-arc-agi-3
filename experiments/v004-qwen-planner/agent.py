"""v004: Qwen planner (L2) wired over the v003 observation encoder.

3계층이 모두 동작하는 첫 버전:
  L0 reflex   — 프레임 diff, no-op 필터 (v002)
  L1 explorer — 상태별 액션-효과, novelty 우선 (v002)
  L2 planner  — Qwen(ollama, OpenAI 스타일 chat) : 관측 텍스트 → JSON 계획

L2 호출 트리거: 계획 큐가 비어 있고 (새 레벨 진입 | STUCK_N 액션 동안 새
상태 없음 | PLAN_EVERY 액션 경과). 게임당 MAX_LLM_CALLS 예산.
계획 실행 중 연속 3회 무변화면 큐 폐기(재계획 유도).

인코더 개선 (v003 대비): 같은 색·근접 위치의 크기 변화를 "resized"로 통합,
동일 변화 반복은 ×N으로 압축.

env: QWEN_ENDPOINT(기본 http://localhost:11434), QWEN_MODEL(기본 qwen3.5:9b).
엔드포인트가 없으면 L2가 자동 비활성화되고 L1로 폴백 — Kaggle에서는 이
클래스의 _complete()만 in-process vLLM으로 교체하면 됨 (v005).
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from collections import Counter, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

SIMPLE_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]


def frame_key(frame: FrameData) -> int:
    return hash(tuple(tuple(map(tuple, g)) for g in frame.frame))


# ─────────────────────────── observation encoding ───────────────────────────

def extract_objects(grid: list[list[int]], max_objects: int = 24) -> list[dict]:
    h, w = len(grid), len(grid[0])
    background = Counter(c for row in grid for c in row).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    objects = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy][sx] or grid[sy][sx] == background:
                continue
            color = grid[sy][sx]
            stack, cells = [(sy, sx)], []
            seen[sy][sx] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            objects.append({
                "color": color, "size": len(cells),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "center": (sum(xs) // len(xs), sum(ys) // len(ys)),
            })
    objects.sort(key=lambda o: -o["size"])
    return objects[:max_objects]


def diff_objects(prev: list[dict], cur: list[dict]) -> dict:
    prev_pool = list(prev)
    moved, matched_cur = [], set()
    for c in cur:
        best, best_d = None, None
        for i, p in enumerate(prev_pool):
            if p is None or p["color"] != c["color"] or p["size"] != c["size"]:
                continue
            d = abs(p["center"][0] - c["center"][0]) + abs(p["center"][1] - c["center"][1])
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None:
            p = prev_pool[best]
            prev_pool[best] = None
            matched_cur.add(id(c))
            if best_d:
                moved.append((p, c, c["center"][0] - p["center"][0],
                              c["center"][1] - p["center"][1]))
    appeared = [c for c in cur if id(c) not in matched_cur]
    disappeared = [p for p in prev_pool if p is not None]

    # v004: 같은 색 + 근접 중심의 (appeared, disappeared) 쌍 → resized
    resized = []
    for a in list(appeared):
        for d in list(disappeared):
            if a["color"] == d["color"] and \
               abs(a["center"][0] - d["center"][0]) + abs(a["center"][1] - d["center"][1]) <= 3:
                resized.append((d, a))
                appeared.remove(a)
                disappeared.remove(d)
                break
    return {"moved": moved, "appeared": appeared,
            "disappeared": disappeared, "resized": resized}


def cells_changed(prev_grid, cur_grid) -> int:
    return sum(1 for pr, cr in zip(prev_grid, cur_grid)
               for a, b in zip(pr, cr) if a != b)


def obj_tag(o: dict) -> str:
    return f"c{o['color']}s{o['size']}@{o['center'][0]},{o['center'][1]}"


class ObservationEncoder:
    def __init__(self, recent_k: int = 8):
        # entries: {"a0": first counter, "a1": last, "n": reps, "act": name, "sig": text}
        self.recent: deque[dict] = deque(maxlen=recent_k)
        self.prev_grid = None
        self.prev_objects: list[dict] = []

    def observe(self, action_name: Optional[str], counter: int,
                latest_frame: FrameData) -> None:
        if not latest_frame.frame:
            return
        grid = latest_frame.frame[-1]
        objects = extract_objects(grid)
        if action_name is not None and self.prev_grid is not None:
            n = cells_changed(self.prev_grid, grid)
            if n == 0:
                sig = "no change"
            else:
                d = diff_objects(self.prev_objects, objects)
                bits = [f"{n} cells"]
                bits += [f"moved {obj_tag(c)} ({dx:+d},{dy:+d})"
                         for _, c, dx, dy in d["moved"][:3]]
                bits += [f"resized c{a['color']} s{b['size']}→{a['size']} "
                         f"@{a['center'][0]},{a['center'][1]}"
                         for b, a in d["resized"][:3]]
                bits += [f"+{obj_tag(o)}" for o in d["appeared"][:2]]
                bits += [f"-{obj_tag(o)}" for o in d["disappeared"][:2]]
                sig = "; ".join(bits)
            last = self.recent[-1] if self.recent else None
            if last and last["act"] == action_name and last["sig"] == sig:
                last["n"] += 1
                last["a1"] = counter
            else:
                self.recent.append({"a0": counter, "a1": counter, "n": 1,
                                    "act": action_name, "sig": sig})
        self.prev_grid = grid
        self.prev_objects = objects

    def encode(self, latest_frame: FrameData, tally: dict[str, list[int]]) -> str:
        lines = [
            f"[PROGRESS] level {latest_frame.levels_completed}/{latest_frame.win_levels}"
            f" state={latest_frame.state.name}",
            f"[AVAILABLE] {','.join(map(str, latest_frame.available_actions or []))}",
            "[OBJECTS] color/size/bbox/center (largest first, background omitted)",
        ]
        for o in self.prev_objects:
            x0, y0, x1, y1 = o["bbox"]
            lines.append(f" c{o['color']} s{o['size']} ({x0},{y0})-({x1},{y1})"
                         f" ctr({o['center'][0]},{o['center'][1]})")
        lines.append("[RECENT]")
        for e in self.recent:
            span = f"a{e['a0']}" + (f"-a{e['a1']} ×{e['n']}" if e["n"] > 1 else "")
            lines.append(f" {span} {e['act']}: {e['sig']}")
        stats = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(tally.items()))
        lines.append(f"[ACTION_STATS changed/tried] {stats}")
        return "\n".join(lines)


# ─────────────────────────────── L2 planner ─────────────────────────────────

SYSTEM_PROMPT = """You control an agent in an unknown 64x64 grid puzzle game (ARC-AGI-3).
Cells hold colors 0-15. You must discover the rules by acting, then complete levels.
Score favors completing levels in FEW actions.
Observation format: [OBJECTS] cN=color sN=size, [RECENT] effects of past actions
(moved/resized/+appeared/-disappeared), [ACTION_STATS] how often each action changed anything.
Actions: 1-5 and 7 are abstract buttons (meaning differs per game; check [RECENT]).
Action 6 is a click and needs x,y (0-63) — click object centers, not empty background.
Reply ONLY with JSON:
{"hypothesis": "<one sentence: what the rules/goal seem to be>",
 "plan": [{"action": <1-7>, "x": <0-63 if action 6>, "y": <0-63 if action 6>}, ...],
 "confidence": <0.0-1.0>}
Plan 3-10 actions that test your hypothesis or make progress."""


class QwenPlanner:
    """ollama chat 호출. 실패 누적 시 자동 비활성화(L1 폴백)."""

    def __init__(self, model: str, endpoint: Optional[str] = None,
                 timeout: float = 60.0, max_fail: int = 3):
        self.endpoint = (endpoint or os.getenv("QWEN_ENDPOINT")
                         or "http://localhost:11434").rstrip("/")
        self.model = os.getenv("QWEN_MODEL", model)
        self.timeout = timeout
        self.fails = 0
        self.max_fail = max_fail
        self.calls = 0
        self.errors = 0
        self.total_s = 0.0
        self.last_hypothesis = ""

    @property
    def enabled(self) -> bool:
        return self.fails < self.max_fail

    def _complete(self, obs_text: str) -> str:
        """Kaggle(v005)에서는 이 메서드만 in-process vLLM으로 교체."""
        body = json.dumps({
            "model": self.model, "stream": False, "think": False,
            "format": "json",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": obs_text}],
            "options": {"num_predict": 500, "temperature": 0.2, "seed": 1337},
        }).encode()
        req = urllib.request.Request(f"{self.endpoint}/api/chat", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)["message"]["content"]

    def plan(self, obs_text: str) -> Optional[list[dict]]:
        if not self.enabled:
            return None
        t0 = time.time()
        try:
            content = self._complete(obs_text)
            data = json.loads(content)
            steps = []
            for step in (data.get("plan") or [])[:10]:
                a = int(step.get("action", 0))
                if not 1 <= a <= 7:
                    continue
                item = {"action": a}
                if a == 6:
                    item["x"] = max(0, min(63, int(step.get("x", 32))))
                    item["y"] = max(0, min(63, int(step.get("y", 32))))
                steps.append(item)
            self.calls += 1
            self.total_s += time.time() - t0
            self.fails = 0
            self.last_hypothesis = str(data.get("hypothesis", ""))[:200]
            return steps or None
        except Exception:
            self.errors += 1
            self.fails += 1
            self.total_s += time.time() - t0
            return None


# ─────────────────────────────── the agent ──────────────────────────────────

class MyAgent(Agent):
    MAX_ACTIONS = 400
    PLAN_EVERY = 40        # 액션 경과 트리거
    STUCK_N = 15           # 새 상태 없음 트리거
    MAX_LLM_CALLS = 12     # 게임당 L2 예산
    MODEL = "qwen3.5:9b"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(1337)
        self.effects: dict[int, dict[str, bool]] = {}
        self.tally: dict[str, list[int]] = {}
        self.prev_key: Optional[int] = None
        self.prev_action: Optional[str] = None
        self.prev_level = 0
        self.encoder = ObservationEncoder()
        self.planner = QwenPlanner(self.MODEL)
        self.plan_queue: list[dict] = []
        self.plan_noop_streak = 0
        self.last_plan_at = 0
        self.stuck = 0
        self.obs_samples: list[tuple[int, str]] = []
        self.obs_chars: list[int] = []
        self.planned_executed = 0

    @property
    def metrics(self) -> dict:
        return {
            "unique_states": len(self.effects),
            "avg_obs_chars": (sum(self.obs_chars) // len(self.obs_chars)
                              if self.obs_chars else None),
            "llm_model": self.planner.model,
            "llm_calls": self.planner.calls,
            "llm_errors": self.planner.errors,
            "llm_time_s": round(self.planner.total_s, 1),
            "planned_actions": self.planned_executed,
        }

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _record_effect(self, latest_frame: FrameData) -> None:
        if self.prev_key is None or self.prev_action is None:
            return
        key = frame_key(latest_frame)
        leveled = latest_frame.levels_completed > self.prev_level
        changed = (key != self.prev_key) or leveled
        self.effects.setdefault(self.prev_key, {})[self.prev_action] = changed
        t = self.tally.setdefault(self.prev_action, [0, 0])
        t[0] += int(changed)
        t[1] += 1
        # 계획 실행 중 무변화 연속 3회 → 재계획
        if self.plan_queue or self.plan_noop_streak:
            self.plan_noop_streak = 0 if changed else self.plan_noop_streak + 1
            if self.plan_noop_streak >= 3:
                self.plan_queue.clear()
                self.plan_noop_streak = 0
        # stuck 판정: 새 unique state가 늘었는가
        if key not in self._seen_states:
            self._seen_states.add(key)
            self.stuck = 0
        else:
            self.stuck += 1
        if leveled:
            self.stuck = 0
            self.plan_queue.clear()   # 새 레벨 → 낡은 계획 폐기
        self.prev_level = latest_frame.levels_completed

    _seen_states: set

    def _should_plan(self, latest_frame: FrameData) -> bool:
        if self.plan_queue or not self.planner.enabled:
            return False
        if self.planner.calls >= self.MAX_LLM_CALLS:
            return False
        since = self.action_counter - self.last_plan_at
        if since < 8:                          # 최소 간격
            return False
        new_level = latest_frame.levels_completed > 0 and self.stuck == 0 and since >= 8
        return (self.stuck >= self.STUCK_N or since >= self.PLAN_EVERY or new_level)

    def choose_action(self, frames: list[FrameData],
                      latest_frame: FrameData) -> GameAction:
        if not hasattr(self, "_seen_states"):
            self._seen_states = set()
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_key = self.prev_action = None
            self.encoder = ObservationEncoder()
            self.plan_queue.clear()
            return GameAction.RESET

        self._record_effect(latest_frame)
        self.encoder.observe(self.prev_action, self.action_counter, latest_frame)

        obs = self.encoder.encode(latest_frame, self.tally)
        self.obs_chars.append(len(obs))
        if self.action_counter % 25 == 0 and len(self.obs_samples) < 12:
            self.obs_samples.append((self.action_counter, obs))

        key = frame_key(latest_frame)
        candidates = self._candidate_actions(latest_frame)

        if self._should_plan(latest_frame):
            planned = self.planner.plan(obs)
            self.last_plan_at = self.action_counter
            if planned:
                self.plan_queue = planned

        action = None
        if self.plan_queue:
            step = self.plan_queue.pop(0)
            ga = GameAction.from_id(step["action"])
            if ga.value in (latest_frame.available_actions or [ga.value]):
                action = ga
                if action.is_complex():
                    action.set_data({"x": step.get("x", 32), "y": step.get("y", 32)})
                self.planned_executed += 1
        if action is None:
            # L1 fallback (v002/v003과 동일)
            tried = self.effects.get(key, {})
            untried = [a for a in candidates if a.name not in tried]
            if untried:
                action = random.choice(untried)
            else:
                effective = [a for a in candidates if tried.get(a.name, True)]
                pool = effective or candidates
                weights = [(self.tally.get(a.name, [0, 0])[0] + 1)
                           / (self.tally.get(a.name, [0, 0])[1] + 2) for a in pool]
                action = random.choices(pool, weights=weights, k=1)[0]
            if action.is_complex():
                targets = [o["center"] for o in self.encoder.prev_objects]
                x, y = random.choice(targets) if targets \
                    else (random.randint(0, 63), random.randint(0, 63))
                action.set_data({"x": x, "y": y})

        action.reasoning = (self.planner.last_hypothesis or "v004 explorer")[:120]
        self.prev_key = key
        self.prev_action = action.name
        return action

    def _candidate_actions(self, latest_frame: FrameData) -> list[GameAction]:
        allowed = set(latest_frame.available_actions or range(1, 8))
        acts = [a for a in SIMPLE_ACTIONS if a.value in allowed]
        if 6 in allowed:
            acts.append(GameAction.ACTION6)
        return acts or SIMPLE_ACTIONS
