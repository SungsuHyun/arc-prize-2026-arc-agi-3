"""v003: observation encoder — frames -> compact LLM-ready text.

v002(effect explorer)에 L2용 관측 인코더를 추가한 버전. 행동 정책은 v002와
동일(L0+L1)하고, 매 스텝 관측을 인코딩해 변화 로그를 축적한다. 목표는
64x64 원시 그리드(4천+ 토큰)를 수백 토큰의 구조화 텍스트로 압축하는 것.

인코딩 구성 (docs/005의 L2 입력 설계):
  [PROGRESS]      레벨/액션/상태
  [AVAILABLE]     사용 가능 액션
  [OBJECTS]       connected component 객체 목록 (색, 크기, bbox, 중심)
  [RECENT]        최근 액션별 변화 로그 (이동/생성/소멸/셀 변화 수)
  [ACTION_STATS]  액션별 효과 통계 (L1 tally)

표준 라이브러리만 사용 — Kaggle 노트북에 그대로 이식 가능.
"""
from __future__ import annotations

import random
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
    """Connected same-color components (4-conn), background excluded.

    Returns dicts: {color, size, bbox:(x0,y0,x1,y1), center:(cx,cy)},
    largest first, capped at max_objects.
    """
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
    """Object-level diff: moved / appeared / disappeared.

    Matching heuristic: same (color, size), nearest centers first. Coarse but
    cheap, and 'roughly what changed' is all the planner needs.
    """
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
    return {"moved": moved, "appeared": appeared, "disappeared": disappeared}


def cells_changed(prev_grid: list[list[int]], cur_grid: list[list[int]]) -> int:
    return sum(1 for pr, cr in zip(prev_grid, cur_grid)
               for a, b in zip(pr, cr) if a != b)


def obj_tag(o: dict) -> str:
    return f"c{o['color']}s{o['size']}@{o['center'][0]},{o['center'][1]}"


class ObservationEncoder:
    """Accumulates per-action change entries and renders the full observation."""

    def __init__(self, recent_k: int = 8):
        self.recent: deque[str] = deque(maxlen=recent_k)
        self.prev_grid: Optional[list[list[int]]] = None
        self.prev_objects: list[dict] = []

    def observe(self, action_name: Optional[str], counter: int,
                latest_frame: FrameData) -> None:
        """Call once per step BEFORE acting, with the action that produced
        this frame (None for the first frame)."""
        if not latest_frame.frame:
            return
        grid = latest_frame.frame[-1]
        objects = extract_objects(grid)
        if action_name is not None and self.prev_grid is not None:
            n = cells_changed(self.prev_grid, grid)
            if n == 0:
                entry = f"a{counter} {action_name}: no change"
            else:
                d = diff_objects(self.prev_objects, objects)
                bits = [f"{n} cells"]
                bits += [f"moved {obj_tag(c)} ({dx:+d},{dy:+d})"
                         for _, c, dx, dy in d["moved"][:3]]
                bits += [f"+{obj_tag(o)}" for o in d["appeared"][:3]]
                bits += [f"-{obj_tag(o)}" for o in d["disappeared"][:3]]
                entry = f"a{counter} {action_name}: " + "; ".join(bits)
            self.recent.append(entry)
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
        lines.extend(f" {e}" for e in self.recent)
        stats = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(tally.items()))
        lines.append(f"[ACTION_STATS changed/tried] {stats}")
        return "\n".join(lines)


# ─────────────────────────────── the agent ──────────────────────────────────

class QwenPlanner:
    """L2 stub — v004에서 Qwen 연결. 계약: plan(obs_text) -> Optional[list]."""

    def plan(self, obs_text: str) -> Optional[list]:
        return None


class MyAgent(Agent):
    MAX_ACTIONS = 400
    OBS_SAMPLE_EVERY = 25

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(1337)
        self.effects: dict[int, dict[str, bool]] = {}
        self.tally: dict[str, list[int]] = {}
        self.prev_key: Optional[int] = None
        self.prev_action: Optional[str] = None
        self.prev_level = 0
        self.encoder = ObservationEncoder()
        self.planner = QwenPlanner()
        self.plan_queue: list = []
        self.obs_samples: list[tuple[int, str]] = []   # (action_counter, text)
        self.obs_chars: list[int] = []

    # exposed to run_experiment.py (merged into the result JSON)
    @property
    def metrics(self) -> dict:
        return {
            "unique_states": len(self.effects),
            "avg_obs_chars": (sum(self.obs_chars) // len(self.obs_chars)
                              if self.obs_chars else None),
            "max_obs_chars": max(self.obs_chars, default=None),
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
        self.prev_level = latest_frame.levels_completed

    def _candidate_actions(self, latest_frame: FrameData) -> list[GameAction]:
        allowed = set(latest_frame.available_actions or range(1, 8))
        acts = [a for a in SIMPLE_ACTIONS if a.value in allowed]
        if 6 in allowed:
            acts.append(GameAction.ACTION6)
        return acts or SIMPLE_ACTIONS

    def choose_action(self, frames: list[FrameData],
                      latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_key = self.prev_action = None
            self.encoder = ObservationEncoder()   # fresh episode, fresh log
            return GameAction.RESET

        self._record_effect(latest_frame)
        self.encoder.observe(self.prev_action, self.action_counter, latest_frame)

        # sample encoded observations for offline eval (scripts/dump_observations.py)
        obs = self.encoder.encode(latest_frame, self.tally)
        self.obs_chars.append(len(obs))
        if self.action_counter % self.OBS_SAMPLE_EVERY == 0 \
                and len(self.obs_samples) < 12:
            self.obs_samples.append((self.action_counter, obs))

        key = frame_key(latest_frame)
        candidates = self._candidate_actions(latest_frame)

        if not self.plan_queue:
            planned = self.planner.plan(obs)
            if planned:
                self.plan_queue = list(planned)
        if self.plan_queue:
            action = self.plan_queue.pop(0)
        else:
            tried = self.effects.get(key, {})
            untried = [a for a in candidates if a.name not in tried]
            if untried:
                action = random.choice(untried)
            else:
                effective = [a for a in candidates if tried.get(a.name, True)]
                pool = effective or candidates
                weights = [
                    (self.tally.get(a.name, [0, 0])[0] + 1)
                    / (self.tally.get(a.name, [0, 0])[1] + 2)
                    for a in pool
                ]
                action = random.choices(pool, weights=weights, k=1)[0]

        if action.is_complex():
            targets = [o["center"] for o in self.encoder.prev_objects]
            x, y = random.choice(targets) if targets \
                else (random.randint(0, 63), random.randint(0, 63))
            action.set_data({"x": x, "y": y})
        action.reasoning = "v003 obs-encoder explorer"

        self.prev_key = key
        self.prev_action = action.name
        return action
