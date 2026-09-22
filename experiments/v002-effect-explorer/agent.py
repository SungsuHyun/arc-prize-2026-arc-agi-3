"""v002: effect-aware explorer + (stub) LLM planner interface.

Layered design (docs/005 참고):
  L0 reflex   — 프레임 diff로 no-op 액션 필터링 (매 액션)
  L1 explorer — 상태별 액션-효과 기록, 미시도 액션 우선, 효과 있던 액션 선호
  L2 planner  — Qwen 플래너 자리 (이 버전은 스텁: 항상 None → L1로 폴백)

점수는 레벨 완료에서만 나오므로, 우선 '프레임을 바꾸는 액션'을 빨리 찾아
행동 예산을 낭비하지 않는 것이 이 버전의 목표.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

SIMPLE_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]


def frame_key(frame: FrameData) -> int:
    """Hash of the visible grid (state identity for the effect table)."""
    return hash(tuple(tuple(map(tuple, g)) for g in frame.frame))


def object_centroids(frame: FrameData, max_objects: int = 32) -> list[tuple[int, int]]:
    """Centroids of connected same-color components — ACTION6 click candidates.

    Clicking object centers beats random pixels: most games attach meaning
    to objects, not to empty background cells.
    """
    if not frame.frame:
        return []
    grid = frame.frame[-1]
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    out: list[tuple[int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy][sx]:
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
            # skip the background-sized blob and 1px noise
            if 4 <= len(cells) <= h * w // 4:
                cy = sum(c[0] for c in cells) // len(cells)
                cx = sum(c[1] for c in cells) // len(cells)
                out.append((cx, cy))
    random.shuffle(out)
    return out[:max_objects]


class QwenPlanner:
    """L2 자리표시자. v004에서 Qwen(~27B급, 양자화) 연결 예정.

    계약: plan(observation: dict) -> Optional[list[GameAction]]
    모델이 없으면 None을 반환하고 에이전트는 L1로 폴백한다.
    """

    def plan(self, observation: dict) -> Optional[list]:
        return None


class MyAgent(Agent):
    MAX_ACTIONS = 400

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(1337)
        # effect table: state_key -> {action_name: frame_changed(bool)}
        self.effects: dict[int, dict[str, bool]] = {}
        # global tally: action_name -> [changed, tried]
        self.tally: dict[str, list[int]] = {}
        self.prev_key: Optional[int] = None
        self.prev_action: Optional[str] = None
        self.prev_level = 0
        self.planner = QwenPlanner()
        self.plan_queue: list = []

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _record_effect(self, latest_frame: FrameData) -> None:
        """L0: did the previous action change anything?"""
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
            return GameAction.RESET

        self._record_effect(latest_frame)
        key = frame_key(latest_frame)
        candidates = self._candidate_actions(latest_frame)

        # L2: consult the planner queue first (stub → always empty)
        if not self.plan_queue:
            planned = self.planner.plan({
                "levels_completed": latest_frame.levels_completed,
                "win_levels": latest_frame.win_levels,
                "tally": self.tally,
            })
            if planned:
                self.plan_queue = list(planned)
        if self.plan_queue:
            action = self.plan_queue.pop(0)
        else:
            # L1: untried actions in this state first
            tried = self.effects.get(key, {})
            untried = [a for a in candidates if a.name not in tried]
            if untried:
                action = random.choice(untried)
            else:
                # then weight by global change-rate (+ smoothing), skip known
                # no-ops in this exact state
                effective = [a for a in candidates if tried.get(a.name, True)]
                pool = effective or candidates
                weights = [
                    (self.tally.get(a.name, [0, 0])[0] + 1)
                    / (self.tally.get(a.name, [0, 0])[1] + 2)
                    for a in pool
                ]
                action = random.choices(pool, weights=weights, k=1)[0]

        if action.is_complex():  # ACTION6 needs x, y
            targets = object_centroids(latest_frame)
            x, y = random.choice(targets) if targets \
                else (random.randint(0, 63), random.randint(0, 63))
            action.set_data({"x": x, "y": y})
        action.reasoning = f"v002 explorer (state has {len(self.effects.get(key, {}))} tried)"

        self.prev_key = key
        self.prev_action = action.name
        return action
