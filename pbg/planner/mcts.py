"""planner/mcts.py — MCTS for the stochastic-transition mode (spec §10): maximise expected progress, 200 rollouts."""
from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Optional

from ..core.types import Action, Scene


class _Node:
    __slots__ = ("scene", "parent", "action", "children", "n", "value", "untried")

    def __init__(self, scene, parent, action, actions):
        self.scene, self.parent, self.action = scene, parent, action
        self.children: list[_Node] = []; self.n = 0; self.value = 0.0; self.untried = list(actions)


def mcts(start: Scene, step: Callable[[Scene, Action], Optional[Scene]], actions_fn: Callable[[Scene], list[Action]],
         is_goal: Callable[[Scene], bool], progress: Callable[[Scene], float], *, rollouts: int = 200, horizon: int = 20,
         c: float = 1.0, rng: Optional[random.Random] = None) -> Optional[list[Action]]:
    rng = rng or random.Random(0)
    root = _Node(start, None, None, actions_fn(start))
    if is_goal(start):
        return []
    for _ in range(rollouts):
        node = root; depth = 0
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ch.value / max(1, ch.n) + c * math.sqrt(math.log(max(1, node.n)) / max(1, ch.n)))
            depth += 1
        if node.untried:
            a = node.untried.pop(rng.randrange(len(node.untried)))
            nxt = step(node.scene, a)
            if nxt is None:
                continue
            child = _Node(nxt, node, a, actions_fn(nxt)); node.children.append(child); node = child; depth += 1
        # rollout
        s = node.scene; reward = progress(s)
        for _ in range(max(0, horizon - depth)):
            if is_goal(s):
                reward = 1.0; break
            acts = actions_fn(s)
            if not acts:
                break
            nxt = step(s, rng.choice(acts))
            if nxt is None:
                break
            s = nxt; reward = max(reward, progress(s))
        while node is not None:
            node.n += 1; node.value += reward; node = node.parent
    if not root.children:
        return None
    path = []; node = root
    while node.children:
        node = max(node.children, key=lambda ch: ch.n)
        path.append(node.action)
        if is_goal(node.scene):
            break
    return path[:1] if path else None
