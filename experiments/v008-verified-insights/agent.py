"""v008: verified action facts + code-computed reward over the v007 reflection loop.

v007의 실패 원인(9b 평가자의 인사이트가 부정 진술뿐, 보상 -1 붕괴)을 겨냥한 보완:

  1. ActionModel — 액션별 효과를 코드로 집계 ("ACTION1: moves c4 (0,-1) 12/15").
     ACTION6는 클릭한 셀 색으로 세분(ACTION6@c7). 게임별 "Verified Action Facts"로
     insights.md에 저장(덮어쓰기)하고 플래너/평가자 프롬프트 최상단에 주입.
     여러 게임에 걸친 집계("ACTION1 had visible effect in 3/4 games")도 렌더.
  2. 코드 보상 — levelups + novelty + change-rate 로 계산. 재계획 트리거는 이
     값으로만 판단. LLM의 reward_score는 참고용 로그.
  3. LLM 인사이트를 구조화({"action","fact"})하고 이 게임에서 실제 시도된 액션에
     대한 것만 저장(검증). next_strategy(다음 창에서 시도할 액션/대상)를 별도 저장.
  4. 평가 기준 개정 제안에 근사 중복 필터. 평가자 모델은 EVAL_MODEL로 분리
     (기본 qwen3.5:9b; 35b로 비대칭 구성 실험).

env: QWEN_ENDPOINT, QWEN_MODEL(플래너), EVAL_MODEL(평가자), INSIGHTS_PATH, INSIGHTS_RESET.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
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
        self.last_events: list[str] = []   # v008: 일반화된 효과 시그니처

    def observe(self, action_name: Optional[str], counter: int,
                latest_frame: FrameData) -> None:
        if not latest_frame.frame:
            return
        grid = latest_frame.frame[-1]
        objects = extract_objects(grid)
        if action_name is not None and self.prev_grid is not None:
            n = cells_changed(self.prev_grid, grid)
            events: list[str] = []
            if n == 0:
                sig = "no change"
                events = ["no change"]
            else:
                d = diff_objects(self.prev_objects, objects)
                events += [f"moves c{c['color']} ({dx:+d},{dy:+d})" for _, c, dx, dy in d["moved"][:3]]
                events += [f"{'grows' if a['size'] > b['size'] else 'shrinks'} c{a['color']}"
                           for b, a in d["resized"][:3]]
                events += [f"adds c{o['color']}" for o in d["appeared"][:2]]
                events += [f"removes c{o['color']}" for o in d["disappeared"][:2]]
                if not events:
                    events = [f"changes ~{min(n, 999)} cells"]
                bits = [f"{n} cells"]
                bits += [f"moved {obj_tag(c)} ({dx:+d},{dy:+d})"
                         for _, c, dx, dy in d["moved"][:3]]
                bits += [f"resized c{a['color']} s{b['size']}→{a['size']} "
                         f"@{a['center'][0]},{a['center'][1]}"
                         for b, a in d["resized"][:3]]
                bits += [f"+{obj_tag(o)}" for o in d["appeared"][:2]]
                bits += [f"-{obj_tag(o)}" for o in d["disappeared"][:2]]
                sig = "; ".join(bits)
            self.last_events = events
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


# ─────────────────────── action model (verified facts) ──────────────────────

class ActionModel:
    """액션 키별 효과 시그니처 카운트. 코드로만 갱신되므로 '검증된' 사실."""

    def __init__(self):
        self.tried: Counter = Counter()
        self.counts: dict[str, Counter] = {}

    def record(self, key: str, events: list[str]) -> None:
        self.tried[key] += 1
        c = self.counts.setdefault(key, Counter())
        for e in set(events):          # 한 스텝에 같은 시그니처 중복 → 1회
            c[e] += 1

    def facts(self, top: int = 2, min_tried: int = 2) -> list[str]:
        out = []
        for key in sorted(self.tried, key=lambda k: (k.split("@")[0], k)):
            n = self.tried[key]
            if n < min_tried:
                continue
            common = self.counts[key].most_common(top)
            out.append(f"{key}: " + ", ".join(f"{e} {m}/{n}" for e, m in common))
        return out


# ───────────────────────── insight memory (markdown) ────────────────────────

DEFAULT_CRITERIA = """1. 외적 보상 (Extrinsic Reward): 승리 목표(레벨 완료)에 직접적으로 다가갔는가? (+1.0 ~ -1.0)
   - 레벨 완료, 새로운 화면 상태 도달, 목표 객체에 가까워짐 → 플러스
   - 아무 변화도 없는 행동 반복, 같은 상태 맴돌기, 액션 낭비 → 마이너스
2. 내적 보상 (Intrinsic Reward - 탐색 및 유연성): 새로운 전략을 시도하거나 변수에 잘 대처했는가? (0.0 ~ +0.5)
   - 이전에 시도하지 않은 낯선 행동(호기심)에 가산점 부여"""

MAX_GENERAL = 12
MAX_PER_GAME = 8
MAX_LOG = 20


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
            if len(w) > 2}


def is_near_duplicate(item: str, pool: list[str], threshold: float = 0.6) -> bool:
    a = _tokens(item)
    if not a:
        return True
    for other in pool:
        b = _tokens(other)
        if b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


class InsightMemory:
    """insights.md 를 읽고/갱신하고/저장하는 단순 마크다운 메모리."""

    def __init__(self, path: Path):
        self.path = path
        self.version = 1
        self.criteria = DEFAULT_CRITERIA
        self.general: list[str] = []
        self.games: dict[str, list[str]] = {}
        self.log: list[dict] = []          # {game, turn, reward, note}
        self.criteria_history: list[str] = []
        self.facts: dict[str, list[str]] = {}      # game -> verified action facts (code)
        self.strategy: dict[str, str] = {}         # game -> next strategy (LLM, replaced)
        if path.exists():
            self._load(path.read_text())

    # ---- parsing (우리가 쓴 형식만 파싱하면 충분) ----
    def _load(self, text: str) -> None:
        section, game = None, None
        criteria_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("## Evaluation Criteria"):
                section = "criteria"
                m = line.rstrip(")").rsplit("(v", 1)
                try:
                    self.version = int(m[1])
                except (IndexError, ValueError):
                    pass
                continue
            if line.startswith("## General Insights"):
                section = "general"; continue
            if line.startswith("## Game Notes"):
                section = "games"; continue
            if line.startswith("## Reward Log"):
                section = "log"; continue
            if line.startswith("## Criteria History"):
                section = "hist"; continue
            if line.startswith("## Verified Action Facts"):
                section = "facts"; continue
            if line.startswith("## Next Strategy"):
                section = "strategy"; continue
            if section == "facts":
                if line.startswith("### "):
                    game = line[4:].strip(); self.facts.setdefault(game, [])
                elif game and line.startswith("- "):
                    self.facts[game].append(line[2:].strip())
                continue
            if section == "strategy" and line.startswith("- "):
                g, _, txt = line[2:].partition(": ")
                if txt:
                    self.strategy[g.strip()] = txt.strip()
                continue
            if section == "criteria":
                criteria_lines.append(line)
            elif section == "general" and line.startswith("- "):
                self.general.append(line[2:].strip())
            elif section == "games":
                if line.startswith("### "):
                    game = line[4:].strip()
                    self.games.setdefault(game, [])
                elif game and line.startswith("- "):
                    self.games[game].append(line[2:].strip())
            elif section == "log" and line.startswith("| ") and not line.startswith("| game") \
                    and not line.startswith("|--"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) >= 4:
                    try:
                        self.log.append({"game": cells[0], "turn": int(cells[1]),
                                         "reward": float(cells[2]), "note": cells[3]})
                    except ValueError:
                        pass
            elif section == "hist" and line.startswith("- "):
                self.criteria_history.append(line[2:].strip())
        if criteria_lines:
            self.criteria = "\n".join(criteria_lines).strip() or DEFAULT_CRITERIA

    # ---- updates ----
    def add_general(self, items: list[str]) -> int:
        n = 0
        for it in items:
            it = it.strip()
            if it and not is_near_duplicate(it, self.general):
                self.general.append(it); n += 1
        self.general = self.general[-MAX_GENERAL:]
        return n

    def add_game(self, game: str, items: list[str]) -> int:
        notes = self.games.setdefault(game, [])
        n = 0
        for it in items:
            it = it.strip()
            if it and not is_near_duplicate(it, notes):
                notes.append(it); n += 1
        self.games[game] = notes[-MAX_PER_GAME:]
        return n

    def set_facts(self, game: str, facts: list[str]) -> None:
        self.facts[game] = facts[:12]

    def set_strategy(self, game: str, text: str) -> None:
        text = text.strip().replace("\n", " ")
        if text:
            self.strategy[game] = text[:200]

    def cross_game_facts(self) -> list[str]:
        """액션(베이스 이름)별로 몇 게임에서 가시 효과가 있었는지 + 대표 효과."""
        per_action: dict[str, list[tuple[str, str]]] = {}
        for game, facts in self.facts.items():
            for f in facts:
                key, _, rest = f.partition(": ")
                base = key.split("@")[0]
                top = rest.split(", ")[0] if rest else ""
                if top and not top.startswith("no change"):
                    per_action.setdefault(base, []).append((game, top.rsplit(" ", 1)[0]))
        n_games = max(1, len(self.facts))
        out = []
        for base in sorted(per_action):
            games = {g for g, _ in per_action[base]}
            kinds = Counter(e.split(" ")[0] for _, e in per_action[base]).most_common(1)[0][0]
            out.append(f"{base}: visible effect in {len(games)}/{n_games} games (typically {kinds})")
        return out

    def update_criteria(self, suggestion: str) -> bool:
        suggestion = suggestion.strip()
        if not suggestion:
            return False
        existing = [l[2:] for l in self.criteria.splitlines() if l.startswith("+ ")]
        if is_near_duplicate(suggestion, existing + self.criteria_history, 0.5):
            return False
        self.criteria_history.append(f"v{self.version}→v{self.version + 1}: {suggestion[:200]}")
        self.criteria_history = self.criteria_history[-10:]
        self.version += 1
        # 기준은 계속 자라지 않도록 추가 항목을 최대 4개까지만 유지
        extra = [l for l in self.criteria.splitlines() if l.startswith("+ ")]
        extra.append(f"+ (v{self.version}) {suggestion[:300]}")
        base = [l for l in self.criteria.splitlines() if not l.startswith("+ ")]
        self.criteria = "\n".join(base + extra[-4:])
        return True

    def add_log(self, game: str, turn: int, reward: float, note: str) -> None:
        self.log.append({"game": game, "turn": turn, "reward": reward,
                         "note": note.replace("|", "/")[:100]})
        self.log = self.log[-MAX_LOG:]

    # ---- rendering ----
    def render(self) -> str:
        out = ["# Agent Insight Memory",
               f"updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
               "", f"## Evaluation Criteria (v{self.version})", self.criteria, "",
               "## General Insights"]
        out += [f"- {g}" for g in self.general] or ["- (none yet)"]
        out += ["", "## Verified Action Facts (code-measured: effect count/tried)"]
        for game, facts in self.facts.items():
            out.append(f"### {game}")
            out += [f"- {f}" for f in facts]
        out += ["", "## Next Strategy"]
        out += [f"- {g}: {t}" for g, t in self.strategy.items()]
        out += ["", "## Game Notes"]
        for game, notes in self.games.items():
            out.append(f"### {game}")
            out += [f"- {n}" for n in notes]
        out += ["", "## Reward Log", "| game | turn | reward | note |", "|---|---|---|---|"]
        out += [f"| {e['game']} | {e['turn']} | {e['reward']:+.2f} | {e['note']} |" for e in self.log]
        out += ["", "## Criteria History"]
        out += [f"- {h}" for h in self.criteria_history]
        return "\n".join(out) + "\n"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.render())

    def prompt_block(self, game: str, max_chars: int = 2400) -> str:
        """플래너 프롬프트에 넣을 압축 버전."""
        lines = [f"[ACTION MODEL: measured effects in this game {game}]"]
        lines += [f"- {f}" for f in self.facts.get(game, [])] or ["- (nothing measured yet)"]
        cross = [f"- {c}" for c in self.cross_game_facts() if game not in c]
        if cross:
            lines.append("[ACTION MODEL: across games]")
            lines += cross[:6]
        if game in self.strategy:
            lines.append(f"[NEXT STRATEGY suggested by evaluator] {self.strategy[game]}")
        lines.append("[INSIGHTS: general, learned across games]")
        lines += [f"- {g}" for g in self.general[-6:]] or ["- (none yet)"]
        notes = self.games.get(game, [])
        lines.append(f"[INSIGHTS: this game {game}]")
        lines += [f"- {n}" for n in notes[-6:]] or ["- (none yet)"]
        text = "\n".join(lines)
        return text[:max_chars]


# ─────────────────────────── reward evaluator (L3) ──────────────────────────

EVALUATOR_TEMPLATE = """[System Role]
당신은 복잡한 환경(예: 보드게임)에서 에이전트의 행동을 평가하고 보상(Reward)을 산정하는 '객관적인 보상 평가자(Reward Evaluator)'입니다.
주어진 평가 기준에 따라 에이전트의 행동을 분석하고, 수치화된 보상과 그 이유를 JSON 형식으로만 출력하십시오.

[Environment Context]
- 현재 턴/진행도: {current_turn}
- 현재 보드/환경 상태: {current_state}
- 에이전트의 목표: {agent_goal}

[Agent Action]
- 에이전트가 선택한 행동: {agent_action}

[Measured by code — 사실이며 반박 불가]
{measured}

[Evaluation Criteria (현재 버전: {template_version})]
{criteria}

[Accumulated Insights] (이미 기록된 것 — 같은 내용을 다시 제안하지 말고 새로운 사실만. 부정 진술("X는 무의미")보다 "액션→효과" 사실을 우선)
{insights}

[Output Format]
반드시 아래의 JSON 형식으로만 응답하십시오.
{{
  "reasoning": "행동에 대한 단계별 평가 논리 (외적/내적 보상 기준)",
  "reward_score": [최종 합산된 실수 형태의 보상 점수, 예: 0.8],
  "template_update_suggestion": {{
    "needs_update": [true/false],
    "suggestion": "현재 평가 기준([Evaluation Criteria])에 추가, 삭제 또는 수정해야 할 사항이 있다면 그 이유와 구체적인 문구를 제안하세요."
  }},
  "game_insights": [{{"action": "ACTION1..ACTION7 중 하나 또는 any", "fact": "그 액션이 이 게임에서 하는 일 또는 목표에 대한 구체적 사실 (영어 1문장). 위 [Measured] 와 모순되면 안 됨"}}],
  "general_insights": ["처음 보는 다른 게임에도 적용될 일반 교훈 (영어, 최대 1개, 없으면 빈 배열)"],
  "next_strategy": {{"action": "다음 25턴에 집중할 액션 (ACTION1..7)", "target": "ACTION6이면 클릭할 객체 색 cN, 아니면 none", "why": "영어 1문장"}}
}}"""


class RewardEvaluator:
    def __init__(self, planner: "QwenPlanner", memory: InsightMemory,
                 model: Optional[str] = None):
        self.planner = planner
        self.memory = memory
        self.model = model or planner.model
        self.code_rewards: list[float] = []
        self.dropped_insights = 0
        self.calls = 0
        self.errors = 0
        self.total_s = 0.0
        self.criteria_updates = 0
        self.rewards: list[float] = []
        self.samples: list[dict] = []

    def evaluate(self, game_id: str, turn: int, state_text: str, goal: str,
                 action_text: str, measured: str, code_reward: float,
                 tried: set[str]) -> Optional[dict]:
        self.code_rewards.append(code_reward)
        prompt = EVALUATOR_TEMPLATE.format(
            current_turn=turn, current_state=state_text, agent_goal=goal,
            agent_action=action_text, template_version=f"v{self.memory.version}",
            criteria=self.memory.criteria, measured=measured,
            insights=self.memory.prompt_block(game_id, 1600))
        t0 = time.time()
        try:
            content = self.planner._complete_raw(
                system="You are a strict JSON-only reward evaluator.", user=prompt,
                num_predict=700, model=self.model)
            data = json.loads(content)
            reward = float(data.get("reward_score", 0.0))
            reward = max(-1.0, min(1.5, reward))
        except Exception:
            self.errors += 1
            self.total_s += time.time() - t0
            return None
        self.calls += 1
        self.total_s += time.time() - t0
        self.rewards.append(reward)
        reasoning = str(data.get("reasoning", ""))[:300]
        self.memory.add_log(game_id, turn, reward, f"code={code_reward:+.2f} " + reasoning[:90])
        upd = data.get("template_update_suggestion") or {}
        if isinstance(upd, dict) and upd.get("needs_update") is True:
            sug = str(upd.get("suggestion", "")).strip()
            if len(sug) > 15 and self.memory.update_criteria(sug):
                self.criteria_updates += 1
        gi = data.get("game_insights") or []
        ge = data.get("general_insights") or []
        if isinstance(gi, list):
            kept = []
            for x in gi[:3]:
                if isinstance(x, dict):
                    act = str(x.get("action", "any")).upper().strip()
                    fact = str(x.get("fact", "")).strip()
                else:
                    act, fact = "ANY", str(x).strip()
                if not fact:
                    continue
                # 검증: 이 게임에서 실제 시도된 액션(또는 any)에 대한 사실만 저장
                if act != "ANY" and act not in tried:
                    self.dropped_insights += 1
                    continue
                kept.append(f"{act}: {fact}" if act != "ANY" else fact)
            self.memory.add_game(game_id, kept[:2])
        ns = data.get("next_strategy") or {}
        if isinstance(ns, dict) and ns.get("action"):
            self.memory.set_strategy(
                game_id, f"{str(ns.get('action')).upper()} target={ns.get('target', 'none')}"
                         f" — {str(ns.get('why', ''))[:120]}")
        if isinstance(ge, list):
            self.memory.add_general([str(x) for x in ge][:1])
        self.memory.save()
        if len(self.samples) < 4:
            self.samples.append({"turn": turn, "reward": reward,
                                 "raw": content[:900]})
        return {"reward": reward, "code_reward": code_reward, "reasoning": reasoning,
                "strategy": self.memory.strategy.get(game_id, "")}


# ─────────────────────────────── L2 planner ─────────────────────────────────

SYSTEM_PROMPT = """You control an agent in an unknown 64x64 grid puzzle game (ARC-AGI-3).
Cells hold colors 0-15. You must discover the rules by acting, then complete levels.
Score favors completing levels in FEW actions.
Observation format: [OBJECTS] cN=color sN=size, [RECENT] effects of past actions
(moved/resized/+appeared/-disappeared), [ACTION_STATS] how often each action changed anything.
Actions: 1-5 and 7 are abstract buttons (meaning differs per game; check [RECENT]).
Action 6 is a click and needs x,y (0-63) — click object centers, not empty background.
[ACTION MODEL] below is MEASURED ground truth about what each action did in this
game (effect count/tried) — trust it over guesses. Use it plus [NEXT STRATEGY] and
[INSIGHTS] (learned from earlier play, possibly other games) to avoid useless
actions and to make progress. ACTION6@cN means clicking a cell of color N.
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

    def _complete_raw(self, system: str, user: str, num_predict: int = 500,
                      model: Optional[str] = None) -> str:
        """Kaggle(v005)에서는 이 메서드만 in-process vLLM으로 교체."""
        body = json.dumps({
            "model": model or self.model, "stream": False, "think": False,
            "format": "json",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"num_predict": num_predict, "temperature": 0.2, "seed": 1337},
        }).encode()
        req = urllib.request.Request(f"{self.endpoint}/api/chat", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)["message"]["content"]

    def _complete(self, obs_text: str, insights: str = "") -> str:
        system = SYSTEM_PROMPT + ("\n\n" + insights if insights else "")
        return self._complete_raw(system, obs_text)

    def plan(self, obs_text: str, insights: str = "") -> Optional[list[dict]]:
        if not self.enabled:
            return None
        t0 = time.time()
        try:
            content = self._complete(obs_text, insights)
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
    REFLECT_EVERY = 25     # 보상 평가자 호출 주기(액션)
    MAX_REFLECT_CALLS = 16 # 게임당 평가자 예산
    LOW_REWARD = -0.2      # 코드 보상이 이 이하이면 계획 폐기 + 즉시 재계획 허용
    EVAL_MODEL = os.getenv("EVAL_MODEL", "qwen3.5:9b")
    INSIGHTS_PATH = Path(__file__).resolve().parent / "memory" / "insights.md"

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
        path = Path(os.getenv("INSIGHTS_PATH", str(self.INSIGHTS_PATH)))
        # 리셋은 프로세스당 1회만 (러너는 게임마다 새 에이전트를 만든다)
        if os.environ.pop("INSIGHTS_RESET", None) == "1" and path.exists():
            path.unlink()
        self.memory = InsightMemory(path)
        self.memory_version_at_start = self.memory.version
        self.memory_general_at_start = len(self.memory.general)
        self.evaluator = RewardEvaluator(self.planner, self.memory, self.EVAL_MODEL)
        self.action_model = ActionModel()
        self.prev_model_key: Optional[str] = None
        self.window: list[dict] = []          # 평가 창의 행동 로그
        self.window_start_level = 0
        self.window_start_states = 0
        self.last_reflect_at = 0
        self.last_reward: Optional[float] = None
        self.last_eval_note = ""
        self.force_plan = False
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
            "eval_model": self.evaluator.model,
            "reflect_calls": self.evaluator.calls,
            "code_rewards": [round(r, 2) for r in self.evaluator.code_rewards],
            "mean_code_reward": (round(sum(self.evaluator.code_rewards) / len(self.evaluator.code_rewards), 3)
                                 if self.evaluator.code_rewards else None),
            "dropped_insights": self.evaluator.dropped_insights,
            "action_facts": self.memory.facts.get(self._short_game, []),
            "reflect_errors": self.evaluator.errors,
            "reflect_time_s": round(self.evaluator.total_s, 1),
            "rewards": [round(r, 2) for r in self.evaluator.rewards],
            "mean_reward": (round(sum(self.evaluator.rewards) / len(self.evaluator.rewards), 3)
                            if self.evaluator.rewards else None),
            "criteria_updates": self.evaluator.criteria_updates,
            "criteria_version": self.memory.version,
            "general_insights": len(self.memory.general),
            "game_insights": len(self.memory.games.get(self._short_game, [])),
            "insights_path": str(self.memory.path),
            "reflect_samples": self.evaluator.samples,
        }

    @property
    def _short_game(self) -> str:
        return str(self.game_id).split("-")[0]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _record_effect(self, latest_frame: FrameData) -> None:
        if self.prev_key is None or self.prev_action is None:
            return
        key = frame_key(latest_frame)
        leveled = latest_frame.levels_completed > self.prev_level
        changed = (key != self.prev_key) or leveled
        self.effects.setdefault(self.prev_key, {})[self.prev_action] = changed
        self.window.append({"a": self.prev_action, "changed": changed, "leveled": leveled})
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
        if self.force_plan and since >= 3:
            self.force_plan = False
            return True
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
        if self.prev_model_key is not None and self.encoder.last_events:
            self.action_model.record(self.prev_model_key, self.encoder.last_events)
            self.encoder.last_events = []

        obs = self.encoder.encode(latest_frame, self.tally)
        self.obs_chars.append(len(obs))
        if self.action_counter % 25 == 0 and len(self.obs_samples) < 12:
            self.obs_samples.append((self.action_counter, obs))

        key = frame_key(latest_frame)
        candidates = self._candidate_actions(latest_frame)

        if self._should_reflect():
            self._reflect(latest_frame, obs)

        if self._should_plan(latest_frame):
            insights = self.memory.prompt_block(self._short_game)
            if self.last_reward is not None:
                insights += (f"\n[LAST EVAL] reward {self.last_reward:+.2f}: "
                             f"{self.last_eval_note[:160]}")
            planned = self.planner.plan(obs, insights)
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
        self.prev_model_key = action.name
        if action.is_complex() and latest_frame.frame:
            d = action.action_data if hasattr(action, "action_data") else None
            x, y = (d.x, d.y) if d is not None and hasattr(d, "x") else (None, None)
            if x is not None:
                try:
                    self.prev_model_key = f"ACTION6@c{latest_frame.frame[-1][y][x]}"
                except (IndexError, TypeError):
                    pass
        return action

    # ----- L3: reflection -----
    def _should_reflect(self) -> bool:
        if not self.planner.enabled or self.evaluator.calls + self.evaluator.errors >= self.MAX_REFLECT_CALLS:
            return False
        return (self.action_counter - self.last_reflect_at) >= self.REFLECT_EVERY and len(self.window) >= 5

    def _reflect(self, latest_frame: FrameData, obs: str) -> None:
        self.last_reflect_at = self.action_counter
        # 행동 창 압축: "ACTION1×5(changed 3) ACTION6×2(changed 0) ..."
        runs: list[list] = []
        for w in self.window:
            if runs and runs[-1][0] == w["a"]:
                runs[-1][1] += 1; runs[-1][2] += int(w["changed"])
            else:
                runs.append([w["a"], 1, int(w["changed"])])
        action_text = " ".join(f"{a}×{n}(changed {c})" for a, n, c in runs[-25:])
        leveled = latest_frame.levels_completed - self.window_start_level
        new_states = len(self._seen_states) - self.window_start_states
        summary = (f"window of {len(self.window)} actions: levels gained={leveled}, "
                   f"new unique states={new_states}, actions changing screen="
                   f"{sum(w['changed'] for w in self.window)}")
        n_w = max(1, len(self.window))
        n_changed = sum(w["changed"] for w in self.window)
        novelty = new_states / n_w
        change = n_changed / n_w
        code_reward = round(min(1.5, 1.0 * leveled + 0.4 * novelty + 0.2 * change
                                - 0.5 * (1.0 - change)), 3)
        facts = self.action_model.facts()
        self.memory.set_facts(self._short_game, facts)
        measured = "\n".join([f"- code_reward={code_reward:+.2f} (levelups {leveled}, "
                              f"novelty {novelty:.2f}, change-rate {change:.2f})"]
                             + [f"- {f}" for f in facts[:12]])
        hypo = self.planner.last_hypothesis or "(no hypothesis yet)"
        goal = (f"complete levels of unknown game {self._short_game} "
                f"({latest_frame.levels_completed}/{latest_frame.win_levels} done) in few actions; "
                f"planner hypothesis: {hypo}")
        res = self.evaluator.evaluate(self._short_game, self.action_counter,
                                      obs[:1500], goal, f"{summary}\n{action_text}",
                                      measured, code_reward, set(self.tally))
        self.window.clear()
        self.window_start_level = latest_frame.levels_completed
        self.window_start_states = len(self._seen_states)
        # 재계획 트리거는 코드 보상으로만 판단 (LLM 보상 붕괴 대비)
        if code_reward <= self.LOW_REWARD:
            self.plan_queue.clear()
            self.force_plan = True
        if res is None:
            self.memory.save()
            return
        self.last_reward = code_reward
        self.last_eval_note = res["reasoning"]

    def _candidate_actions(self, latest_frame: FrameData) -> list[GameAction]:
        allowed = set(latest_frame.available_actions or range(1, 8))
        acts = [a for a in SIMPLE_ACTIONS if a.value in allowed]
        if 6 in allowed:
            acts.append(GameAction.ACTION6)
        return acts or SIMPLE_ACTIONS
