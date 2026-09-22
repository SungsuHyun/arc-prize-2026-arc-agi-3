"""v010: LLM only on demand (v009 + click-game gate).

v009 25게임: LLM 없음 0.408 vs 35b 0.388 — 내비 게이트로 이동 게임의 손해는
사라졌지만, 클릭 게임에서는 40액션마다 오는 LLM 계획이 클릭 스위프를 밀어내
레벨을 잃음 (ft09/sp80 lv1→0, lp85/vc33 하락). 그래서 클릭 게임에서도 LLM은
(a) 레벨 진입 직후 1회, (b) stuck(새 상태 없음)일 때만. 주기적(PLAN_EVERY)
호출은 제거. 인프라(백엔드/스모크)는 그대로 유지되므로 Kaggle에서 LLM이 살아
있어도 파이썬 계층을 방해하지 않는다.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
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


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


HEX = "0123456789abcdef"


class ObservationEncoder:
    def __init__(self, recent_k: int = 8):
        self.recent: deque[dict] = deque(maxlen=recent_k)
        self.prev_grid = None
        self.prev_objects: list[dict] = []
        self.move_log: dict[str, list[tuple[int, int]]] = {}   # action → deltas
        self.player_votes: Counter = Counter()                  # (color,size) → n
        self.player_parts: set[tuple[int, int]] = set()          # co-moving (color,size)
        self.gauge_hist: dict[int, list[int]] = {}              # color → sizes
        self.memory: list[str] = []                             # level/game-over notes
        self.prev_hypothesis = ""
        self.last_map: Optional[dict] = None
        self.nav_lines: list[str] = []

    # ── bookkeeping ────────────────────────────────────────────────────────
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
                # 아바타 후보: 가장 크게 이동한 객체 (바닥처럼 거대한 객체는 제외 —
                # 아바타가 지나가면 바닥의 중심도 1px씩 흔들려 오인됨)
                movers = [m for m in d["moved"] if m[1]["size"] <= 400]
                if movers and action_name != "ACTION6":     # 클릭은 아바타 이동이 아님
                    _, c, dx, dy = max(movers, key=lambda m: (abs(m[2]) + abs(m[3]), m[1]["size"]))
                    log = self.move_log.setdefault(action_name, [])
                    log.append((dx, dy))
                    del log[:-6]
                    self.player_votes[(c["color"], c["size"])] += 1
                    # 같은 델타로 함께 움직인 조각들 = 다색 아바타의 부분
                    self.player_parts = {(m[1]["color"], m[1]["size"]) for m in movers
                                         if (m[2], m[3]) == (dx, dy)}
                for _, a in d["resized"]:
                    h = self.gauge_hist.setdefault(a["color"], [])
                    if len(h) >= 3 and h[-1] < h[-2] < h[-3] and a["size"] < 300 \
                            and a["size"] - h[-1] >= 4 * (h[-2] - h[-1]):
                        self.remember(f"a{counter}: c{a['color']} bar jumped {h[-1]}→{a['size']} "
                                      "(refilled: level restarted, probably energy ran out)")
                    h.append(a["size"])
                    del h[:-10]
            last = self.recent[-1] if self.recent else None
            if last and last["act"] == action_name and last["sig"] == sig:
                last["n"] += 1
                last["a1"] = counter
            else:
                self.recent.append({"a0": counter, "a1": counter, "n": 1,
                                    "act": action_name, "sig": sig})
        self.prev_grid = grid
        self.prev_objects = objects

    def remember(self, note: str) -> None:
        self.memory.append(note[:300])
        del self.memory[:-4]

    # ── derived facts ──────────────────────────────────────────────────────
    def move_delta(self, action_name: str) -> Optional[tuple[int, int]]:
        """액션의 지배적 이동 델타 (2회 이상 관측 + 과반)."""
        log = self.move_log.get(action_name) or []
        if not log:
            return None
        if len(log) == 1:                       # 1회 관측: 작은 이동만 잠정 수용
            d = log[0]
            return d if abs(d[0]) + abs(d[1]) <= 8 else None
        (delta, n), = Counter(log).most_common(1)
        return delta if n >= 2 and n * 2 > len(log) else None

    def player(self) -> Optional[dict]:
        """아바타 = 최다 득표 객체 + 함께 움직인 인접 조각들의 합집합 bbox."""
        main = None
        for (color, size), _ in self.player_votes.most_common(3):
            for o in self.prev_objects:
                if o["color"] == color and o["size"] == size:
                    main = o
                    break
            if main:
                break
        if main is None:
            return None
        x0, y0, x1, y1 = main["bbox"]
        size, colors = main["size"], {main["color"]}
        for o in self.prev_objects:
            if o is main or (o["color"], o["size"]) not in self.player_parts:
                continue
            bx0, by0, bx1, by1 = o["bbox"]
            if bx0 > x1 + 1 or bx1 < x0 - 1 or by0 > y1 + 1 or by1 < y0 - 1:
                continue                                    # 인접하지 않음
            x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
            size += o["size"]
            colors.add(o["color"])
        return {"color": main["color"], "colors": sorted(colors), "size": size,
                "bbox": (x0, y0, x1, y1), "center": ((x0 + x1) // 2, (y0 + y1) // 2)}

    def cell_size(self) -> int:
        g = 0
        for a in self.move_log:                 # 지배적 델타만 사용 (텔레포트/노이즈 제외)
            d = self.move_delta(a)
            if d:
                for v in (abs(d[0]), abs(d[1])):
                    if v:
                        g = _gcd(g, v)
        return g if 2 <= g <= 8 else 4

    def map_data(self, player: Optional[dict]) -> Optional[dict]:
        """코스 맵: cell/origin/colors[row][col]/player block. (없으면 None)"""
        if self.prev_grid is None:
            return None
        cell = self.cell_size()
        ox = oy = 0
        if player:
            ox, oy = player["bbox"][0] % cell, player["bbox"][1] % cell
        colors, y = [], oy
        while y < 64 and len(colors) < 16:
            row, x = [], ox
            while x < 64 and len(row) < 16:
                block = [self.prev_grid[yy][xx] for yy in range(y, min(64, y + cell))
                         for xx in range(x, min(64, x + cell))]
                row.append(Counter(block).most_common(1)[0][0])
                x += cell
            colors.append(row)
            y += cell
        pb = None
        if player:
            pb = ((player["bbox"][0] - ox) // cell, (player["bbox"][1] - oy) // cell)
        self.last_map = {"cell": cell, "ox": ox, "oy": oy, "colors": colors, "player": pb,
                         "ncols": len(colors[0]), "nrows": len(colors)}
        return self.last_map

    def coarse_map(self, player: Optional[dict]) -> list[str]:
        md = self.map_data(player)
        if md is None:
            return []
        cell, ox, oy = md["cell"], md["ox"], md["oy"]
        pb = md["player"]
        pw = ph = 1
        if player:
            pw = max(1, (player["bbox"][2] - player["bbox"][0] + cell) // cell)
            ph = max(1, (player["bbox"][3] - player["bbox"][1] + cell) // cell)
        rows = []
        for r, row in enumerate(md["colors"]):
            line = []
            for c, col in enumerate(row):
                if pb and pb[0] <= c < pb[0] + pw and pb[1] <= r < pb[1] + ph:
                    line.append("P")
                else:
                    line.append(HEX[col])
            rows.append("".join(line))
        return [f"[MAP] cell={cell} origin=({ox},{oy}); char=(col,row) block majority color hex, "
                f"P=player; pixel x=col*{cell}+{ox}, y=row*{cell}+{oy}"] + \
               [f" {i:2d} {r}" for i, r in enumerate(rows)]

    def gauges(self) -> list[str]:
        out = []
        for color, hist in self.gauge_hist.items():
            if len(hist) < 3:
                continue
            steps = [b - a for a, b in zip(hist, hist[1:])]
            s = steps[-1]
            k = 0                                   # 마지막 일정-스텝 구간 길이
            while k < len(steps) and steps[-1 - k] == s:
                k += 1
            if s != 0 and k >= 2:
                left = hist[-1] // abs(s) if s < 0 else None
                out.append(f" c{color} bar {hist[-1 - k]}→{hist[-1]} cells, {s:+d} per action"
                           + (f" (~{left} actions until empty)" if left is not None else ""))
        return out

    # ── text ───────────────────────────────────────────────────────────────
    def encode(self, latest_frame: FrameData, tally: dict[str, list[int]]) -> str:
        lines = [
            f"[PROGRESS] level {latest_frame.levels_completed}/{latest_frame.win_levels}"
            f" state={latest_frame.state.name}",
            f"[AVAILABLE] {','.join(map(str, latest_frame.available_actions or []))}",
        ]
        player = self.player()
        moves = [(a, self.move_delta(a)) for a in sorted(self.move_log)]
        moves = [(a, d) for a, d in moves if d]
        if player:
            x0, y0, x1, y1 = player["bbox"]
            cs = "+".join(f"c{c}" for c in player.get("colors", [player["color"]]))
            lines.append(f"[PLAYER] {cs} s{player['size']} bbox({x0},{y0})-({x1},{y1})"
                         f" ctr({player['center'][0]},{player['center'][1]}) — your avatar")
        if moves:
            lines.append("[MOVES] " + "  ".join(f"{a}=({dx:+d},{dy:+d})" for a, (dx, dy) in moves))
        lines += self.coarse_map(player)
        if self.nav_lines:
            lines += self.nav_lines
        g = self.gauges()
        if g:
            lines.append("[GAUGE] shrinking/growing bar (likely energy/timer)")
            lines += g
        lines.append("[OBJECTS] color/size/bbox/center (largest first, background omitted)")
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
        if self.memory:
            lines.append("[MEMORY]")
            lines += [f" {m}" for m in self.memory]
        if self.prev_hypothesis:
            lines.append(f"[PREVIOUS_HYPOTHESIS] {self.prev_hypothesis}")
        return "\n".join(lines)


# ─────────────────────────────── L2 planner ─────────────────────────────────

SYSTEM_PROMPT = """You control an agent in an unknown 64x64 grid puzzle game (ARC-AGI-3).
Cells hold colors 0-15. Discover the rules by acting, then complete levels.
Score favors completing levels in FEW actions, so do not waste moves.
Observation sections:
[PLAYER]/[MOVES] which actions move your avatar and by how much. If present this is a
  navigation game: read [MAP] and plan a PATH toward likely goals (distinct objects,
  doors, keys, tiles that differ from the floor). A move that produced "no change" hit a wall.
  Do not immediately undo a move.
[MAP] coarse map, one char per block (hex color, P=your avatar). Row index = y block,
  column index = x block; convert with the formula given on the [MAP] line.
[GAUGE] a bar shrinking every action is energy/time: when empty you lose. Be efficient.
[OBJECTS] cN=color sN=size. [RECENT] effects of past actions (moved/resized/+appeared/-disappeared).
[ACTION_STATS] how often each action changed anything.
[MEMORY] what happened at previous level completions / game overs. Rules persist across
  levels, so reuse what worked. [PREVIOUS_HYPOTHESIS] is your own last guess: refine it.
Actions: 1-5 and 7 are abstract buttons (meaning differs per game; check [MOVES]/[RECENT]).
Action 6 is a click and needs x,y (0-63) — click object centers, not empty background.
Reply ONLY with JSON:
{"hypothesis": "<one sentence: what the rules/goal seem to be>",
 "plan": [{"action": <1-7>, "x": <0-63 if action 6>, "y": <0-63 if action 6>}, ...],
 "confidence": <0.0-1.0>}
Plan 3-10 actions that test your hypothesis or make progress."""

NAV_ADDENDUM = """
NAVIGATION MODE ([MOVES] present): you may also use {"action": "goto", "target": "T2"} (a target
id from [TARGETS]) or {"action": "goto", "x": X, "y": Y} (pixel): the system path-finds the
avatar there using the known moves, avoiding walls. Prefer goto over manual move sequences.
[NAV] tells which colors are floor (walkable) and which are walls. [TARGETS] lists objects
that differ from the floor (doors, keys, switches, patterned tiles) with path length in
moves. Unvisited, reachable targets are the most informative; the level goal is usually
one of them (often the one that looks like a door/exit, sometimes after touching others).
Plans may have up to 12 steps."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "plan": {"type": "array", "maxItems": 12, "items": {
            "type": "object",
            "properties": {"action": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 7},
                                                {"type": "string", "enum": ["goto"]}]},
                           "target": {"type": "string"},
                           "x": {"type": "integer", "minimum": 0, "maximum": 63},
                           "y": {"type": "integer", "minimum": 0, "maximum": 63}},
            "required": ["action"]}},
        "confidence": {"type": "number"},
    },
    "required": ["hypothesis", "plan"],
}

MAX_NEW_TOKENS = 500
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ENGINES: dict[str, Any] = {}   # 프로세스 전역 모델 캐시 (backend:path → engine)


def extract_json(text: str) -> dict:
    """<think> 블록 제거 후 첫 '{'에서 JSON 객체 하나를 디코드."""
    text = _THINK_RE.sub("", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


def find_model_path() -> Optional[str]:
    """QWEN_MODEL_PATH 또는 /kaggle/input/models 아래의 HF 체크포인트 디렉터리."""
    p = os.getenv("QWEN_MODEL_PATH")
    if p and os.path.isdir(p):
        return p
    for cfg in sorted(glob.glob("/kaggle/input/models/**/config.json", recursive=True)):
        d = os.path.dirname(cfg)
        if glob.glob(os.path.join(d, "*.safetensors")):
            return d
    return None


class QwenPlanner:
    """LLM 계획 호출. 백엔드 교체 가능, 실패 누적 시 자동 비활성화(L1 폴백)."""

    def __init__(self, model: str, endpoint: Optional[str] = None,
                 timeout: float = 60.0, max_fail: int = 3):
        self.endpoint = (endpoint or os.getenv("QWEN_ENDPOINT")
                         or "http://localhost:11434").rstrip("/")
        self.model = os.getenv("QWEN_MODEL", model)
        self.model_path = find_model_path()
        self.last_error = ""
        self.backend = self._pick_backend()
        if self.backend in ("vllm", "hf"):
            parts = [x for x in self.model_path.rstrip("/").split("/") if x and not x.isdigit()]
            self.model = (parts[-1] if parts else "model") + f"[{self.backend}]"
        self.timeout = timeout
        self.fails = 0 if self.backend != "none" else max_fail
        self.max_fail = max_fail
        self.calls = 0
        self.errors = 0
        self.total_s = 0.0
        self.load_s = 0.0
        self.last_hypothesis = ""

    def _pick_backend(self) -> str:
        forced = os.getenv("QWEN_BACKEND")
        if forced:
            return forced
        if self.model_path:
            if not self._gpu_fits():
                return "none"
            try:
                import vllm  # noqa: F401
                return "vllm"
            except Exception:
                return "hf"
        return "ollama"

    def _gpu_fits(self) -> bool:
        """체크포인트가 GPU 총 메모리의 90%를 넘으면 로드 시도 자체를 건너뜀 (T4 안전장치)."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.last_error = "no CUDA device"
                return False
            total = sum(torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count()))
            weights = sum(os.path.getsize(f) for f in glob.glob(os.path.join(self.model_path, "*.safetensors")))
            if weights > 0.9 * total:
                self.last_error = (f"model {weights / 2**30:.0f}GiB > 90% of GPU {total / 2**30:.0f}GiB; "
                                   "L2 disabled")
                return False
        except Exception as e:                                  # torch 없음 등 → 시도는 해 봄
            self.last_error = f"gpu check skipped: {type(e).__name__}"
        return True

    @property
    def enabled(self) -> bool:
        return self.fails < self.max_fail

    # ── backends ───────────────────────────────────────────────────────────
    def _messages(self, obs_text: str) -> list[dict]:
        system = SYSTEM_PROMPT + (NAV_ADDENDUM if "[MOVES]" in obs_text else "")
        return [{"role": "system", "content": system},
                {"role": "user", "content": obs_text}]

    def _complete_ollama(self, obs_text: str) -> str:
        body = json.dumps({
            "model": self.model, "stream": False, "think": False,
            "format": "json", "messages": self._messages(obs_text),
            "options": {"num_predict": MAX_NEW_TOKENS, "temperature": 0.2, "seed": 1337},
        }).encode()
        req = urllib.request.Request(f"{self.endpoint}/api/chat", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)["message"]["content"]

    def _engine(self):
        key = f"{self.backend}:{self.model_path}"
        if key not in _ENGINES:
            t0 = time.time()
            if self.backend == "vllm":
                # FlashInfer의 샘플러/GDN prefill은 첫 호출 때 nvcc JIT 빌드를 요구
                # (툴킷 없는 환경에서 실패). Triton/PyTorch 경로로 고정.
                os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
                # FP8 MoE의 DeepGEMM 경로는 nvcc>=12.9 JIT를 요구 (Kaggle 시스템 nvcc가
                # 더 낮아 엔진 초기화 실패) → Triton/CUTLASS FP8 커널로 고정
                os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
                from vllm import LLM
                _ENGINES[key] = LLM(
                    model=self.model_path, trust_remote_code=True,
                    max_model_len=int(os.getenv("QWEN_MAX_LEN", "4096")),
                    gpu_memory_utilization=float(os.getenv("QWEN_GPU_UTIL", "0.88")),
                    # 한 번에 1개 요청만 보냄. Qwen3.5(hybrid DeltaNet)는 시퀀스마다
                    # 상태 캐시 블록이 필요해 기본 256이면 소형 GPU에서 초기화 실패
                    max_num_seqs=int(os.getenv("QWEN_MAX_SEQS", "4")),
                    additional_config={"gdn_prefill_backend": os.getenv("QWEN_GDN_PREFILL", "triton")},
                    seed=1337)
            else:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(self.model_path)
                mdl = AutoModelForCausalLM.from_pretrained(
                    self.model_path, dtype="auto", device_map="auto")
                mdl.eval()
                _ENGINES[key] = (tok, mdl, torch)
            self.load_s = time.time() - t0
        return _ENGINES[key]

    def _complete_vllm(self, obs_text: str) -> str:
        from vllm import SamplingParams
        llm = self._engine()
        kw: dict = dict(temperature=0.2, max_tokens=MAX_NEW_TOKENS, seed=1337)
        try:   # vllm >= 0.10: structured_outputs / older: guided_decoding
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=PLAN_SCHEMA)
        except Exception:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kw["guided_decoding"] = GuidedDecodingParams(json=PLAN_SCHEMA)
            except Exception:
                pass
        out = llm.chat(self._messages(obs_text), SamplingParams(**kw),
                       chat_template_kwargs={"enable_thinking": False},
                       use_tqdm=False)
        return out[0].outputs[0].text

    def _complete_hf(self, obs_text: str) -> str:
        tok, mdl, torch = self._engine()
        try:
            ids = tok.apply_chat_template(self._messages(obs_text), add_generation_prompt=True,
                                          enable_thinking=False, return_tensors="pt",
                                          return_dict=True)
        except TypeError:
            ids = tok.apply_chat_template(self._messages(obs_text), add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
        ids = {k: v.to(mdl.device) for k, v in ids.items()}
        with torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def _complete(self, obs_text: str) -> str:
        if self.backend == "ollama":
            return self._complete_ollama(obs_text)
        if self.backend == "vllm":
            return self._complete_vllm(obs_text)
        if self.backend == "hf":
            return self._complete_hf(obs_text)
        raise RuntimeError(f"backend {self.backend} disabled")

    # ── public ─────────────────────────────────────────────────────────────
    def plan(self, obs_text: str) -> Optional[list[dict]]:
        if not self.enabled:
            return None
        t0 = time.time()
        try:
            data = extract_json(self._complete(obs_text))
            steps = []
            for step in (data.get("plan") or [])[:12]:
                raw = step.get("action", 0)
                if isinstance(raw, str) and raw.strip().lower() == "goto":
                    item = {"action": "goto", "target": str(step.get("target") or "").strip().upper()}
                    if "x" in step and "y" in step:
                        item["x"] = max(0, min(63, int(step.get("x", 32))))
                        item["y"] = max(0, min(63, int(step.get("y", 32))))
                    steps.append(item)
                    continue
                try:
                    a = int(raw)
                except (TypeError, ValueError):
                    continue
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
        except Exception as e:
            self.errors += 1
            self.fails += 1
            self.total_s += time.time() - t0
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return None


# ─────────────────────────────── navigator ──────────────────────────────────

class Navigator:
    """코스 맵 위 BFS 경로 탐색. 걷기 가능 = 바닥색(플레이어가 떠난 블록의 색)이고
    벽으로 확인되지 않은 블록. 목표 블록은 색과 무관하게 마지막 한 걸음 허용."""

    def __init__(self):
        self.floor_votes: Counter = Counter()
        self.walls: set[tuple[int, int]] = set()
        self.visited: set[tuple[int, int]] = set()

    @property
    def floor_colors(self) -> set[int]:
        """2회 이상 걸어본 색 (없으면 최다 1회 색) — 레벨 전환 프레임 등의 오염 방지."""
        strong = {c for c, n in self.floor_votes.items() if n >= 2}
        if strong:
            return strong
        return {self.floor_votes.most_common(1)[0][0]} if self.floor_votes else set()

    def new_level(self) -> None:
        self.walls.clear()
        self.visited.clear()

    @staticmethod
    def block_of(md: dict, x: int, y: int) -> tuple[int, int]:
        return (max(0, min(md["ncols"] - 1, (x - md["ox"]) // md["cell"])),
                max(0, min(md["nrows"] - 1, (y - md["oy"]) // md["cell"])))

    @staticmethod
    def moves(enc: "ObservationEncoder", md: dict) -> dict[str, tuple[int, int]]:
        out = {}
        for a in enc.move_log:
            d = enc.move_delta(a)
            if d and (d[0] % md["cell"] == 0 and d[1] % md["cell"] == 0):
                out[a] = (d[0] // md["cell"], d[1] // md["cell"])
        return out

    def observe(self, enc: "ObservationEncoder", md: Optional[dict],
                prev_block: Optional[tuple[int, int]], action_name: Optional[str],
                changed: bool) -> None:
        if md is None or md["player"] is None:
            return
        self.visited.add(md["player"])
        if prev_block is None or action_name is None:
            return
        mv = self.moves(enc, md).get(action_name)
        if mv is None:
            return
        # 게이지처럼 매 액션 바뀌는 요소가 있으므로 프레임 변화가 아니라
        # 플레이어 블록의 실제 이동으로 판정
        if md["player"] == prev_block:
            self.walls.add((prev_block[0] + mv[0], prev_block[1] + mv[1]))
        elif md["player"] == (prev_block[0] + mv[0], prev_block[1] + mv[1]):
            c, r = prev_block
            if 0 <= r < md["nrows"] and 0 <= c < md["ncols"]:
                self.floor_votes[md["colors"][r][c]] += 1

    def walkable(self, md: dict, b: tuple[int, int]) -> bool:
        c, r = b
        if not (0 <= r < md["nrows"] and 0 <= c < md["ncols"]) or b in self.walls:
            return False
        return md["colors"][r][c] in self.floor_colors or b == md["player"]

    def path(self, enc: "ObservationEncoder", md: dict,
             target: tuple[int, int]) -> Optional[list[int]]:
        """플레이어 블록 → target 블록 액션 id 리스트. 도달 불가면 None, 이미 도착이면 []."""
        start = md["player"]
        if start is None:
            return None
        moves = self.moves(enc, md)
        if not moves:
            return None
        if start == target:
            return []
        if target in self.walls:                # 벽으로 확인된 목표: 마지막 한 걸음도 불가
            return None
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {start: (start, "")}
        q = deque([start])
        while q:
            cur = q.popleft()
            for a, (dc, dr) in moves.items():
                nb = (cur[0] + dc, cur[1] + dr)
                if nb in prev:
                    continue
                if nb == target or self.walkable(md, nb):
                    prev[nb] = (cur, a)
                    if nb == target:
                        q.clear()
                        break
                    q.append(nb)
        if target not in prev:
            return None
        acts, b = [], target
        while b != start:
            b, a = prev[b]
            acts.append(GameAction[a].value)
        return acts[::-1]

    def targets(self, enc: "ObservationEncoder", md: dict, background: int,
                max_n: int = 8) -> list[dict]:
        """바닥/배경이 아닌 작은 객체들 → 블록 좌표 + 경로 길이 (가까운 순)."""
        if md["player"] is None:
            return []
        out, seen_blocks = [], set()
        pl = enc.player()
        pb = pl["bbox"] if pl else None
        for o in enc.prev_objects:
            if o["size"] > 120 or o["color"] == background or o["color"] in self.floor_colors:
                continue
            x0, y0, x1, y1 = o["bbox"]
            if pb and not (x0 > pb[2] or x1 < pb[0] or y0 > pb[3] or y1 < pb[1]):
                continue                                    # 아바타 자신의 조각
            b = self.block_of(md, o["center"][0], o["center"][1])
            if b in seen_blocks or b == md["player"]:
                continue
            seen_blocks.add(b)
            path = self.path(enc, md, b)
            out.append({"obj": o, "block": b, "len": None if path is None else len(path),
                        "visited": b in self.visited or b in self.walls})
        out.sort(key=lambda t: (t["len"] is None, t["len"] or 0))
        return out[:max_n]

    def frontier(self, enc: "ObservationEncoder", md: dict) -> Optional[tuple[int, int]]:
        """가장 가까운 미방문 걷기가능 블록, 없으면 도달 영역에 인접한 미지의 블록."""
        start = md["player"]
        moves = self.moves(enc, md)
        if start is None or not moves:
            return None
        seen, q, unknown = {start}, deque([start]), []
        while q:
            cur = q.popleft()
            for dc, dr in moves.values():
                nb = (cur[0] + dc, cur[1] + dr)
                if nb in seen:
                    continue
                seen.add(nb)
                if self.walkable(md, nb):
                    if nb not in self.visited:
                        return nb
                    q.append(nb)
                elif nb not in self.walls and 0 <= nb[1] < md["nrows"] and 0 <= nb[0] < md["ncols"]:
                    unknown.append(nb)
        return unknown[0] if unknown else None


# ─────────────────────────────── click sweep ────────────────────────────────

class ClickSweeper:
    """클릭 게임 L1. 단계: (A) 작은 객체부터 중심을 한 번씩 → (B) 실제 변화를 낸
    버킷을 상한(MAX_RETRY)까지 재클릭 → (C) 아직 안 눌러본 4px 버킷 격자 스위프.
    같은 버킷 연속 2회 이상 금지. 게이지/HUD 띠는 제외. 레벨 바뀌면 초기화."""

    MAX_RETRY = 6
    MAX_SWEEP = 24      # 라운드당 격자 스위프 상한, 소진되면 새 라운드(기록 초기화)

    def __init__(self):
        self.log: dict[tuple[int, int], list[int]] = {}   # bucket → [tried, changed]
        self.last: Optional[tuple[int, int]] = None
        self.last_n = 0
        self.sweeps = 0
        self.prev_sigs: set[tuple[int, int, int]] = set()

    @staticmethod
    def bucket(x: int, y: int) -> tuple[int, int]:
        return (x // 4, y // 4)

    def new_level(self) -> None:
        self.log.clear()
        self.last = None
        self.last_n = 0
        self.sweeps = 0
        self.prev_sigs.clear()

    def record(self, changed: bool) -> None:
        if self.last is None:
            return
        t = self.log.setdefault(self.last, [0, 0])
        t[0] += 1
        t[1] += int(changed)

    @staticmethod
    def is_gauge_like(o: dict, gauge_colors: set[int]) -> bool:
        """액션 게이지/HUD: 게이지 색이거나, 화면 가장자리에 붙은 길고 얇은 띠."""
        x0, y0, x1, y1 = o["bbox"]
        w, h = x1 - x0 + 1, y1 - y0 + 1
        thin_edge = ((h <= 3 and w >= 24 and (y0 <= 1 or y1 >= 62)) or
                     (w <= 3 and h >= 24 and (x0 <= 1 or x1 >= 62)))
        return o["color"] in gauge_colors or thin_edge

    def _pick(self, x: int, y: int) -> tuple[int, int]:
        b = self.bucket(x, y)
        self.last_n = self.last_n + 1 if b == self.last else 1
        self.last = b
        return (x, y)

    def choose(self, objects: list[dict], background: int,
               gauge_colors: set[int] = frozenset(), grid: Optional[list] = None) -> tuple[int, int]:
        cands = [o for o in objects if o["color"] != background
                 and not self.is_gauge_like(o, gauge_colors)]
        sigs = {(o["color"], o["size"], self.bucket(*o["center"])) for o in cands}
        appeared = sigs - self.prev_sigs if self.prev_sigs else set()
        self.prev_sigs = sigs

        def tries(o):
            return self.log.get(self.bucket(*o["center"]), [0, 0])
        def not_repeat(o):
            return not (self.bucket(*o["center"]) == self.last and self.last_n >= 2)

        # (A) 새로 나타난 객체, 그다음 안 눌러본 객체 (작은 것부터)
        fresh = [o for o in cands if tries(o)[0] == 0 and not_repeat(o)]
        if fresh:
            best = min(fresh, key=lambda o: (0 if (o["color"], o["size"], self.bucket(*o["center"])) in appeared else 1,
                                             o["size"], random.random()))
            return self._pick(*best["center"])
        # (B) 실제 변화를 낸 버킷 재클릭 (상한까지)
        retry = [o for o in cands if 0 < tries(o)[1] and tries(o)[0] < self.MAX_RETRY and not_repeat(o)]
        if retry:
            best = max(retry, key=lambda o: (tries(o)[1] / tries(o)[0], -o["size"], random.random()))
            return self._pick(*best["center"])
        # (C) 안 눌러본 버킷 격자 스위프 (배경이 아닌 칸 우선, 라운드당 상한)
        unvisited = [(bx, by) for bx in range(16) for by in range(16) if (bx, by) not in self.log]
        if unvisited and self.sweeps < self.MAX_SWEEP:
            if grid is not None:
                nonbg = [b for b in unvisited if grid[b[1] * 4 + 2][b[0] * 4 + 2] != background]
                unvisited = nonbg or unvisited
            bx, by = random.choice(unvisited)
            self.sweeps += 1
            return self._pick(bx * 4 + 2, by * 4 + 2)
        # (D) 라운드 소진 → 기록 초기화하고 새 라운드 (효과 있던 객체부터 다시)
        self.log.clear()
        self.sweeps = 0
        pool = cands or objects
        if pool:
            best = min(pool, key=lambda o: (o["size"], random.random()))
            return self._pick(*best["center"])
        return self._pick(random.randint(0, 63), random.randint(0, 63))


# ─────────────────────────────── the agent ──────────────────────────────────

class MyAgent(Agent):
    MAX_ACTIONS = 400
    PLAN_EVERY = 40
    STUCK_N = 15
    MAX_LLM_CALLS = 16
    MODEL = "qwen3.5:35b"

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
        self.nav = Navigator()
        self.clicker = ClickSweeper()
        self.sweep_clicks = 0
        self.plan_queue: list[dict] = []
        self.goto_target: Optional[tuple[int, int]] = None
        self.plan_noop_streak = 0
        self.last_plan_at = 0
        self.stuck = 0
        self.obs_samples: list[tuple[int, str]] = []
        self.obs_chars: list[int] = []
        self.planned_executed = 0
        self.goto_expanded = 0
        self.frontier_moves = 0
        self._seen_states: set = set()
        self.history: list[str] = []
        self.level_start = 0
        self.level_changed = False
        self.prev_block: Optional[tuple[int, int]] = None
        self.last_changed = True
        self.targets: list[dict] = []

    @property
    def metrics(self) -> dict:
        return {
            "unique_states": len(self.effects),
            "avg_obs_chars": (sum(self.obs_chars) // len(self.obs_chars)
                              if self.obs_chars else None),
            "llm_backend": self.planner.backend,
            "llm_model": self.planner.model,
            "llm_calls": self.planner.calls,
            "llm_errors": self.planner.errors,
            "llm_last_error": self.planner.last_error or None,
            "llm_time_s": round(self.planner.total_s, 1),
            "llm_load_s": round(self.planner.load_s, 1),
            "planned_actions": self.planned_executed,
            "goto_expanded": self.goto_expanded,
            "frontier_moves": self.frontier_moves,
            "sweep_clicks": self.sweep_clicks,
            "moves_known": {a: self.encoder.move_delta(a) for a in sorted(self.encoder.move_log)
                            if self.encoder.move_delta(a)},
            "floor_colors": sorted(self.nav.floor_colors),
            "walls_found": len(self.nav.walls),
            "memory": list(self.encoder.memory),
        }

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ── bookkeeping after each action ──────────────────────────────────────
    def _record_effect(self, latest_frame: FrameData) -> None:
        if self.prev_key is None or self.prev_action is None:
            self.last_changed = True
            return
        key = frame_key(latest_frame)
        leveled = latest_frame.levels_completed > self.prev_level
        changed = (key != self.prev_key) or leveled
        self.last_changed = changed
        self.effects.setdefault(self.prev_key, {})[self.prev_action] = changed
        if self.prev_action == "ACTION6":
            self.clicker.record(self._effective_change(latest_frame))
        t = self.tally.setdefault(self.prev_action, [0, 0])
        t[0] += int(changed)
        t[1] += 1
        if self.plan_queue or self.plan_noop_streak:
            self.plan_noop_streak = 0 if changed else self.plan_noop_streak + 1
            if self.plan_noop_streak >= 3:
                self.plan_queue.clear()
                self.goto_target = None
                self.plan_noop_streak = 0
        if key not in self._seen_states:
            self._seen_states.add(key)
            self.stuck = 0
        else:
            self.stuck += 1
        if leveled:
            self.stuck = 0
            self.plan_queue.clear()
            self.goto_target = None
            self.level_changed = True
            self.nav.new_level()
            self.clicker.new_level()
            self.encoder.gauge_hist.clear()     # 새 레벨의 게이지를 '리필'로 오인하지 않게
            used = self.action_counter - self.level_start
            tail = ",".join(self.history[-15:])
            where = ""
            if self.encoder.last_map and self.encoder.last_map["player"]:
                c, r = self.encoder.last_map["player"]
                where = (f" avatar was at block (col {c},row {r}) = pixel "
                         f"({c * self.encoder.last_map['cell'] + self.encoder.last_map['ox']},"
                         f"{r * self.encoder.last_map['cell'] + self.encoder.last_map['oy']});")
            self.encoder.remember(
                f"level {latest_frame.levels_completed} completed after {used} actions;{where}"
                f" last actions: {tail}; hypothesis then: {self.planner.last_hypothesis[:120]}")
            self.level_start = self.action_counter
        self.prev_level = latest_frame.levels_completed

    def _gauge_colors(self) -> set[int]:
        return set(self.encoder.gauge_hist)

    def _effective_change(self, latest_frame: FrameData) -> bool:
        """게이지(매 액션 변하는 HUD)를 뺀 실제 변화가 있었는가."""
        prev, cur = self.encoder.prev_grid, latest_frame.frame[-1] if latest_frame.frame else None
        if prev is None or cur is None:
            return True
        masks = [o["bbox"] for o in self.encoder.prev_objects
                 if self.clicker.is_gauge_like(o, self._gauge_colors())]
        for y in range(64):
            pr, cr = prev[y], cur[y]
            if pr == cr:
                continue
            for x in range(64):
                if pr[x] != cr[x] and not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in masks):
                    return True
        return False

    def _should_plan(self, latest_frame: FrameData) -> bool:
        llm_steps = [x for x in self.plan_queue if x.get("via") != "auto"]
        if llm_steps or not self.planner.enabled:
            return False
        if self.planner.calls >= self.MAX_LLM_CALLS:
            return False
        if self.level_changed:
            self.level_changed = False
            md0 = self.encoder.last_map
            if not (md0 and self.nav.moves(self.encoder, md0)):
                return True                  # 클릭 게임: 새 레벨에 즉시 계획
        since = self.action_counter - self.last_plan_at
        md = self.encoder.last_map
        nav_mode = bool(md and self.nav.moves(self.encoder, md))
        if nav_mode and self._nav_busy(md):
            return False                     # 내비게이터가 할 일이 남아 있으면 LLM 보류
        if since < (4 if nav_mode else 8):
            return False
        # v010: 주기 호출 없음 — 내비게이터/클릭 스위프가 막혔을 때(stuck)만 L2
        return self.stuck >= self.STUCK_N

    def _nav_busy(self, md: dict) -> bool:
        """진행 중인 goto, 미방문·도달가능 타깃, 프런티어가 있으면 True."""
        if self.goto_target is not None or any(x.get("via") == "auto" for x in self.plan_queue):
            return True
        if any(t["len"] and not t["visited"] for t in self.targets):
            return True
        return md["player"] is not None and self.nav.frontier(self.encoder, md) is not None

    # ── plan execution helpers ─────────────────────────────────────────────
    def _expand_goto(self, step: dict) -> bool:
        """goto 스텝을 경로 액션들로 큐 앞에 삽입. 실패 시 False."""
        md = self.encoder.last_map
        if not md or md["player"] is None:
            return False
        target = None
        tid = step.get("target") or ""
        if tid.startswith("T") and tid[1:].isdigit() and 0 < int(tid[1:]) <= len(self.targets):
            target = self.targets[int(tid[1:]) - 1]["block"]
        elif "x" in step:
            target = self.nav.block_of(md, step["x"], step["y"])
        if target is None:
            return False
        path = self.nav.path(self.encoder, md, target)
        if path is None:
            return False
        self.goto_target = target if path else None
        self.plan_queue[:0] = [{"action": a, "via": "goto"} for a in path[:24]]
        self.goto_expanded += 1
        return True

    def _repath_after_wall(self) -> None:
        """goto 실행 중 벽에 막히면 같은 목표로 재탐색."""
        md = self.encoder.last_map
        if self.goto_target is None or not md or md["player"] is None:
            return
        # 남은 goto 스텝 제거 후 재삽입
        via = "auto" if any(x.get("via") == "auto" for x in self.plan_queue) else "goto"
        self.plan_queue = [x for x in self.plan_queue if x.get("via") not in ("goto", "auto")]
        path = self.nav.path(self.encoder, md, self.goto_target)
        if path:
            self.plan_queue[:0] = [{"action": a, "via": via} for a in path[:24]]
        else:
            self.goto_target = None

    def choose_action(self, frames: list[FrameData],
                      latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self.encoder.remember(
                    f"GAME OVER at action {self.action_counter} (level {latest_frame.levels_completed});"
                    f" last actions: {','.join(self.history[-12:])}")
            self.prev_key = self.prev_action = None
            self.prev_block = None
            self.encoder.prev_grid, self.encoder.prev_objects = None, []
            self.encoder.recent.clear()
            self.plan_queue.clear()
            self.goto_target = None
            self.history.append("RESET")
            return GameAction.RESET

        self._record_effect(latest_frame)
        self.encoder.observe(self.prev_action, self.action_counter, latest_frame)
        player = self.encoder.player()
        md = self.encoder.map_data(player)
        if not self.level_changed:              # 레벨 전환 프레임은 벽/바닥 학습에서 제외
            self.nav.observe(self.encoder, md, self.prev_block, self.prev_action, self.last_changed)
        elif md and md["player"] is not None:
            self.nav.visited.add(md["player"])
        self.targets = []
        self.encoder.nav_lines = []
        if md and md["player"] is not None and self.nav.moves(self.encoder, md):
            grid = latest_frame.frame[-1]
            background = Counter(c for row in grid for c in row).most_common(1)[0][0]
            self.targets = self.nav.targets(self.encoder, md, background)
            nl = [f"[NAV] avatar block (col {md['player'][0]},row {md['player'][1]}); "
                  f"floor colors: {sorted(self.nav.floor_colors) or '?'}; background c{background}; "
                  f"walls found: {len(self.nav.walls)}"]
            if self.targets:
                nl.append("[TARGETS] id color size pixel block path-length")
                for i, t in enumerate(self.targets, 1):
                    o = t["obj"]
                    nl.append(f" T{i} c{o['color']} s{o['size']} @({o['center'][0]},{o['center'][1]})"
                              f" block({t['block'][0]},{t['block'][1]})"
                              + (f" {t['len']} moves" if t["len"] is not None else " unreachable")
                              + (" visited" if t["visited"] else ""))
            self.encoder.nav_lines = nl
        if md and self.prev_action and self.goto_target is not None \
                and self.prev_block is not None and md["player"] == self.prev_block \
                and self.nav.moves(self.encoder, md).get(self.prev_action):
            self._repath_after_wall()

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
                self.goto_target = None
            if self.planner.last_hypothesis:
                self.encoder.prev_hypothesis = self.planner.last_hypothesis

        action = None
        while self.plan_queue and action is None:
            step = self.plan_queue.pop(0)
            if step["action"] == "goto":
                self._expand_goto(step)      # 실패하면 그냥 다음 스텝
                continue
            ga = GameAction.from_id(step["action"])
            if ga.value in (latest_frame.available_actions or [ga.value]):
                action = ga
                if action.is_complex():
                    action.set_data({"x": step.get("x", 32), "y": step.get("y", 32)})
                self.planned_executed += 1

        if action is None:
            # 아직 한 번도 안 눌러본 단순 액션이 있으면 먼저 (이동 방향 학습 비용 1액션)
            never = [a for a in candidates if not a.is_complex()
                     and self.tally.get(a.name, [0, 0])[1] == 0]
            if never:
                action = never[0]

        known_dirs = {self.nav.moves(self.encoder, md).get(a) for a in self.encoder.move_log} - {None} if md else set()
        if action is None and md and md["player"] is not None and len(known_dirs) >= 2:
            # 내비게이션 모드 L1: 프런티어 탐색 (미방문 바닥 블록으로 BFS)
            # 1) 미방문·도달가능 타깃(가까운 순) 2) 프런티어 블록
            fr, path = None, None
            for t in self.targets:
                if not t["visited"] and t["len"]:
                    fr = t["block"]
                    break
            if fr is None:
                fr = self.nav.frontier(self.encoder, md)
            path = self.nav.path(self.encoder, md, fr) if fr else None
            if path:
                self.goto_target = fr
                self.plan_queue = [{"action": a, "via": "auto"} for a in path[1:24]]
                action = GameAction.from_id(path[0])
                self.frontier_moves += 1

        if action is None:
            tried = self.effects.get(key, {})
            untried = [a for a in candidates if a.name not in tried]
            six = GameAction.ACTION6
            if six in candidates and not known_dirs:
                # 클릭 게임: 단순 액션이 모두 무효(≥3회 시도, 변화 0)면 클릭만 사용
                dead = all(self.tally.get(a.name, [0, 0])[1] >= 3 and self.tally.get(a.name, [0, 0])[0] == 0
                           for a in candidates if a is not six)
                if dead:
                    untried = []
                    candidates = [six]
            if untried:
                action = random.choice(untried)
            else:
                effective = [a for a in candidates if tried.get(a.name, True)]
                pool = effective or candidates
                weights = [(self.tally.get(a.name, [0, 0])[0] + 1)
                           / (self.tally.get(a.name, [0, 0])[1] + 2) for a in pool]
                last_d = self.encoder.move_delta(self.prev_action) if self.prev_action else None
                if last_d:
                    for i, a in enumerate(pool):
                        d = self.encoder.move_delta(a.name)
                        if d and d[0] == -last_d[0] and d[1] == -last_d[1]:
                            weights[i] *= 0.2
                action = random.choices(pool, weights=weights, k=1)[0]
            if action.is_complex():
                grid = latest_frame.frame[-1]
                background = Counter(c for row in grid for c in row).most_common(1)[0][0]
                x, y = self.clicker.choose(self.encoder.prev_objects, background,
                                           self._gauge_colors(), grid)
                action.set_data({"x": x, "y": y})
                self.sweep_clicks += 1

        action.reasoning = (self.planner.last_hypothesis or "v010 explorer")[:120]
        self.history.append(str(action.value) if not action.is_complex()
                            else f"6@{action.action_data.x},{action.action_data.y}")
        self.prev_key = key
        self.prev_action = action.name
        self.prev_block = md["player"] if md else None
        return action

    def _candidate_actions(self, latest_frame: FrameData) -> list[GameAction]:
        allowed = set(latest_frame.available_actions or range(1, 8))
        acts = [a for a in SIMPLE_ACTIONS if a.value in allowed]
        if 6 in allowed:
            acts.append(GameAction.ACTION6)
        return acts or SIMPLE_ACTIONS
