"""System prompt and per-turn text for the arcnav agent (our wording)."""
from __future__ import annotations

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code in the persistent game sandbox. Use action(...) inside it to play. Print what you need to see.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                       "required": ["code"]},
    },
}

PROPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_solver",
        "description": "Store a persistent solver. `code` is Python source defining `def solve():` (and optionally `def predict(before_frame, action_name)`) "
                       "that reads current_frame, transitions, level_transitions, nav, valid_actions and returns the next actions (a list) or [].",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source defining def solve(): ..."}},
                       "required": ["code"]},
    },
}

SYSTEM_PROMPT = """You are an autonomous agent playing an unknown grid puzzle game (ARC-AGI-3). The board is a 64x64 grid of colour
indices 0-15, shown as ascii rows using the hex digits 0-9a-f (one character per cell, row 0 at the top). Nobody tells you the rules:
discover them by acting, then complete as many levels as possible using as FEW actions as possible (the score per level is
(baseline_actions/your_actions)^2, so wasted actions cost score; unfinished levels score 0).

You act only through the `python` tool. The sandbox is a persistent Python process with these variables and helpers:
- `current_frame`: Frame with `.ascii` (the board as text), `.grid` (list of 64 rows of ints), `.level`, `.step`,
  `.segmentation` = {'nodes': [{id, color, pixels, bbox(r0,c0,r1,c1), center(row,col), hash, children, hud}], 'adjacency': [[i,j]...], 'background'}.
  `hud` marks thin strips along an edge (usually a status bar / action counter, not a clickable object).
- `transitions`: list of Transition(action, before_frame, after_frame, changed) for every action taken so far (newest last);
  `level_transitions` is the part that belongs to the current level.
- `valid_actions`: names allowed now, among UP, DOWN, LEFT, RIGHT, SPACE and MOUSE (a click: {'action':'MOUSE','row':r,'col':c}).
- `action(x)`: execute one action or a list of actions, e.g. action('UP') or action(['LEFT','LEFT']) or action({'action':'MOUSE','row':10,'col':20}).
  It returns {'executed_count','board_changed','level_completed','game_over','stopped_reason'} and refreshes every variable above.
  Execution stops early when a level completes or the game ends, and at most 24 actions run per call. After a game over the harness resets the game for you; the level restarts.
- `nav`: navigation helper rebuilt from `transitions` on every call (None until the first frame). `nav.summary()` gives the avatar
  (the object your movement keys move), learned move deltas, floor/wall knowledge, an action gauge if the game has one, candidate targets
  with (row, col) and BFS path length, and the nearest unexplored block. `nav.path_to(row, col)` returns the list of moves to reach the block
  containing that pixel (pass it straight to action), `nav.frontier()` the nearest unexplored walkable block (row, col) or None, `nav.targets()` a list of dicts with keys
  row, col, color, size, path_len (None if unreachable), visited, last_visit,
  `nav.map()` a coarse map (P = avatar, # = wall). It learns only from real moves, so press each movement key once or twice first.
- `notes`: a string you own. Keep the rules learned so far in it (what each key does, objects and their roles, the goal
  hypothesis, the next plan). It persists and is shown to you at every turn, so update it instead of re-analysing the board.
- `propose_solver(code)`: store a persistent solver. `code` is a string defining `def solve():` that reads the variables above and returns the
  next actions (a list, or [] when undecided). Optionally define `def predict(before_frame, action_name)` returning the expected next
  board as an ascii string; it is scored against the recorded transitions and proposals with accuracy below 0.5 are rejected.
  Once stored, the harness runs solve() on its own every turn without asking you, until it returns [], raises, stops changing the
  board, or the level stops advancing; then you get a failure report, repair the code and propose again. Never define your own
  `propose_solver` or `action`.

How to work:
1. Look first: print `current_frame.ascii` (or parts of it) and `current_frame.segmentation` summaries. Identify the avatar, walls,
   collectables, doors, counters, and the HUD.
2. Probe cheaply: one action per movement key, one click per distinct object type, and compare `transitions[-1].before_frame.ascii`
   with `.after_frame.ascii` (or print `nav.summary()`). Write down what each action does.
3. Form a hypothesis about the goal (reach a target, collect items in some order, match a pattern, toggle objects), then act on it with
   short scripted sequences, not one action per turn. Use `nav.path_to` for movement instead of hand-written step lists.
4. When the rules are clear or a level was just completed, encode them in `propose_solver(code)` so the following levels are played
   programmatically and cheaply. Keep solve() deterministic and short; prefer search over hand-written sequences.
5. A thin strip along an edge whose length changes by a fixed amount every action is an action counter (gauge). It is never the
   goal: do not try to fill or empty it, and never click it. When it runs out the level restarts (game over), so plan within it.
6. Every tool call should either gain information or make progress. Do not repeat a probe whose result you already know. If the board
   stops changing, change strategy (different key, different object, the SPACE key, a different order).

Keep your written reasoning brief (a few sentences; let the code do the counting). Each reply must contain exactly one
tool call (`python`, or `propose_solver` when you are ready to store a solver), and most calls should execute several actions (a whole route or a probe batch), not a single step.
Do not restate the game analysis every turn: read `notes`, update it with what changed, act.
"""

NAV_TEMPLATE = '''def solve():
    if nav is None:
        return []
    recent = [t.action for t in level_transitions[-8:]]
    for a in ['UP', 'DOWN', 'LEFT', 'RIGHT']:   # learn every movement key once before routing
        if a in valid_actions and a not in nav.moves and a not in recent:
            return [a]
    for t in nav.targets():
        if not t['visited'] and t['path_len']:
            return nav.path_to(t['row'], t['col']) or []
    reach = [t for t in nav.targets() if t['path_len']]   # all visited: revisit the least recently visited
    if reach:
        t = min(reach, key=lambda t: t['last_visit'])
        return nav.path_to(t['row'], t['col']) or []
    fr = nav.frontier()
    return (nav.path_to(fr[0], fr[1]) or []) if fr else []
'''

CLICK_TEMPLATE = '''def solve():
    nodes = [n for n in current_frame.segmentation['nodes'] if not n['hud'] and n['pixels'] <= 600]
    nodes.sort(key=lambda n: n['pixels'])
    clicked = {}
    for t in transitions:
        a = t.action if isinstance(t.action, dict) else None
        if a and a.get('action') == 'MOUSE':
            key = (a['row'] // 4, a['col'] // 4)
            clicked.setdefault(key, [0, 0]); clicked[key][0] += 1; clicked[key][1] += int(t.changed)
    key = lambda n: (n['center'][0] // 4, n['center'][1] // 4)
    fresh = [n for n in nodes if key(n) not in clicked]
    pool = fresh or [n for n in nodes if clicked.get(key(n), [0, 0])[1] > 0] or nodes
    if not pool:
        return []
    r, c = pool[0]['center']
    return [{'action': 'MOUSE', 'row': r, 'col': c}]
'''

SOLVER_GUIDE = f"""
Solver templates (adapt the target choice to the rules you inferred):
- Navigation game (visit the nearest unvisited target, else explore):
propose_solver('''
{NAV_TEMPLATE}''')
- Click game (no avatar, MOUSE only: sweep objects, re-click the ones that changed the board, skip HUD strips):
propose_solver('''
{CLICK_TEMPLATE}''')
"""


def system_prompt() -> str:
    return SYSTEM_PROMPT + SOLVER_GUIDE


def turn_header(*, level: int, levels_total: int, actions_used: int, level_actions: int, valid_actions: list[str], budget_line: str) -> str:
    return (f"Level {level}/{levels_total}. Actions used: {actions_used} total, {level_actions} on this level. "
            f"Valid actions now: {', '.join(valid_actions) or 'none'}. {budget_line}")
