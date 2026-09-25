#!/usr/bin/env python
"""D2: can the local model write a correct transition simulator predict(before_frame, action) from recorded transitions?

For each game: play level 1 with the nav template (no LLM) to record transitions, split them into shown/held-out,
ask the model (thinking on) to write predict(), score it on the held-out transitions, feed back mismatches, up to
ROUNDS rounds. Reports accuracy per round. Usage: python scripts/d2_simulator_synthesis.py ls20 m0r0 vc33 sb26
"""
import json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
from arcnav.runner import make_arcade, DEFAULT_CONFIG
from arcnav.agent import GameSession
from arcnav.llm import ChatClient
from arcnav.prompts import NAV_TEMPLATE

ROUNDS = 3
OUT = Path("/home/hss/code/kaggle/2026/arc-prize-2026-arc-agi-3/vendor/arcnav-runs/d2")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = """You are given recorded transitions of an unknown grid game: for each, the board BEFORE (64x64 ascii, hex digit per cell),
the ACTION, and the board AFTER. Write a Python function

    def predict(before_frame, action_name):
        ...
        return after_ascii   # string, 64 rows of 64 hex chars, or None if you cannot predict

`before_frame.grid` is a list of 64 lists of ints (0-15); `before_frame.ascii` is the same board as text; action_name is one of
UP, DOWN, LEFT, RIGHT, SPACE, ACTION7 or "MOUSE(row=r, col=c)". Infer the rules (what moves, what blocks it, what counter changes,
what happens when objects touch) and implement them generally, not by memorising these examples. Return ONLY the code.
"""


def collect(game, n_actions=40):
    arc = make_arcade(); env = arc.make(game)
    s = GameSession(env, game, None, log_dir=OUT / "logs", max_minutes=5, verbose=False); s.reset()
    if "MOUSE" in s.valid_actions and "UP" not in s.valid_actions:
        # click game: click each non-hud object once, smallest first
        s._run_tool("nodes=[n for n in current_frame.segmentation['nodes'] if not n['hud']]; nodes.sort(key=lambda n:n['pixels'])\n"
                    "for n in nodes[:%d]:\n    action({'action':'MOUSE','row':n['center'][0],'col':n['center'][1]})" % n_actions, who="d2")
    else:
        s._run_tool(NAV_TEMPLATE + "\nfor _ in range(%d):\n    a = solve()\n    if not a: break\n    r = action(a[:4])\n    if r['level_completed'] or r['game_over']: break\n" % n_actions, who="d2")
    tr = [(t.action if isinstance(t.action, str) else f"MOUSE(row={t.action['row']}, col={t.action['col']})", t.before_frame, t.after_frame)
          for t in s.host_transitions if t.before_frame.level == t.after_frame.level]
    s.sandbox.close()
    return tr


def render(tr, full_pairs=2, max_cells=60):
    """Compact: the first board in full, the first `full_pairs` transitions as full before/after, the rest as changed-cell lists."""
    out = [f"INITIAL BOARD (transition 0 BEFORE):\n{tr[0][1].ascii}\n"]
    for i, (a, b, af) in enumerate(tr):
        if i < full_pairs:
            out.append(f"### transition {i}\nACTION: {a}\nBEFORE:\n{b.ascii}\nAFTER:\n{af.ascii}\n")
        else:
            cells = [(r, c, b.grid[r][c], af.grid[r][c]) for r in range(64) for c in range(64) if b.grid[r][c] != af.grid[r][c]]
            desc = ", ".join(f"({r},{c}):{o}->{n}" for r, c, o, n in cells[:max_cells]) + (f" ... (+{len(cells)-max_cells} more)" if len(cells) > max_cells else "")
            out.append(f"### transition {i}\nACTION: {a}\nCHANGED CELLS (row,col):old->new: {desc or 'none'}\n")
    return "\n".join(out)


def score(code, tr):
    ns = {}
    try:
        exec(compile(code, "predict.py", "exec"), ns)
        fn = ns["predict"]
    except Exception as e:
        return 0.0, [f"code failed to load: {type(e).__name__}: {e}"]
    hits, mism = 0, []
    for i, (a, b, af) in enumerate(tr):
        try:
            p = fn(b, a)
        except Exception as e:
            mism.append(f"transition {i} ({a}): predict raised {type(e).__name__}: {e}"); continue
        if p is None:
            mism.append(f"transition {i} ({a}): returned None"); continue
        p = str(p).strip()
        if p == af.ascii.strip():
            hits += 1
        else:
            exp, got = af.ascii.splitlines(), p.splitlines()
            bad = [r for r in range(min(len(exp), len(got))) if exp[r] != got[r]][:4]
            detail = "; ".join(f"row {r}: expected {exp[r]} got {got[r] if r < len(got) else '?'}" for r in bad[:2])
            mism.append(f"transition {i} ({a}): rows differ {bad} — {detail}")
    return hits / max(1, len(tr)), mism


def main(games):
    cfg = DEFAULT_CONFIG
    client = ChatClient(cfg["base_url"], cfg["model"], temperature=0.6, top_p=0.95, max_tokens=10000,
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    results = {}
    for game in games:
        tr = collect(game)
        if len(tr) < 8:
            print(f"{game}: only {len(tr)} transitions, skipped"); continue
        shown, held = tr[: len(tr) * 2 // 3], tr[len(tr) * 2 // 3:]
        messages = [{"role": "system", "content": "You are an expert at inferring the rules of grid games and implementing exact simulators in Python."},
                    {"role": "user", "content": PROMPT + "\n" + render(shown[:14])}]
        accs = []
        for rnd in range(ROUNDS):
            t0 = time.time(); r = client.chat(messages, tools=None); code = r.message.get("content") or ""
            if "```" in code:
                code = code.split("```")[1]; code = code.split("\n", 1)[1] if code.startswith("python") else code
            acc_shown, _ = score(code, shown); acc_held, mism = score(code, held)
            accs.append((acc_shown, acc_held)); print(f"{game} round {rnd+1}: shown={acc_shown:.2f} held-out={acc_held:.2f} ({time.time()-t0:.0f}s, {r.usage.get('completion_tokens')} tok)")
            (OUT / f"{game}-round{rnd+1}.py").write_text(code)
            if acc_held >= 0.95:
                break
            messages += [{"role": "assistant", "content": code},
                         {"role": "user", "content": "Your predict() disagrees with the recorded transitions. Mismatches:\n" + "\n".join(mism[:8]) +
                          "\nFix the rules (do not special-case examples) and return the full corrected code only."}]
        results[game] = {"transitions": len(tr), "rounds": accs}
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["ls20", "m0r0", "vc33", "sb26"])
