"""Python tool sandbox: runs model-written code in a separate `python -I`
process with a JSON-lines protocol. Inside the child the code sees
`current_frame`, `history`, `transitions`, `valid_actions`, `nav`,
`action(...)` (executes actions on the host and refreshes state) and
`propose_solver(code)` (stores a persistent `solve()` on the host)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

_PKG = Path(__file__).resolve().parent
_SANDBOX_SRC_FILES = ("frame.py", "nav.py")

CHILD_PROGRAM = r'''
import ast, json, os, resource, signal, sys, traceback, types
_out = os.fdopen(os.dup(1), "w"); _in = sys.stdin; _err = sys.stderr
_devnull = os.open(os.devnull, os.O_WRONLY); os.dup2(_devnull, 1)   # fd 1 is no longer the protocol channel
sys.stdout = open(os.devnull, "w")

def _send(obj):
    _out.write(json.dumps(obj) + "\n"); _out.flush()

def _recv():
    line = _in.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)

_init = _recv()
_mods = {}
for _name, _src in _init["sources"].items():
    _m = types.ModuleType("arcnav." + _name); sys.modules["arcnav." + _name] = _m
    exec(compile(_src, _name + ".py", "exec"), _m.__dict__); _mods[_name] = _m
os.dup2(_devnull, 2); sys.stderr = open(os.devnull, "w")
Frame = _mods["frame"].Frame
NavHelper = _mods["nav"].NavHelper

class Transition:
    """One executed action: `action` (name or MOUSE(row=r, col=c)), `before_frame`, `after_frame`, `changed`."""
    def __init__(self, action, before_frame, after_frame, result=None):
        self.action, self.before_frame, self.after_frame, self.result = action, before_frame, after_frame, result or {}
    @property
    def changed(self):
        return self.before_frame.ascii != self.after_frame.ascii
    def __repr__(self):
        return f"Transition({self.action}, changed={self.changed}, level={self.after_frame.level})"

G = {"__name__": "__main__", "Frame": Frame, "NavHelper": NavHelper, "Transition": Transition}

def _refresh(state):
    G["current_frame"] = Frame.from_payload(state["frame"]) if state.get("frame") else None
    G["valid_actions"] = list(state.get("valid_actions") or [])
    G["level"] = int(state.get("level", 1)); G["levels_total"] = int(state.get("levels_total", 0))
    trans = []
    for t in state.get("transitions") or []:
        trans.append(Transition(t["action"], Frame.from_payload(t["before"]), Frame.from_payload(t["after"]), t.get("result")))
    G["transitions"] = trans
    G["history"] = trans  # alias
    G["last_action_result"] = state.get("last_action_result")
    G["level_recaps"] = list(state.get("level_recaps") or [])
    if "notes" not in G or not G["notes"]:
        G["notes"] = state.get("notes") or ""
    try:
        cur = [t for t in trans if t.before_frame.level == t.after_frame.level == G["level"]]
        G["level_transitions"] = cur
        G["nav"] = NavHelper(cur, G["current_frame"]) if G["current_frame"] is not None else None
        G["nav_error"] = None
    except Exception as e:  # pragma: no cover
        G["nav"], G["nav_error"] = None, repr(e)

def _normalize(actions):
    if isinstance(actions, (str, dict)):
        actions = [actions]
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("action(actions) expects an action or a non-empty list of actions")
    out = []
    for a in actions:
        if isinstance(a, str):
            if a.strip().upper() in ("MOUSE", "CLICK", "ACTION6"):
                raise ValueError("MOUSE needs coordinates: use {'action': 'MOUSE', 'row': r, 'col': c}")
            out.append({"action": a.strip().upper()})
        elif isinstance(a, dict):
            d = {k: v for k, v in a.items()}; d["action"] = str(d.get("action", "")).strip().upper()
            if d["action"] in ("MOUSE", "CLICK", "ACTION6"):
                d["action"] = "MOUSE"
                if "row" not in d or "col" not in d:
                    raise ValueError("MOUSE needs row and col (0-63)")
                d["row"], d["col"] = int(d["row"]), int(d["col"])
            out.append(d)
        else:
            raise ValueError(f"unsupported action {a!r}")
    return out

def action(actions):
    """Execute one or more actions on the game and refresh all runtime variables.
    Returns {'executed_count', 'board_changed', 'level_completed', 'game_over', 'stopped_reason', 'results'}."""
    acts = _normalize(actions)
    _send({"type": "action", "actions": acts})
    reply = _recv()
    if reply.get("type") != "action_result":
        raise RuntimeError("protocol error")
    _refresh(reply["state"])
    return reply["result"]

def propose_solver(code, verify_last=12):
    """Store a persistent solver. `code` must define `def solve():` returning the next actions (list) or [].
    Optional `def predict(before_frame, action_name)` -> ascii string is checked against recent transitions."""
    if not isinstance(code, str):
        raise TypeError("propose_solver(code) expects a string of python code")
    report = {"ok": False}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        report["reason"] = f"syntax error: {e}"; _send({"type": "solver", "code": code, "report": report}); return _recv().get("result", report)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "solve" not in names:
        report["reason"] = "code must define `def solve():`"; _send({"type": "solver", "code": code, "report": report}); return _recv().get("result", report)
    if "propose_solver" in names or "action" in names:
        report["reason"] = "do not redefine the built-ins `propose_solver` / `action`"; _send({"type": "solver", "code": code, "report": report}); return _recv().get("result", report)
    ns = dict(G)
    try:
        exec(compile(code, "solver.py", "exec"), ns)
        dry = ns["solve"]()
        dry = list(dry or [])
        _normalize(dry) if dry else None
        report["dry_run_actions"] = [a if isinstance(a, str) else dict(a) for a in dry[:8]]
    except Exception as e:
        report["reason"] = f"solve() raised on dry run: {type(e).__name__}: {e}"
        _send({"type": "solver", "code": code, "report": report}); return _recv().get("result", report)
    if "predict" in names and G.get("transitions"):
        checked = G["transitions"][-verify_last:]; hits = 0; mism = []
        for t in checked:
            try:
                p = ns["predict"](t.before_frame, str(t.action))
            except Exception as e:
                p = None; mism.append(f"predict raised {type(e).__name__}: {e}")
            if p is None:
                continue
            if str(p).strip() == t.after_frame.ascii.strip():
                hits += 1
            else:
                exp, got = t.after_frame.ascii.splitlines(), str(p).strip().splitlines()
                bad = [i for i in range(min(len(exp), len(got))) if exp[i] != got[i]][:3]
                mism.append(f"{t.action}: rows differ at {bad}")
        report["predict_accuracy"] = hits / max(1, len(checked)); report["predict_mismatches"] = mism[:5]
        if report["predict_accuracy"] < 0.5:
            report["reason"] = f"predict() accuracy {report['predict_accuracy']:.2f} < 0.5 — fix the world model first"
            _send({"type": "solver", "code": code, "report": report}); return _recv().get("result", report)
    report["ok"] = True
    report["verified"] = bool("predict" in names and report.get("predict_accuracy", 0) >= 0.8 and len(G.get("transitions") or []) >= 6)
    exec(compile(code, "solver.py", "exec"), G)   # accepted: solve()/predict() live in the runtime namespace
    _send({"type": "solver", "code": code, "report": report})
    return _recv().get("result", report)

G["action"] = action; G["propose_solver"] = propose_solver
_refresh(_init["state"])

def _run(code):
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<tool>", "exec"), G)
        v = eval(compile(last, "<tool>", "eval"), G)
        if v is not None:
            print(repr(v), file=_cap)
    else:
        exec(compile(tree, "<tool>", "exec"), G)

import io
_cap = io.StringIO()
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
while True:
    msg = _recv()
    if msg.get("type") == "state":
        _refresh(msg["state"]); continue
    if msg.get("type") != "run":
        break
    _cap = io.StringIO(); sys.stdout = sys.stderr = _cap
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("tool timeout")))
    signal.alarm(int(msg.get("timeout", 30)))
    err = None
    try:
        _run(msg["code"])
    except SystemExit:
        pass
    except BaseException as e:
        tb = traceback.format_exception_only(type(e), e)
        err = "".join(tb).strip()
        if isinstance(e, TimeoutError):
            err = "TimeoutError: tool timeout"
    finally:
        signal.alarm(0)
    sys.stdout = sys.stderr = open(os.devnull, "w")
    _send({"type": "done", "stdout": _cap.getvalue(), "error": err, "notes": str(G.get("notes") or "")[:3000]})
'''


def _sources() -> dict[str, str]:
    return {name[:-3]: (_PKG / name).read_text() for name in _SANDBOX_SRC_FILES}


class Sandbox:
    """A persistent child interpreter for one game. `action_handler(actions) -> (result, state)`;
    `solver_handler(code, report) -> result`; `state_provider() -> state payload`."""

    def __init__(self, *, action_handler: Callable[[list[dict]], tuple[dict, dict]],
                 solver_handler: Callable[[str, dict], dict], state_provider: Callable[[], dict],
                 timeout: int = 30, max_output_chars: int = 6000):
        self.action_handler, self.solver_handler, self.state_provider = action_handler, solver_handler, state_provider
        self.timeout, self.max_output_chars = timeout, max_output_chars
        self.proc: Optional[subprocess.Popen] = None

    def _start(self) -> None:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": os.environ.get("HOME", "/tmp")}
        self.proc = subprocess.Popen([sys.executable, "-I", "-c", CHILD_PROGRAM], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, env=env, start_new_session=True)
        self._send({"sources": _sources(), "state": self.state_provider()})

    def _send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n"); self.proc.stdin.flush()

    def _recv(self) -> dict:
        assert self.proc and self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("sandbox died")
        return json.loads(line)

    def close(self) -> None:
        if self.proc:
            try:
                os.killpg(self.proc.pid, 9)
            except Exception:
                pass
            self.proc = None

    def run(self, code: str, *, timeout: Optional[int] = None) -> dict:
        """Run tool code; returns {'stdout', 'error', 'actions_executed', 'proposals': [...]}"""
        if self.proc is None or self.proc.poll() is not None:
            self.close(); self._start()
        else:
            self._send({"type": "state", "state": self.state_provider()})
        self._send({"type": "run", "code": code, "timeout": timeout or self.timeout})
        executed, proposals = 0, []
        deadline = time.time() + (timeout or self.timeout) + 15
        while True:
            try:
                msg = self._recv()
            except RuntimeError:
                self.close()
                return {"stdout": "", "error": "RuntimeError: python tool process died (memory/time limit?)", "actions_executed": executed, "proposals": proposals}
            t = msg.get("type")
            if t == "action":
                result, state = self.action_handler(msg["actions"])
                executed += int(result.get("executed_count", 0))
                self._send({"type": "action_result", "result": result, "state": state})
            elif t == "solver":
                res = self.solver_handler(msg["code"], msg["report"]); proposals.append(res)
                self._send({"type": "solver_result", "result": res})
            elif t == "done":
                out = msg.get("stdout", "")
                if len(out) > self.max_output_chars:
                    out = out[: self.max_output_chars // 2] + "\n...[truncated]...\n" + out[-self.max_output_chars // 2:]
                return {"stdout": out, "error": msg.get("error"), "actions_executed": executed, "proposals": proposals, "notes": msg.get("notes", "")}
            if time.time() > deadline:
                self.close()
                return {"stdout": "", "error": "TimeoutError: tool exceeded its time limit", "actions_executed": executed, "proposals": proposals}
