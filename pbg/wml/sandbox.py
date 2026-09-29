"""wml/sandbox.py — loading LLM-written model / goal code safely (spec §8).

* static checks: game-id string comparisons, >= 20 hard-coded coordinate tuples, eval/exec, disallowed imports
* validation run in a subprocess with resource limits (CPU 2 s, 256 MB, no network/file access — no socket/open in
  the namespace) that builds the model and evaluates it on the log
* after validation the same code is loaded in-process (restricted builtins) so the planner can call predict() cheaply"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

ALLOWED_IMPORTS = {"numpy", "dataclasses", "typing", "itertools", "math", "core", "core.types", "core.contracts", "core.mechanisms",
                   "pbg.core", "pbg.core.types", "pbg.core.contracts", "pbg.memory.priors.mechanisms", "collections", "functools"}
GAME_ID_RE = re.compile(r"^[a-z]{2}\d{2}$")
COORD_RE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")
ROOT = Path(__file__).resolve().parents[2]


class StaticCheckError(ValueError):
    pass


def static_check(code: str, allow_getattr: bool = False) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise StaticCheckError(f"syntax error: {e}")
    for node in ast.walk(tree):
        if allow_getattr and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr"):
            names = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if any(n.startswith("__") for n in names):
                raise StaticCheckError("dunder access not allowed: getattr")
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                root = n.split(".")[0]
                if n not in ALLOWED_IMPORTS and root not in {"numpy", "dataclasses", "typing", "itertools", "math", "core", "collections", "functools"}:
                    raise StaticCheckError(f"import not allowed: {n}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile", "open", "__import__", "getattr", "globals", "locals"):
            raise StaticCheckError(f"call not allowed: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__name__",):
            raise StaticCheckError(f"dunder access not allowed: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and GAME_ID_RE.match(node.value):
            raise StaticCheckError(f"game id literal '{node.value}' (no-game-id-branch)")
    n_coords = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Tuple) and len(node.elts) == 2
                   and all(isinstance(e, ast.Constant) and isinstance(e.value, int) for e in node.elts))
    if n_coords >= 20:
        raise StaticCheckError("too many hard-coded coordinate tuples in code (>= 20)")
    if "def build_model" not in code and "def build_goal" not in code:
        raise StaticCheckError("code must define build_model() or build_goal()")


SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "frozenset", "int", "isinstance", "len", "list", "map", "max", "min",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "print", "Exception", "ValueError", "KeyError", "IndexError",
    "TypeError", "None", "True", "False", "hasattr", "setattr", "object", "staticmethod", "property", "type", "repr", "divmod", "pow", "iter", "next")}


def _namespace() -> dict:
    import numpy as np
    from ..core import contracts, types
    from ..memory.priors import mechanisms
    ns: dict = {"__builtins__": dict(SAFE_BUILTINS), "np": np, "numpy": np}
    for mod in (types, contracts, mechanisms):
        for k in getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]:
            ns[k] = getattr(mod, k)
    ns["Rule"], ns["RuleModel"], ns["Scene"], ns["Action"], ns["Object"] = contracts.Rule, contracts.RuleModel, types.Scene, types.Action, types.Object
    ns["Optional"] = Optional
    import dataclasses, itertools, math, typing, collections, functools
    ns.update({"dataclasses": dataclasses, "itertools": itertools, "math": math, "typing": typing, "collections": collections, "functools": functools,
               "dataclass": dataclasses.dataclass, "field": dataclasses.field})
    real_import = __import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("core", "core.types", "core.contracts", "core.mechanisms"):
            return {"core": _CoreShim(types, contracts, mechanisms), "core.types": types, "core.contracts": contracts, "core.mechanisms": mechanisms}[name]
        if name.split(".")[0] in ("numpy", "dataclasses", "typing", "itertools", "math", "collections", "functools"):
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError(f"import not allowed in sandbox: {name}")
    ns["__builtins__"]["__import__"] = safe_import
    return ns


class _CoreShim:
    def __init__(self, types, contracts, mechanisms):
        self.types, self.contracts, self.mechanisms = types, contracts, mechanisms


def load_in_process(code: str, allow_getattr: bool = False):
    """Execute model code in a restricted namespace; returns the namespace (build_model / build_goal available)."""
    static_check(code, allow_getattr=allow_getattr)
    ns = _namespace()
    if allow_getattr:
        ns["__builtins__"] = dict(ns["__builtins__"]) if isinstance(ns.get("__builtins__"), dict) else ns.get("__builtins__")
        if isinstance(ns["__builtins__"], dict):
            ns["__builtins__"]["getattr"] = getattr; ns["__builtins__"]["hasattr"] = hasattr
    exec(compile(code, "<model>", "exec"), ns)   # noqa: S102 — after static checks, restricted builtins
    return ns


class Sandbox:
    def __init__(self, cpu_seconds: int = 2, memory_mb: int = 256, python: Optional[str] = None, log=None, allow_getattr: bool = False):
        self.cpu_seconds, self.memory_mb = cpu_seconds, memory_mb
        self.python = python or sys.executable
        self.log = log or (lambda *a, **k: None)
        self.allow_getattr = allow_getattr

    def load_namespace(self, code: str):
        """Validated in-process load; returns the module namespace or None (error logged)."""
        try:
            return load_in_process(code, allow_getattr=self.allow_getattr)
        except Exception as e:
            self.log(f"sandbox load failed: {type(e).__name__}: {str(e)[:200]}")
            return None

    def validate(self, code: str, log_json: list[dict]) -> dict:
        """Run static checks + build + evaluate in a resource-limited subprocess. Returns {ok, score, coverage, violations, error}."""
        try:
            static_check(code)
        except StaticCheckError as e:
            return {"ok": False, "error": f"static: {e}"}
        payload = json.dumps({"code": code, "log": log_json, "cpu": self.cpu_seconds, "mem": self.memory_mb})
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            r = subprocess.run([self.python, "-m", "pbg.wml.sandbox_worker"], input=payload, capture_output=True, text=True, timeout=self.cpu_seconds * 5 + 10, env=env, cwd=str(ROOT))
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout (wall)"}
        if r.returncode != 0:
            return {"ok": False, "error": f"worker exit {r.returncode}: {(r.stderr or r.stdout)[-800:]}"}
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return {"ok": False, "error": f"bad worker output: {r.stdout[-400:]} {r.stderr[-400:]}"}

    def load(self, code: str):
        """In-process load after validation. Returns the WorldModel or None (error logged)."""
        try:
            ns = load_in_process(code, allow_getattr=self.allow_getattr)
            if "build_model" not in ns:
                return None
            model = ns["build_model"]()
            return model
        except Exception as e:
            self.log(f"sandbox load failed: {type(e).__name__}: {e}")
            return None

    def load_goal(self, code: str):
        try:
            ns = load_in_process(code)
            if "build_goal" not in ns:
                return None
            g = ns["build_goal"]()
            from ..goal.templates import GoalInstance
            if isinstance(g, GoalInstance):
                return g
            if hasattr(g, "is_goal") and hasattr(g, "progress"):
                return GoalInstance(getattr(g, "name", "llm_goal"), "llm", {}, g.is_goal, g.progress, origin="llm", code=code)
        except Exception as e:
            self.log(f"sandbox goal load failed: {type(e).__name__}: {e}")
        return None
