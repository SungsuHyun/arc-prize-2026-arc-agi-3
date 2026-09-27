"""Coder mode: the model writes Python against the rulebook library and the harness runs it in-process with a time limit.

Namespace the model's code sees (rebuilt every turn):
  board, ascii, level, valid_actions, objects, regions, entities, transitions, book, goal, candidates, plans, notes
  act(a), click(r, c), press(key), path_to(r, c), predict(a), scene(), check_rule(fn), define_goal(name, fn), note(...),
  refute(id, why), set_plan(text), remember(text)
Every action that act() executes is verified against the predictor exactly like a chosen candidate; the level end, a game
over and the per-turn action cap unwind the model's code with an exception the harness catches.
"""
from __future__ import annotations

import io
import sys
import time
import traceback
from contextlib import redirect_stdout
from typing import Optional

from arcnav.frame import Frame, masked_ascii

from .entities import build_scene, scene_text
from .env import action_label

ACTION_CAP = 30       # actions per python call
TIME_LIMIT = 25.0     # seconds of model code per call (game steps included)


class StopTurn(Exception):
    pass


class Repl:
    def __init__(self, agent):
        self.a = agent
        self.notes = ""
        self.turn_actions = 0
        self.turn_status = "ok"
        self.custom_goals: dict = {}

    # ── namespace ─────────────────────────────────────────────────────────
    def _objects(self, sc) -> list:
        return [{"id": o.id, "color": o.color, "size": o.size, "bbox": o.bbox, "center": o.center, "region": o.region, "hud": o.hud,
                 "cells": o.cells} for o in sc.objs]

    def namespace(self, ev, cands, plans, goal, level0, reviews0) -> dict:
        g = self.a.g
        sc = ev.scene
        self.turn_actions = 0; self.turn_status = "ok"

        def act(x):
            acts = x if isinstance(x, list) else [x]
            out = []
            for a in acts:
                if isinstance(a, (tuple, list)) and len(a) == 2:
                    a = {"action": "MOUSE", "row": int(a[0]), "col": int(a[1])}
                if isinstance(a, str):
                    a = a.upper()
                if self.turn_actions >= ACTION_CAP:
                    raise StopTurn(f"action cap ({ACTION_CAP} per call) reached; look at the board and call again")
                self.turn_actions += 1
                st = self.a._step(a, "code", level0, reviews0, kind="code", review_mismatch=False)
                pred, v, res = self.a.last_verdict if self.a.last_verdict else (None, None, {})
                out.append({"action": action_label(a), "status": st, "predicted": pred.text if pred else "", "actual": v.text if v else "",
                            "changed": bool(res.get("changed"))})
                print(f"  {action_label(a)} -> {st}: {v.text[:120] if v else ''}")
                if st in ("won", "level"):
                    self.turn_status = st; raise StopTurn("LEVEL COMPLETED — the board is new now")
                if st == "over":
                    self.turn_status = st; raise StopTurn("GAME OVER — the level was reset")
                if st == "invalid":
                    raise StopTurn(f"{action_label(a)} is not a valid action now ({g.valid_actions})")
            return out

        def click(r, c):
            return act({"action": "MOUSE", "row": int(r), "col": int(c)})

        def press(key, n=1):
            return act([str(key).upper()] * int(n))

        def path_to(r, c):
            nav = self.a.evidence().nav
            return nav.path_to(int(r), int(c), ignore_wall_target=True) if nav else None

        def predict(a):
            if isinstance(a, (tuple, list)) and len(a) == 2:
                a = {"action": "MOUSE", "row": int(a[0]), "col": int(a[1])}
            p = self.a.evidence().predict(a if isinstance(a, dict) else str(a).upper())
            return f"{p.kind}: {p.text}"

        def scene():
            e = self.a.evidence(); return {"board": e.frame.grid, "ascii": e.frame.ascii, "objects": self._objects(e.scene), "regions": [(r.id, r.color, r.bbox) for r in e.scene.regions]}

        def check_rule(fn, name: Optional[str] = None):
            """fn(before_board, action) -> after_board or None. Scored on every recorded transition of this level; >= 0.8 on >= 4
            transitions makes it a verified rule the predictor uses first."""
            trans = g.level_transitions(); n = ok = 0; examples = []
            for t in trans:
                try:
                    out = fn([row[:] for row in t.before_frame.grid], t.action)
                except Exception as e:
                    examples.append(f"{action_label(t.action)}: raised {type(e).__name__}: {e}"); continue
                if out is None:
                    continue
                n += 1
                good = masked_ascii(Frame(out)) == masked_ascii(t.after_frame)
                ok += good
                if not good and len(examples) < 3:
                    diff = sum(1 for i in range(64) for j in range(64) if out[i][j] != t.after_frame.grid[i][j])
                    examples.append(f"{action_label(t.action)}: {diff} cells wrong")
            nm = name or getattr(fn, "__name__", "rule")
            acc = ok / n if n else 0.0
            verdict = "VERIFIED" if n >= 4 and acc >= 0.8 else "not verified"
            if verdict == "VERIFIED":
                self.a.custom_rules = [r for r in self.a.custom_rules if r[0] != nm] + [(nm, fn, acc, n)]
                self.a.book.add("rules", f"model rule `{nm}` predicts the board after an action; verified by replay ({ok}/{n} transitions)", kind="other", level=g.level, status="confirmed", source="llm")
                self.a.save_book()
            msg = f"check_rule({nm}): applied to {n} of {len(trans)} recorded transitions, {ok} exact -> {verdict}" + ("; " + "; ".join(examples) if examples else "")
            print(msg); return {"applicable": n, "correct": ok, "accuracy": acc, "verified": verdict == "VERIFIED"}

        def define_goal(name: str, fn, needs_submit: bool = False, text: str = ""):
            """fn(objects, board) -> bool: a win condition the harness evaluates on every board and refutes automatically."""
            from . import goals2
            cg = goals2.CustomGoal(str(name)[:40], fn, bool(needs_submit))
            e = next((x for x in self.a.book.section("win") if x.text.startswith(f"GOAL {name}:")), None)
            if e is None:
                e = self.a.book.add("win", f"GOAL {name}: {text or getattr(fn, '__doc__', '') or 'model-defined predicate'}", kind="other", level=g.level)
            else:
                e.status = "hypothesis"; e.note = ""
            self.a.preds[e.id] = cg; self.a.save_book()
            try:
                ok, txt, _ = cg.evaluate(self.a.evidence().scene, self.a.evidence())
            except Exception as ex:
                ok, txt = False, f"raised {type(ex).__name__}: {ex}"
            print(f"define_goal({name}) -> {e.id}; now: {'HOLDS' if ok else 'not yet'} ({txt})"); return e.id

        def note(text, section="rules", kind="other", params=None):
            e = self.a.book.add(section, str(text), kind=kind or "other", params=params or {}, level=g.level); self.a.save_book()
            print(f"noted {e.id}"); return e.id

        def refute(id_, why=""):
            r = self.a.book.refute(str(id_), str(why) or "refuted by the model"); self.a.save_book(); print(r or f"{id_}: nothing to refute"); return r

        def set_plan(text):
            self.a.book.plan = str(text)[:600]; self.a.save_book(); print("plan updated")

        def remember(text):
            self.notes = (self.notes + "\n" + str(text)).strip()[-1500:]; print("remembered")

        ns = {
            "board": ev.frame.grid, "ascii": ev.frame.ascii, "level": g.level, "valid_actions": list(g.valid_actions),
            "objects": self._objects(sc), "regions": [(r.id, r.color, r.bbox) for r in sc.regions], "entities": scene_text(sc),
            "transitions": [{"action": action_label(t.action), "raw_action": t.action, "before": t.before_frame.grid, "after": t.after_frame.grid, "level": t.level}
                            for t in g.transitions[-60:]],
            "book": self.a.book.render(), "goal": list(goal.get("lines", [])) if goal else [],
            "candidates": [(c.label, c.prediction) for c in cands if c.kind != "info"], "plans": [(c.label, c.prediction, c.actions) for c in plans if c.kind == "plan"],
            "plan_actions": {c.label: c.actions for c in plans if c.kind == "plan"}, "notes": self.notes,
            "act": act, "click": click, "press": press, "path_to": path_to, "predict": predict, "scene": scene, "check_rule": check_rule,
            "define_goal": define_goal, "note": note, "refute": refute, "set_plan": set_plan, "remember": remember,
            "masked": lambda grid: masked_ascii(Frame(grid)), "Counter": __import__("collections").Counter,
        }
        return ns

    # ── execution ─────────────────────────────────────────────────────────
    def run(self, code: str, ns: dict, *, time_limit: float = TIME_LIMIT) -> str:
        buf = io.StringIO(); t0 = time.time()

        def tracer(frame, event, arg):
            if time.time() - t0 > time_limit:
                raise StopTurn(f"time limit ({time_limit:.0f}s) reached")
            return tracer
        old = sys.gettrace()
        sys.settrace(tracer)
        try:
            with redirect_stdout(buf):
                try:
                    exec(compile(code, "<model>", "exec"), ns)
                except StopTurn as e:
                    print(f"[stopped: {e}]")
                except Exception:
                    tb = traceback.format_exc(limit=3)
                    print("[error]\n" + tb[-900:])
        finally:
            sys.settrace(old)
        out = buf.getvalue()
        return out[-4000:] if len(out) > 4000 else out
