"""The rulebook: an explicit, editable list of hypotheses about the game.

Three sections — env (facts about the board/HUD/avatar), rules (what actions do), win (how a level is completed) — plus a
free-text plan. Every entry has a status: hypothesis / confirmed / refuted. Entries whose `kind` is machine-checkable carry
`params` so the harness can score them against the recorded transitions (`sync`); the others are text-only and are judged by
the model during reviews. The model edits the book through a small edit language (`apply_edits`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SECTIONS = ("env", "rules", "win")
PREFIX = {"env": "E", "rules": "R", "win": "W"}
STATUSES = ("hypothesis", "confirmed", "refuted")

# machine-checkable kinds and the params that identify one rule of that kind
KIND_KEYS = {
    "move": ("action",), "wall": ("color",), "noop": ("action",), "gauge": ("color",), "refill": ("color",),
    "collect": ("color",), "hazard": ("color",), "click": ("color", "region"), "mirror": ("color",),
    "button": ("color", "region", "bbox"), "coupled": ("marker", "color"),
    # win kinds (from goals.infer)
    "reach": ("reach",), "collect_reach": ("collect", "reach"), "collect_all": ("collect",), "click_sequence": (), "merge": (),
    # env kinds
    "avatar": (),
}


@dataclass
class Entry:
    id: str
    section: str
    text: str
    kind: str = "other"
    params: dict = field(default_factory=dict)
    status: str = "hypothesis"
    support: int = 0
    counter: int = 0
    source: str = "llm"        # llm | harness
    note: str = ""             # latest evidence / reason for the status
    level: int = 0             # level on which the entry was written

    def line(self) -> str:
        mark = {"hypothesis": "?", "confirmed": "OK", "refuted": "X"}[self.status]
        ev = f" [{self.support} for / {self.counter} against]" if (self.support or self.counter) else ""
        kp = f" <{self.kind} {json.dumps(self.params, separators=(',', ':'))}>" if self.kind != "other" and self.params else (f" <{self.kind}>" if self.kind != "other" else "")
        note = f" — {self.note}" if self.note else ""
        return f"[{self.id}][{mark}] {self.text}{kp}{ev}{note}"


def normalize_kind(kind: str, params: dict) -> tuple[str, dict]:
    """Models often put the section name in `kind` and the real kind in params ('action'/'type'). Repair that."""
    kind = str(kind or "other").strip().lower(); params = dict(params or {})
    if kind in KIND_KEYS:
        return kind, params
    for key in ("type", "kind", "action"):
        v = str(params.get(key, "")).strip().lower()
        if v in KIND_KEYS and not (v in ("move", "noop") and key == "action"):
            params.pop(key); return v, params
    return ("other" if kind in ("env", "rules", "rule", "win", "", "none") else kind), params


class Rulebook:
    def __init__(self):
        self.entries: list[Entry] = []
        self.plan: str = ""
        self.history: list[str] = []      # one line per edit (audit trail)
        self.version = 0

    # ── access ────────────────────────────────────────────────────────────
    def section(self, name: str) -> list[Entry]:
        return [e for e in self.entries if e.section == name]

    def get(self, id_: str) -> Optional[Entry]:
        for e in self.entries:
            if e.id == id_:
                return e
        return None

    def next_id(self, section: str) -> str:
        n = 1 + max([int(e.id[1:]) for e in self.entries if e.section == section] or [0])
        return f"{PREFIX[section]}{n}"

    def find(self, kind: str, params: dict, exact: bool = False) -> Optional[Entry]:
        """Entry of the same kind and identifying params. An entry agreeing on every given param (exact) is preferred over one
        that agrees only on the identifying keys; with exact=True only the former counts."""
        keys = KIND_KEYS.get(kind)
        if keys is None:
            return None
        loose = None
        for e in self.entries:
            # a key the entry does not carry is a wildcard (a model-written 'click colour 1' matches every region)
            if e.kind == kind and all(k not in e.params or str(e.params.get(k)) == str(params.get(k)) for k in keys):
                if all(str(e.params.get(k)) == str(v) for k, v in params.items() if k in e.params):
                    return e
                loose = loose or e
        return None if exact else loose

    # ── edits ─────────────────────────────────────────────────────────────
    def add(self, section: str, text: str, *, kind: str = "other", params: Optional[dict] = None, source: str = "llm",
            status: str = "hypothesis", note: str = "", level: int = 0, support: int = 0, counter: int = 0) -> Entry:
        if section not in SECTIONS:
            section = "rules"
        kind, params = normalize_kind(kind, params or {})
        e = Entry(self.next_id(section), section, text.strip()[:300], kind, params, status,
                  support, counter, source, note[:200], level)
        self.entries.append(e); self.version += 1
        self.history.append(f"+{e.id} ({source}) {e.text[:80]}")
        return e

    def apply_edits(self, edits: list, *, level: int = 0, source: str = "llm") -> list[str]:
        """Edit ops: {op: add, section, text, kind?, params?} | {op: confirm|refute|remove, id, note?} | {op: edit, id, text?, kind?, params?, note?}
        Machine-checkable entries keep their harness-decided status: the model may only add/edit/remove them."""
        done = []
        for ed in edits or []:
            if not isinstance(ed, dict):
                continue
            op = str(ed.get("op", "")).lower()
            try:
                if op == "add" and ed.get("text"):
                    e = self.add(str(ed.get("section", "rules")), str(ed["text"]), kind=str(ed.get("kind") or "other"),
                                 params=ed.get("params") if isinstance(ed.get("params"), dict) else {}, source=source, level=level)
                    done.append(f"added {e.id}")
                elif op in ("confirm", "refute") and self.get(str(ed.get("id", ""))):
                    e = self.get(str(ed["id"]))
                    if e.kind in KIND_KEYS and e.source == "harness":
                        done.append(f"{e.id}: status is decided by the harness for kind {e.kind}"); continue
                    e.status = "confirmed" if op == "confirm" else "refuted"; e.note = str(ed.get("note", ""))[:200]
                    self.version += 1; self.history.append(f"{op} {e.id}: {e.note[:60]}"); done.append(f"{op} {e.id}")
                elif op == "remove" and self.get(str(ed.get("id", ""))):
                    e = self.get(str(ed["id"])); self.entries.remove(e); self.version += 1
                    self.history.append(f"-{e.id}"); done.append(f"removed {e.id}")
                elif op == "edit" and self.get(str(ed.get("id", ""))):
                    e = self.get(str(ed["id"]))
                    if ed.get("text"):
                        e.text = str(ed["text"]).strip()[:300]
                    if ed.get("kind") or isinstance(ed.get("params"), dict):
                        e.kind, e.params = normalize_kind(ed.get("kind") or e.kind, ed.get("params") if isinstance(ed.get("params"), dict) else e.params)
                    if ed.get("note"):
                        e.note = str(ed["note"])[:200]
                    if ed.get("status") in STATUSES and not (e.kind in KIND_KEYS and e.source == "harness"):
                        e.status = ed["status"]
                    self.version += 1; self.history.append(f"edit {e.id}: {e.text[:60]}"); done.append(f"edited {e.id}")
            except Exception as ex:  # a malformed edit never breaks play
                done.append(f"edit failed: {type(ex).__name__}")
        return done

    # ── harness evidence ─────────────────────────────────────────────────
    def sync(self, facts: list[dict], *, level: int) -> list[str]:
        """facts: [{kind, params, support, counter, text}] induced from the transitions. Matching entries (same kind + key params)
        get the counts and a status; contradicting entries of the same kind (e.g. a different delta for the same key) are refuted;
        unmatched facts with support are added as harness entries. Returns change descriptions."""
        changes = []
        for f in facts:
            kind, params = f["kind"], f.get("params", {})
            section = "win" if kind in ("reach", "collect_reach", "collect_all", "click_sequence", "merge") else "env" if kind in ("avatar",) else "rules"
            e = self.find(kind, params)
            support, counter = int(f.get("support", 0)), int(f.get("counter", 0))
            status = "confirmed" if counter == 0 and support >= 2 else "refuted" if counter >= 1 and counter >= support else "hypothesis"
            if e is not None:
                # same identity: does the model's parametrisation agree with the evidence?
                agree = all(str(e.params.get(k)) == str(params.get(k)) for k in params if k in e.params)
                if not agree and e.source == "llm":
                    if e.status != "refuted":
                        e.status = "refuted"; e.note = f"evidence says {json.dumps(params, separators=(',', ':'))}"; self.version += 1
                        changes.append(f"{e.id} refuted: {e.note}")
                    if self.find(kind, params, exact=True) is None:   # add the harness version next to it (once)
                        n = self.add(section, f["text"], kind=kind, params=params, source="harness", status=status, level=level,
                                     support=support, counter=counter, note="induced from transitions")
                        changes.append(f"{n.id} added (harness): {n.text}")
                    continue
                new_status = status if e.source == "harness" or e.kind in KIND_KEYS else e.status
                if (e.support, e.counter, e.status) != (support, counter, new_status):
                    if e.status != new_status:
                        changes.append(f"{e.id} {e.status} -> {new_status} ({support} for / {counter} against)")
                    e.support, e.counter, e.status = support, counter, new_status; self.version += 1
                if e.source == "harness":
                    e.text = f["text"]
                    for k, v in params.items():
                        e.params[k] = v
            elif support >= 1:
                n = self.add(section, f["text"], kind=kind, params=params, source="harness", status=status, level=level,
                             support=support, counter=counter, note="induced from transitions")
                changes.append(f"{n.id} added (harness): {n.text}")
        # a refuted model entry that a confirmed harness entry of the same identity supersedes is noise: drop it (history keeps it)
        for e in list(self.entries):
            if e.source == "llm" and e.status == "refuted" and e.kind in KIND_KEYS:
                keys = KIND_KEYS[e.kind]
                if any(h is not e and h.source == "harness" and h.kind == e.kind and h.status == "confirmed"
                       and all(str(h.params.get(k)) == str(e.params.get(k)) for k in keys) for h in self.entries):
                    self.entries.remove(e); self.version += 1; self.history.append(f"-{e.id} (superseded)")
                    changes.append(f"{e.id} removed (superseded by the verified rule)")
        return changes

    # ── text / io ─────────────────────────────────────────────────────────
    def render(self, *, include_refuted: bool = True) -> str:
        out = []
        titles = {"env": "ENVIRONMENT (what is on the board)", "rules": "RULES (what actions do)", "win": "WIN CONDITIONS (how a level is completed)"}
        for s in SECTIONS:
            es = [e for e in self.section(s) if include_refuted or e.status != "refuted"]
            out.append(titles[s] + ":")
            out.extend("  " + e.line() for e in es) if es else out.append("  (none)")
        out.append("PLAN: " + (self.plan or "(none)"))
        return "\n".join(out)

    def stats(self) -> dict:
        return {s: {st: sum(1 for e in self.section(s) if e.status == st) for st in STATUSES} for s in SECTIONS}

    def to_json(self) -> dict:
        return {"version": self.version, "plan": self.plan, "entries": [asdict(e) for e in self.entries], "history": self.history[-200:]}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @staticmethod
    def from_json(d: dict) -> "Rulebook":
        b = Rulebook(); b.plan = d.get("plan", ""); b.version = int(d.get("version", 0)); b.history = list(d.get("history", []))
        b.entries = [Entry(**e) for e in d.get("entries", [])]
        return b
