#!/usr/bin/env python
"""Evaluation report for rulebook runs: per-game / per-level metrics, prediction quality, plan usage, failure taxonomy,
and mean±sd across passes. Reads run JSONs (experiments/rulebook/results/run-*.json) and the per-game logs next to them.

  python scripts/rulebook_eval.py --tag eval-choose-15m            # every run whose tag contains the string
  python scripts/rulebook_eval.py --run 20260927-080029-4032268    # one run id
Writes experiments/rulebook/eval/<name>.md and .json and prints the markdown."""
from __future__ import annotations

import argparse, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "rulebook" / "results"
OUT = ROOT / "experiments" / "rulebook" / "eval"

UNKNOWN_WORDS = ("never", "unknown", "not predictable", "undetermined", "not modelled", "not yet", "effect model was wrong", "no consistent")


def parse_log(path: Path) -> dict:
    """Counters per level from the agent log."""
    per: dict = defaultdict(lambda: Counter())
    labels: dict = defaultdict(Counter)
    latencies: dict = defaultdict(list)
    goal_progress: dict = defaultdict(list)
    if not path.exists():
        return {}
    in_code = None
    for ln in path.read_text(errors="replace").splitlines():
        m = re.match(r"\[\s*(\d+)s a\s*(\d+) L(\d+)\] (.*)", ln)
        if not m:
            if in_code is not None and "plan_actions[" in ln:   # coder mode: a plan run from model code counts as a chosen plan
                per[in_code]["plan_chosen"] += 1; in_code = None
            continue
        in_code = int(m.group(3)) if m.group(4).startswith("CODE:") else None
        t, a, lv, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        c = per[lv]
        if rest.startswith("OK :"):
            c["ok"] += 1
        elif rest.startswith("MISMATCH:"):
            c["mismatch"] += 1
        elif rest.startswith("obs:"):
            c["obs"] += 1
            if any(w in rest.split("| actual:")[0] for w in UNKNOWN_WORDS):
                c["unknown"] += 1
        elif rest.startswith("DECIDE:"):
            c["decisions"] += 1
            body = rest[len("DECIDE: "):]
            if "-> fallback" in body or "not a candidate" in body:
                c["fallback"] += 1
                lab = body.split("-> fallback")[-1].split("->")[-1].strip()
            elif "-> program runs" in body or "-> " in body and "already tried" in body:
                lab = body.split("-> ")[-1].strip()
            else:
                lab = body.split(" (")[0].split(" already")[0].split(" resolved")[0].strip()
            labels[lv][lab] += 1
            if "already tried here" in rest:
                c["veto"] += 1
            if lab.startswith("plan:"):
                c["plan_chosen"] += 1
            if lab == "submit":
                c["submit"] += 1
        elif rest.startswith("CODE:"):
            c["code_turns"] += 1
        elif "calls without an action" in rest:
            c["idle_guard"] += 1
        elif rest.startswith("plan candidate: plan:"):
            c["plan_offered"] += 1
        elif rest.startswith("goal state:"):
            c["goal_lines"] += 1
            if "HOLDS" in rest:
                c["goal_holds"] += 1
        elif rest.startswith("win predicates refuted"):
            c["preds_refuted"] += rest.count("refuted:")
        elif rest.startswith("win facts:"):
            c["win_facts"] += 1
        elif rest.startswith("REVIEW ("):
            c["reviews"] += 1
        elif "GAME OVER" in rest and rest.startswith("obs:") is False and "REVIEW" not in rest:
            pass
        mm = re.match(r"\[model:(\w+)\] think=(\w+) (\d+)s", rest)
        if mm:
            latencies[mm.group(1)].append(int(mm.group(3)))
            c[f"calls_{mm.group(1)}"] += 1
        if "GAME OVER —" in rest:
            c["game_over"] += 1
    out = {}
    for lv, c in per.items():
        d = dict(c)
        acts = c["ok"] + c["mismatch"] + c["obs"]
        d["actions"] = acts
        top = labels[lv].most_common(1)
        d["top_label"] = top[0][0] if top else ""
        d["top_label_share"] = round(top[0][1] / max(1, c["decisions"]), 2) if top else 0.0
        d["distinct_labels"] = len(labels[lv])
        out[lv] = d
    return {"levels": out, "latency": {k: (round(statistics.mean(v), 1), max(v)) for k, v in latencies.items()}}


def taxonomy(lv_stats: dict, completed: bool, reason: str) -> str:
    if completed:
        return "completed"
    if not lv_stats:
        return "no_data"
    acts = max(1, lv_stats.get("actions", 0)); dec = max(1, lv_stats.get("decisions", 0))
    if lv_stats.get("game_over", 0) >= 3:
        return "game_over_loop"
    if lv_stats.get("goal_lines", 0) == 0:
        return "no_win_hypothesis"
    if lv_stats.get("preds_refuted", 0) >= 1 and lv_stats.get("goal_holds", 0) == 0 and lv_stats.get("plan_offered", 0) == 0:
        return "hypotheses_refuted"
    if dec >= 15 and lv_stats.get("top_label_share", 0) >= 0.4:
        return "stuck_repeating"
    if lv_stats.get("unknown", 0) / acts >= 0.6:
        return "predictor_blind"
    if lv_stats.get("mismatch", 0) / acts >= 0.3:
        return "mismatch_heavy"
    if lv_stats.get("plan_offered", 0) > 0 and lv_stats.get("plan_chosen", 0) == 0:
        return "plans_ignored"
    return "budget_" + (reason or "unknown").replace(" ", "_")[:20]


def load_runs(tag: str, run_ids: list) -> list:
    runs = []
    for p in sorted(RES.glob("run-*.json")):
        d = json.loads(p.read_text())
        if run_ids and d["run_id"] not in run_ids:
            continue
        if tag and tag not in d.get("tag", ""):
            continue
        runs.append(d)
    return runs


def evaluate(runs: list) -> dict:
    rows = []; tax = Counter(); by_game: dict = defaultdict(list)
    for d in runs:
        log_dir = RES / "logs" / d["run_id"]
        for g in d.get("games", []):
            gid = g["game_id"].split("-")[0]
            lg = parse_log(log_dir / f"{gid}.log")
            levels = g.get("levels", []) or []
            lal = g.get("level_action_log", []) or []
            base = g.get("level_baseline_actions") or []
            eff = []
            for i, used in enumerate(lal):
                if i < len(base) and base[i] and used:
                    eff.append(round(min(1.0, base[i] / used) ** 2, 2))
            last = levels[-1] if levels else {}
            lv_last = last.get("level", g.get("levels_completed", 0) + 1)
            ls = lg.get("levels", {}).get(lv_last, {})
            tx = taxonomy(ls, bool(last.get("completed")), last.get("reason", g.get("stop_reason", "")))
            tax[tx] += 1
            tot = Counter()
            for lv, c in lg.get("levels", {}).items():
                for k in ("ok", "mismatch", "obs", "unknown", "decisions", "fallback", "veto", "plan_offered", "plan_chosen", "submit", "reviews", "game_over", "code_turns", "idle_guard", "preds_refuted"):
                    tot[k] += c.get(k, 0)
            acts = max(1, tot["ok"] + tot["mismatch"] + tot["obs"])
            row = {"run": d["run_id"][:15], "tag": d.get("tag", ""), "mode": d["config"]["params"].get("mode", "choose"),
                   "game": gid, "levels": g.get("levels_completed", 0), "score": round(g.get("score") or 0, 3),
                   "actions": g.get("actions", 0), "per_level_actions": lal, "baseline": base[:len(lal) + 1], "efficiency": eff,
                   "seconds": g.get("seconds"), "model_seconds": g.get("model_seconds"), "calls": g.get("model_calls", {}),
                   "ok_rate": round(tot["ok"] / acts, 2), "mismatch_rate": round(tot["mismatch"] / acts, 2), "unknown_rate": round(tot["unknown"] / acts, 2),
                   "decisions": tot["decisions"], "fallback": tot["fallback"], "veto": tot["veto"], "plans": f"{tot['plan_chosen']}/{tot['plan_offered']}",
                   "submits": tot["submit"], "reviews": tot["reviews"], "game_overs": tot["game_over"], "preds_refuted": tot["preds_refuted"],
                   "latency": lg.get("latency", {}), "last_level": lv_last, "last_level_top_label": ls.get("top_label", ""),
                   "last_level_top_share": ls.get("top_label_share", 0), "taxonomy": tx, "stop": g.get("stop_reason", g.get("error", ""))}
            rows.append(row); by_game[gid].append(row)
    agg = {}
    for gid, rs in by_game.items():
        sc = [r["score"] for r in rs]; lv = [r["levels"] for r in rs]
        agg[gid] = {"n": len(rs), "score_mean": round(statistics.mean(sc), 2), "score_sd": round(statistics.pstdev(sc), 2) if len(sc) > 1 else 0.0,
                    "levels_mean": round(statistics.mean(lv), 2), "levels_max": max(lv), "taxonomy": Counter(r["taxonomy"] for r in rs).most_common(1)[0][0]}
    overall = {"runs": len(runs), "games": len(rows), "score_mean": round(statistics.mean([r["score"] for r in rows]), 3) if rows else 0,
               "level1_rate": round(sum(1 for r in rows if r["levels"] >= 1) / max(1, len(rows)), 2), "level2_rate": round(sum(1 for r in rows if r["levels"] >= 2) / max(1, len(rows)), 2),
               "ok_rate": round(statistics.mean([r["ok_rate"] for r in rows]), 2) if rows else 0, "unknown_rate": round(statistics.mean([r["unknown_rate"] for r in rows]), 2) if rows else 0,
               "mismatch_rate": round(statistics.mean([r["mismatch_rate"] for r in rows]), 2) if rows else 0, "taxonomy": dict(tax.most_common())}
    return {"rows": rows, "by_game": agg, "overall": overall}


def markdown(ev: dict, name: str) -> str:
    o = ev["overall"]
    lines = [f"# rulebook eval — {name}", "", f"runs {o['runs']}, game-runs {o['games']}, score mean {o['score_mean']}, level1 rate {o['level1_rate']}, level2 rate {o['level2_rate']}, "
             f"prediction ok/unknown/mismatch {o['ok_rate']}/{o['unknown_rate']}/{o['mismatch_rate']}", "",
             "Failure taxonomy (last level played): " + ", ".join(f"{k} {v}" for k, v in o["taxonomy"].items()), "",
             "| game | n | score mean±sd | levels mean/max | taxonomy |", "|---|---|---|---|---|"]
    for gid, a in sorted(ev["by_game"].items(), key=lambda kv: -kv[1]["score_mean"]):
        lines.append(f"| {gid} | {a['n']} | {a['score_mean']}±{a['score_sd']} | {a['levels_mean']}/{a['levels_max']} | {a['taxonomy']} |")
    lines += ["", "| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ev["rows"]:
        lines.append(f"| {r['run']} | {r['game']} | {r['levels']} | {r['score']} | {r['per_level_actions']} | {r['efficiency']} | {r['ok_rate']}/{r['unknown_rate']}/{r['mismatch_rate']} | "
                     f"{r['decisions']} | {r['fallback']}/{r['veto']} | {r['plans']} | {r['submits']} | {r['reviews']} | {r['game_overs']} | {r['taxonomy']} | {r['last_level_top_label'][:18]} ({r['last_level_top_share']}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--run", action="append", default=[]); ap.add_argument("--name", default="")
    a = ap.parse_args()
    runs = load_runs(a.tag, a.run)
    if not runs:
        print("no runs"); return
    ev = evaluate(runs)
    name = a.name or a.tag or a.run[0]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(ev, indent=1, default=str))
    md = markdown(ev, name); (OUT / f"{name}.md").write_text(md); print(md)


if __name__ == "__main__":
    main()
