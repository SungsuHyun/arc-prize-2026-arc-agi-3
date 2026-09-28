"""Subprocess worker for Sandbox.validate: resource limits, then build_model() + evaluate on the log (JSON in/out)."""
from __future__ import annotations

import json
import resource
import sys
import traceback


def main() -> None:
    payload = json.loads(sys.stdin.read())
    cpu, mem = int(payload.get("cpu", 2)), int(payload.get("mem", 256))
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem * 2 ** 20, mem * 2 ** 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    except Exception:
        pass
    try:
        from pbg.core.types import Action, Scene, Transition, Diff
        from pbg.wml.evaluate import evaluate
        from pbg.wml.sandbox import load_in_process
        ns = load_in_process(payload["code"])
        if "build_model" not in ns:
            print(json.dumps({"ok": False, "error": "no build_model"})); return
        model = ns["build_model"]()
        log = []
        for d in payload["log"]:
            t = Transition(Scene.from_json(d["before"]), Action.from_json(d["action"]), Scene.from_json(d["after"]), Diff(), d.get("status_change"), [], d.get("id", ""),
                           int(d.get("level", 1)), int(d.get("step_idx", 0)))
            log.append(t)
        res = evaluate(model, log)
        rules = [{"name": r.name, "confidence": r.confidence} for r in model.rules()] if hasattr(model, "rules") else []
        print(json.dumps({"ok": True, "score": res.score, "coverage": res.coverage, "violations": res.violations[:50], "unknown": res.unknown[:50],
                          "rule_confidence": res.rule_confidence, "prediction_key": res.prediction_key, "n": res.n, "rules": rules,
                          "per_class": {k: list(v) for k, v in res.per_class.items()}}))
    except MemoryError:
        print(json.dumps({"ok": False, "error": "memory limit"}))
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()[-1500:]}))


if __name__ == "__main__":
    main()
