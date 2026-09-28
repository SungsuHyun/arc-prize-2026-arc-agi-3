"""wml/refine.py — World Model Lab (spec §8): generate candidate models (priors-driven induction first, then K LLM
candidates), verify each on the WHOLE log, keep N hypotheses with score/coverage/violations, detect hidden state."""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

from ..core.contracts import Hypothesis, RuleModel
from ..core.types import Action, Scene, Transition
from ..perception.summarize import scene_text, summarize
from .context import observation_lines, select_context
from .evaluate import dedupe, evaluate, make_hypothesis, promotable
from .experiment import most_informative_action as _mia
from .induce import induce_hypotheses, induced_code
from .prompts import critique_prompt, diagnose_prompt, repair_prompt, world_model_prompt
from .sandbox import Sandbox


class WorldModelLab:
    def __init__(self, *, llm=None, sandbox: Optional[Sandbox] = None, memory=None, K: int = 4, N: int = 5, tau: float = 0.3, log=None,
                 validate_in_subprocess: bool = True):
        self.llm, self.sandbox, self.memory = llm, sandbox or Sandbox(), memory
        self.K, self.N, self.tau = K, N, tau
        self.log = log or (lambda *a, **k: None)
        self.validate_in_subprocess = validate_in_subprocess
        self.llm_calls = 0
        self.hidden_state_suspected = False
        self.last_context: list[dict] = []
        self.generation_log: list[dict] = []
        self._job: Optional[threading.Thread] = None      # background LLM generation (spec §15: keep planning meanwhile)
        self._job_result: list[Hypothesis] = []
        self._job_started = 0.0
        self.async_rounds = 0

    # ── public ──
    def refine(self, log: list[Transition], current: list[Hypothesis], priors: Optional[dict], *, scene: Scene, semantics: dict,
               available: list[Action], level_note: str = "", use_llm: bool = True, K: Optional[int] = None) -> list[Hypothesis]:
        H: list[Hypothesis] = []
        # 1. re-verify what we already have on the (grown) log
        for h in current:
            res = evaluate(h.model, log)
            H.append(make_hypothesis(h.model, res, h.code, h.name, h.origin, recent_mismatches=max(0, h.recent_mismatches - 1)))
        # 2. deterministic induction from semantics + priors
        for name, model in induce_hypotheses(log, semantics, available):
            res = evaluate(model, log)
            H.append(make_hypothesis(model, res, induced_code(name), name, "induced"))
        H = dedupe(H); H.sort(key=lambda h: h.sort_key(), reverse=True)
        best = H[0] if H else None
        # 3. LLM candidates when nothing is perfect — synchronously here, or in the background via start_llm_job()
        if use_llm and self.llm is not None and (best is None or best.score < 1.0 or best.coverage < 1.0):
            H += self._llm_candidates(log, best, priors, scene=scene, semantics=semantics, level_note=level_note, K=K or self.K)
            H = dedupe(H); H.sort(key=lambda h: h.sort_key(), reverse=True)
        # merge candidates a finished background job produced (re-verified on the current, larger log)
        for h in self.collect_job():
            res = evaluate(h.model, log)
            H.append(make_hypothesis(h.model, res, h.code, h.name, h.origin))
        H = dedupe(H); H.sort(key=lambda h: h.sort_key(), reverse=True)
        self._detect_hidden_state(log)
        for h in H:
            h.verified = promotable(evaluate(h.model, log)) if h.score >= 1.0 else False
        return H[:self.N]

    # ── background generation ──
    def job_running(self) -> bool:
        return self._job is not None and self._job.is_alive()

    def start_llm_job(self, log: list[Transition], best: Optional[Hypothesis], priors, *, scene: Scene, semantics: dict, level_note: str = "",
                      K: Optional[int] = None) -> bool:
        """Generate LLM candidates in a thread on a snapshot of the log; the orchestrator keeps acting on the current
        hypotheses and merges the result at a later HYPOTHESIZE (spec §15 'LLM latency -> plan on the top hypothesis')."""
        if self.llm is None or self.job_running() or self.llm.exhausted():
            return False
        snapshot = list(log)
        def run():
            try:
                self._job_result = self._llm_candidates(snapshot, best, priors, scene=scene, semantics=dict(semantics), level_note=level_note, K=K or self.K)
            except Exception as e:
                self.log(f"llm job failed: {e!r}")
                self._job_result = []
        self._job_result = []
        self._job = threading.Thread(target=run, daemon=True, name="wml-llm")
        self._job_started = time.time()
        self._job.start(); self.async_rounds += 1
        return True

    def collect_job(self) -> list[Hypothesis]:
        if self._job is None or self._job.is_alive():
            return []
        out, self._job_result, self._job = self._job_result, [], None
        if out:
            self.log(f"llm job delivered {len(out)} candidate(s) after {time.time() - self._job_started:.0f}s")
        return out

    def wait_job(self, timeout: float) -> bool:
        """Block up to `timeout` seconds for a running job (used only when nothing useful can be done meanwhile)."""
        if not self.job_running():
            return True
        self._job.join(timeout)
        return not self.job_running()

    def most_informative_action(self, H: list[Hypothesis], scene: Scene, available: list[Action], *, semantics=None, extra_clicks=(),
                                exclude: Optional[set] = None, state_key=None) -> Optional[Action]:
        return _mia(H, scene, available, tau=self.tau, semantics=semantics, extra_clicks=extra_clicks, exclude=exclude, state_key=state_key)

    def diagnose(self, h: Hypothesis, log: list[Transition], scene: Scene) -> Optional[dict]:
        """Violation diagnosis after a level change (spec §14, JSON {rule, cause, fix_hint})."""
        if self.llm is None or not h.violations:
            return None
        viol = [t for t in log if t.id in set(h.violations)][-5:]
        code = h.code or ""
        try:
            reply = self.llm.chat(diagnose_prompt(code[:4000], [summarize(t) for t in viol], scene_text(scene)), purpose="diagnose")
            self.llm_calls += 1
            return self.llm.extract_json(reply)
        except Exception as e:
            self.log(f"diagnose failed: {e!r}")
            return None

    # ── internals ──
    def _llm_candidates(self, log, best, priors, *, scene, semantics, level_note, K) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        ctx = select_context(log, best.violations if best else [], cap=40)
        obs = observation_lines(ctx)
        self.last_context = [{"id": t.id, "summary": summarize(t)} for t in ctx]
        sigs = (priors or {}).get("signatures", "") if priors else ""
        current_code = best.code if best and best.origin == "llm" and best.code else None
        violated = []
        if best and isinstance(best.model, RuleModel):
            violated = [r.name for r in best.model.rules() if r.violated_by]
        sem_text = json.dumps({k: v for k, v in semantics.items() if not k.startswith("_")}, default=str)[:3000]
        msgs = world_model_prompt(signatures=sigs, current_code=current_code, violated_rules=violated, observations=obs, semantics_text=sem_text,
                                  scene_text=scene_text(scene), hidden_state_hint=self.hidden_state_suspected, level_note=level_note)
        log_json = [self._t_json(t) for t in log]
        # K candidates are generated in parallel threads (each: one generation + at most one follow-up)
        results: list[list[Hypothesis]] = [[] for _ in range(K)]
        workers = [threading.Thread(target=self._one_candidate, args=(k, msgs, log, log_json, results), daemon=True) for k in range(K)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        for r in results:
            out.extend(r)
        return out

    def _one_candidate(self, k: int, msgs: list[dict], log: list[Transition], log_json: list[dict], results: list) -> None:
        out: list[Hypothesis] = []
        try:
            self._generate_candidate(k, msgs, log, log_json, out)
        except Exception as e:
            self.log(f"llm candidate {k} crashed: {e!r}")
        results[k] = out

    def _generate_candidate(self, k: int, msgs: list[dict], log: list[Transition], log_json: list[dict], out: list[Hypothesis]) -> None:
        for _ in range(1):
            if self.llm.exhausted():
                break
            try:
                reply = self.llm.chat(msgs, purpose="world_model", use_cache=(k == 0), override={"temperature": 0.6 + 0.1 * k} if k else None)
                self.llm_calls += 1
            except Exception as e:
                self.log(f"llm world_model failed: {e!r}")
                break
            code = self.llm.extract_code(reply)
            attempts = 0
            while attempts < 2:      # generate + at most ONE follow-up (repair or critique) per candidate
                if not code:
                    self.generation_log.append({"k": k, "error": "no code block"})
                    break
                if self.validate_in_subprocess:
                    v = self.sandbox.validate(code, log_json)
                else:
                    m = self.sandbox.load(code)
                    v = {"ok": m is not None, "error": "load failed"} if m is None else {"ok": True, **evaluate(m, log).__dict__}
                if v.get("ok"):
                    model = self.sandbox.load(code)
                    if model is None:
                        break
                    res = evaluate(model, log)
                    h = make_hypothesis(model, res, code, f"llm:{len(self.generation_log)}", "llm")
                    out.append(h)
                    self.generation_log.append({"k": k, "score": res.score, "coverage": res.coverage, "n_rules": len(model.rules())})
                    self.log(f"llm candidate {k}: score={res.score:.2f} cov={res.coverage:.2f} rules={len(model.rules())}")
                    # one critique round when the verification is imperfect (spec §14 'violation diagnosis' feedback)
                    if res.score < 1.0 and attempts < 1 and not self.llm.exhausted():
                        attempts += 1
                        mism = self._mismatch_lines(model, log, res.violations[:6])
                        unk = [summarize(t) for t in log if t.id in set(res.unknown[:4])]
                        try:
                            reply = self.llm.chat(critique_prompt(msgs + [{"role": "assistant", "content": reply}], res.score, res.coverage, mism, unk),
                                                  purpose="world_model", use_cache=False)
                            self.llm_calls += 1
                            code = self.llm.extract_code(reply)
                            continue
                        except Exception as e:
                            self.log(f"llm critique failed: {e!r}")
                    break
                attempts += 1
                self.generation_log.append({"k": k, "attempt": attempts, "error": str(v.get("error"))[:300]})
                self.log(f"llm candidate {k} rejected: {str(v.get('error'))[:200]}")
                if attempts >= 2 or self.llm.exhausted():
                    break
                try:
                    reply = self.llm.chat(repair_prompt(msgs + [{"role": "assistant", "content": reply}], str(v.get("error"))), purpose="world_model", use_cache=False)
                    self.llm_calls += 1
                    code = self.llm.extract_code(reply)
                except Exception as e:
                    self.log(f"llm repair failed: {e!r}")
                    break

    @staticmethod
    def _mismatch_lines(model, log: list[Transition], ids: list[str]) -> list[str]:
        from ..core.contracts import scene_mismatch
        out = []
        for t in log:
            if t.id not in ids:
                continue
            try:
                pred = model.predict(t.before, t.action)
            except Exception:
                pred = None
            if pred is None:
                out.append(f"[{t.id}] {summarize(t, 120)} | predicted: UNKNOWN")
                continue
            mm = scene_mismatch(pred, t.after)
            out.append(f"[{t.id}] observed: {summarize(t, 120)} | predicted-only objects (id,colour,bbox): {mm['only_predicted'][:3]} | observed-only: {mm['only_observed'][:3]}")
        return out

    @staticmethod
    def _t_json(t: Transition) -> dict:
        return {"id": t.id, "level": t.level, "step_idx": t.step_idx, "action": t.action.to_json(), "before": t.before.to_json(), "after": t.after.to_json(),
                "status_change": t.status_change}

    def _detect_hidden_state(self, log: list[Transition]) -> None:
        """Same frame hash + same action -> different results twice or more (spec §8 hidden state)."""
        seen: dict[tuple, set] = {}
        for t in log:
            if t.action.type == "RESET":
                continue
            seen.setdefault((t.before.frame_hash, t.action.label()), set()).add(t.after.frame_hash)
        conflicts = sum(1 for v in seen.values() if len(v) > 1)
        self.hidden_state_suspected = conflicts >= 2
