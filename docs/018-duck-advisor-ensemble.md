# 018. Duck 자문 앙상블: 작은 모델들의 의견 → 27B 결정 모델

- **날짜**: 2026-09-23
- **관련 파일/커밋**: harness/duck/ARC3-Inference/inference/agent/advisors.py (신규),
  tool_agent.py / prompts.py 훅(05a799a), scripts/duck_vllm_serve.sh,
  scripts/duck_local_run.sh, scripts/duck_advisor_report.py (538451a)

## 목적

사용자 제안 구조를 별도 버전으로 만들어 017의 하이브리드(Duck 27B + nav)와 같은 조건에서
비교한다: **작은 Qwen 여러 개(자문, advisor)가 매 턴 게임 상태에 대한 의견을 내고, 결정
모델(27B Duck 에이전트)이 모두 고려해 결정**한다. 사전 판단(같은 세션 대화)은 "처리량
병목 때문에 손해일 가능성이 높다"였으므로, 이 실험의 질문은 (1) 자문이 턴당 시간을
얼마나 늘리는가, (2) 그 비용을 액션 절약·레벨 완료로 상쇄하는가이다.

## 방법

1. **자문 계층** (`advisors.py`, `ARC3_ADVISORS` 환경변수 없으면 완전 비활성):
   결정 턴마다 호스트가 상태를 압축 텍스트로 렌더링(32×32 다운샘플 맵, 세그멘테이션
   객체 목록·반복 형상, 직전 프레임 diff, nav 요약, 결정 모델이 들고 있는 월드모델)해
   자문 모델들에 병렬로 질의(ollama `/api/chat`, thinking 끔, ≤256 토큰). 역할은
   mechanics(무엇이 바뀌었나·액션 효과) / goal(목표·최단 계획) / skeptic(가장 약한 가정과
   판별 프로브). 응답은 `HYPOTHESIS / EVIDENCE / RECOMMEND / CONFIDENCE / WARNING` 5줄
   고정 형식. 결정 모델의 유저 프롬프트에 "Advisor ensemble" 블록으로 삽입되고 시스템
   프롬프트에 "너가 결정 모델이다, 자문은 가설이다" addendum이 추가된다.
   턴별 의견·지연시간·직전에 실행된 액션을 `artifacts/<game>_p0_tool_runtime_state_advisors.jsonl`에 기록.
2. **모델**: 결정 = Qwen3.6-27B AWQ(vLLM :1234), 자문 = ollama qwen3.5:9b(Q4_K_M) ×2 역할
   (+ qwen3.5:4b skeptic, 메모리가 허용하는 경우). 5090 32GB 한 장이라 vLLM
   `--gpu-memory-utilization`을 0.90→0.78로 낮춰 ollama와 공존(27B의 KV 95k→약 36k 토큰).
3. **비교**: 017과 같은 3게임(ls20, m0r0, vc33), 게임당 20분, 1패스, 동시 1.
   기준선은 017의 하이브리드(같은 27B, 자문 없음) 수치.

```bash
scripts/duck_vllm_serve.sh UTIL=0.78 MAX_NUM_SEQS=1   # (환경변수 형태로 전달)
ARC3_ADVISORS='[{"name":"q9b-mech","model":"qwen3.5:9b","role":"mechanics"},
                {"name":"q9b-goal","model":"qwen3.5:9b","role":"goal"}]' \
  scripts/duck_local_run.sh ens27-ls20 ls20-9607627b
.venv/bin/python scripts/duck_advisor_report.py vendor/duck-runs/<run_dir>
```

## 결과

(실행 중 — 다른 세션의 25게임 하이브리드 실행이 GPU를 쓰고 있어 그 종료 후 진행)

## 결론 / 다음 단계

(결과 후 작성)
