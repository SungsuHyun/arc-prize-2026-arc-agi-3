# 017. Duck 하네스 로컬 재현 + nav 헬퍼 하이브리드 (로드맵 D5/D8)

- **날짜**: 2026-09-23
- **관련 파일**: harness/duck/ (MIT, Tufalabs/duck-harness 사본),
  harness/duck/ARC3-Inference/inference/agent/nav_helpers.py, python_tool_sandbox.py,
  tool_agent.py, prompts.py; scripts/smoke_llm.py; .venv-llm(vllm 0.30 서버)

## 목적

리더보드 1위 하네스(Duck: 27B + 파이썬 REPL 툴)를 로컬 5090에서 돌려 기준선을 잡고,
우리 강점(이동 게임 내비게이션)을 헬퍼 `nav`로 넣었을 때 얻는 이득을 같은 조건에서
측정한다.

## 방법

1. **재현 환경**: `uv sync --locked`(Python 3.12.12)로 하네스 venv, 모델 서버는
   우리 `.venv-llm`의 `vllm serve`(OpenAI 호환, tool parser qwen3_coder, reasoning
   parser qwen3, 32k 컨텍스트). NVML 불일치 우회를 venv 전역 `.pth` 심(`vllm_nvml_shim`)
   으로 옮겨 서버 프로세스에도 적용. 설정 `configs/local5090.json`
   (게임 1개, 1패스, 동시 1, 게임당 20분).
2. **모델**: (a) Qwen3.5-35B-A3B GPTQ-int4(우리 Kaggle 모델), (b) Duck 원본에
   가까운 Qwen3.6-27B AWQ(Kaggle 미러 `bachhg/qwen3-6-27b-awq`, 19GB; 5090에서 KV
   95k 토큰).
3. **nav 헬퍼**: v006~v011의 인코더/내비게이터를 `transitions`(전/후 프레임)에서
   매 호출 다시 만드는 무상태 클래스로 이식. `nav.summary()`, `nav.map()`,
   `nav.path_to(row,col)`, `nav.targets()`, `nav.frontier()`, `nav.gauge()`.
   샌드박스 전역에 주입하고, 시스템 프롬프트 addendum + 매 턴 유저 프롬프트에
   "Navigation helper summary" 블록을 호스트 측에서 계산해 삽입.
   오프라인 테스트: ls20에서 방향 6회 탐색 → 수정 타일 경로 6 → 자물쇠 경로 7,
   총 19액션으로 레벨 1 완료.

## 결과

| 설정 | ls20 | m0r0 | vc33 | 비고 |
|---|---|---|---|---|
| Duck 저자 example-run (Qwen3.6-27B FP8, 20패스 평균) | 0.35 (0.2 lv) | 0.05 (0.1 lv) | 3.33 (1.6 lv) | 25게임 평균 1.60±0.45, 라벨 "0-history-turns" |
| Duck 원본 + 35B-A3B GPTQ (로컬, 1패스 20분) | 0.00 (0 lv, 972액션, 72턴) | – | – | 마지막 턴 UP×20 연타, 월드모델 텍스트 비어 있음 |
| Duck 원본 + Qwen3.6-27B AWQ (로컬, 1패스 20분) | 0.00 (0 lv, 31액션, 28턴) | – | 0.00 (0 lv, 19액션) | 월드모델 텍스트는 정확(5×5 블록 아바타, 통로/벽, 이동 5칸)하나 사고 토큰으로 64 tok/s → 20분에 액션 31개뿐 |
| **Duck + nav (하이브리드) + 27B AWQ (로컬, 1패스 20분)** | **3.57 (lv1을 20액션, 기준 22)** + lv2 81액션 진행 중 타임아웃 | **0.36 (lv1을 109액션, 기준 30)** + lv2 진행 | 0.00 (lv1에 102액션) | 같은 모델·시간에서 이동 게임 0→레벨 1 |

관찰: 35B-A3B는 빠르지만(180 tok/s, 972액션) 월드모델을 못 쓰고 UP 연타로 빠지고, 27B는
정확하지만 로컬 단일 게임에선 너무 느리다. Duck의 Kaggle 설정은 28게임 동시·게임당 132분이라
처리량 문제를 동시성으로 푼다. 로컬 반복 실험은 35B-A3B, 최종 확인은 27B로 나눈다.

우리 프로그램형 v011(LLM 없음)은 ls20 3.57, m0r0 4.19, vc33 0.4±로, Duck이 약한
이동 게임에서 강하고 Duck이 강한 클릭 게임(vc33 3.33)에서 약하다 — 상보적.

## 결론 / 다음 단계

- **nav 헬퍼는 Duck의 약점(이동 게임)을 메운다**: 같은 27B·같은 20분에서 ls20 0→3.57
  (사람 기준 22액션에 20액션), m0r0 0→0.36. 클릭 게임(vc33)은 영향 없음(예상대로).
  n=1이므로 25게임·시드 반복으로 확인 필요
- 다음: (1) 하이브리드 25게임 동시 4·20분 1패스 → 원본 Duck 동일 조건과 비교,
  (2) Kaggle 패키징(Duck의 `make kaggle-duck` 경로: 소스 데이터셋 + vLLM wheelhouse +
  모델 스냅샷), (3) 클릭 게임용 헬퍼(우리 ClickSweeper의 게이지 인식·버킷 기록)를
  `clicks`로 추가하는 것은 데이터가 더 나온 뒤 결정
