# v004-qwen-planner

## 가설

v003 인코더의 관측 텍스트를 Qwen에 넣어 JSON 계획을 받으면, 변화 로그에
드러난 규칙(이동 체계, 게이지 등)을 활용해 L1 무작위 탐색보다 목적 있는
행동이 가능하다.

## 변경점 (v003 대비)

- QwenPlanner: ollama /api/chat (think=false, format=json, seed 고정,
  system prompt에 게임 설정+JSON 계약). 엔드포인트 없으면 자동 비활성화
- 트리거: 계획 큐 소진 + (stuck 15액션 | 40액션 경과 | 레벨업), 게임당
  12회 예산, 최소 간격 8액션. 계획 실행 중 무변화 3연속 → 재계획
- 인코더 개선: resized 이벤트 통합("resized c3 s60→62"), 동일 변화 반복
  ×N 압축
- metrics에 llm_model/calls/errors/time, planned_actions 기록

## 관찰

- 파이프라인 안정: 게임당 9~12회 호출, 오류 ≈0, 호출당 ~2초(9b/35b 유사)
- **모델 크기가 결정적**:
  - 9b: 4게임 종합 0.0004 (vc33 레벨 1을 401액션 소모 후에야) — v003(0.010)보다 나쁨
  - **35b: vc33 레벨 2 완료, 점수 0.321 — 프로젝트 최고** (tag=model-35b 런)
- 9b는 탐색 커버리지(unique_states)는 늘리지만 계획 품질이 낮아 액션만
  낭비 → 점수 공식 (기준/사용액션)²에 불리
- ls20은 35b도 레벨 0 — 이동+게이지 게임은 더 긴 계획/기억이 필요한 듯

## 결론

- 채택: 35b를 기본 플래너로 (Kaggle RTX6000 30B급 경로와도 일치, docs/005)
- 후속: (v005) Kaggle 패키징 — _complete()만 in-process vLLM으로 교체.
  (v006) ls20류 개선 — 레벨별 기억, 게이지 인식 프롬프트, 계획 길이 튜닝
