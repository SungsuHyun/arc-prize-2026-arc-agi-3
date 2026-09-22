# 005. Qwen 기반 LLM 에이전트 솔루션 초안

- **날짜**: 2026-09-22
- **관련 파일**: experiments/v002-effect-explorer/, agent/my_agent.py(추후), scripts/build_notebook.py(추후 확장)

## 모델 선택

요청은 "Qwen 3.8 27B"였는데 정확히 일치하는 공개 모델명이 없어 **~27B급
Qwen3 계열**로 해석함. 후보:

| 모델 | 특징 | 판단 |
|---|---|---|
| **Qwen3-30B-A3B (MoE)** | 총 30B, 활성 3B → 추론 속도가 dense 3B급 | **1순위** — 액션 루프에 latency가 결정적 |
| Qwen3-32B (dense) | 품질 우위, 속도 열세 | 플래닝 품질이 부족할 때 대안 |

정확한 모델명이 따로 있다면(예: 사내/신규 릴리스) 알려주면 교체 — 아래
설계는 모델 교체가 config 수준이 되도록 잡음.

## 제약 조건 (설계를 결정하는 것들)

1. **Kaggle 재실행은 인터넷 차단** → API 호출 불가. 가중치를 Kaggle
   Model/Dataset으로 노트북에 첨부하고 오프라인 로드해야 함
2. **하드웨어**: RTX 6000(48GB, ARC-AGI-3 전용) 1장이 현실적 타깃.
   30B급은 4bit 양자화(AWQ/GPTQ) 시 약 17–20GB + KV 캐시로 수용 가능
3. **시간 예산 vs 액션 예산**: 점수 공식이 `(기준액션/사용액션)²`이므로
   액션 낭비는 점수를 깎고, LLM 호출 낭비는 시간을 깎음. 둘 다 아껴야 함
4. **관측 크기**: 64×64 그리드를 원시 텍스트로 넣으면 프레임당 4천+ 토큰
   → 압축 인코딩 필수

## 아키텍처: 3계층 하이브리드

LLM을 매 액션 호출하지 않는다. 싼 파이썬 계층이 기본 동작하고, LLM은
"생각이 필요한 순간"에만 개입한다.

```
매 액션   L0 reflex   프레임 diff, no-op 필터, 루프 감지        (~0ms)
매 액션   L1 explorer 상태별 액션-효과 테이블, novelty 우선탐색  (~1ms)
가끔      L2 planner  Qwen: 게임 규칙 가설 + 액션 시퀀스 계획    (~수 초)
```

**L2 호출 트리거**: 새 레벨 진입 / N액션 동안 진전 없음(stuck) /
계획 큐 소진. 계획은 액션 시퀀스(JSON)로 반환되어 큐로 실행하고,
L0가 계획 실행 중 예상 밖 프레임 변화를 감지하면 재계획.

### LLM 입력 (압축 관측)

원시 그리드 대신:
- **객체 목록**: connected component별 (색, bbox, 크기, 중심) — v002에
  이미 구현된 `object_centroids`의 확장
- **변화 로그**: 최근 K개 액션별 "무엇이 어떻게 변했나" (이동/생성/소멸/색변화)
- **액션-효과 통계**: L1의 tally (어떤 액션이 효과가 있었는지)
- 진행도: levels_completed / win_levels, 남은 액션 예산

### LLM 출력 (구조화)

```json
{"hypothesis": "ACTION1-4는 이동, 목표는 파란 칸 도달로 추정",
 "plan": [{"action": 3}, {"action": 3}, {"action": 6, "x": 12, "y": 40}],
 "confidence": 0.6}
```

## 실행 스택

- **추론 엔진**: vLLM (오프라인 wheel로 설치). guided decoding으로 JSON 강제
- **로컬 개발**: OpenAI 호환 로컬 서버(vllm serve/ollama)를 `QWEN_ENDPOINT`
  환경변수로 연결. 없으면 L2가 None을 반환하고 L1로 폴백 (v002 스텁 계약)
- **Kaggle**: 노트북에서 in-process vLLM 로드. `kernel-metadata.json`의
  `model_sources`/`dataset_sources`에 가중치+wheel 데이터셋 추가,
  `build_notebook.py`에 설치 셀 확장 필요

## 로드맵 (실험 버전 계획)

| 버전 | 내용 | 상태 |
|---|---|---|
| v002 | effect-explorer: L0+L1 + L2 스텁 (LLM 없이 동작) | **완료** — 실행 검증 |
| v003 | 관측 인코더: 객체 추출·변화 로그를 텍스트로, 프롬프트 오프라인 평가 | |
| v004 | 로컬 Qwen 연결 (QWEN_ENDPOINT), stuck-trigger 재계획 루프 | |
| v005 | Kaggle 패키징: 가중치/wheel 데이터셋, build_notebook 확장, T4→RTX6000 | |
| v006+ | 프롬프트/트리거 튜닝, 액션 예산 최적화 (점수 공식 역산) | |

## 리스크

- **latency**: RTX6000에서 30B-A3B 4bit이 계획 1회당 몇 초인지 v004에서 실측
  필요. 너무 느리면 계획 주기를 늘리거나 14B급으로 하향
- **T4 폴백 불가**: 2×T4(32GB, compute 7.5)는 30B급에 빠듯 — RTX6000 전제.
  GPU 쿼터 관리 필요
- **게임 다양성**: 히든 게임이 클릭 중심이면 객체 타게팅이, 이동 중심이면
  방향 액션 모델링이 중요 — L1 통계가 게임별로 자동 적응하는 구조 유지
