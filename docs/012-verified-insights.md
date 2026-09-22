# 012. v008 검증된 액션 사실 + 코드 보상 — v007 보완

- **날짜**: 2026-09-22
- **관련 파일**: experiments/v008-verified-insights/ (agent.py, memory/insights.md)

## 목적

docs/009(v007)의 실패 원인 3가지 — 부정 진술뿐인 인사이트, -1로 붕괴한 보상,
검증 없는 저장 — 를 고쳐 9b 자기성찰 루프가 실제로 계획을 개선하는지 본다.
평가자만 35b로 두는 비대칭 구성도 비교한다.

## 방법

- 코드 측정 액션 모델: 인코더의 객체 diff를 "moves cN (dx,dy) / grows cN /
  shrinks cN / adds / removes / no change"로 일반화, 액션 키별(ACTION6은 클릭
  색으로 세분) 카운트 → `insights.md`의 "Verified Action Facts"에 덮어쓰기,
  플래너·평가자 프롬프트 최상단 주입
- 코드 보상 = levelups + 0.4·novelty + 0.2·change − 0.5·(1−change). 재계획
  트리거는 이 값만. LLM 점수는 로그
- LLM 인사이트는 `{action, fact}` 구조, 이 게임에서 시도된 액션만 저장.
  `next_strategy` 별도 저장·주입. 기준 개정 근사 중복 필터
- 실행: 동일 4게임 × 400스텝, 빈 메모리에서 시작

```bash
INSIGHTS_RESET=1 EVAL_MODEL=qwen3.5:9b  make exp-run NAME=v008   # 플래너 9b / 평가자 9b
INSIGHTS_RESET=1 EVAL_MODEL=qwen3.5:35b make exp-run NAME=v008   # 플래너 9b / 평가자 35b
```

## 결과

| run | 플래너/평가자 | 종합 점수 | 레벨 | 코드 보상 평균 | LLM 시간/게임 |
|---|---|---|---|---|---|
| eval-9b-fresh | 9b / 9b | **0.0058** | vc33 1 | +0.39 ~ +0.62 | ~70–88s |
| eval-35b-fresh | 9b / 35b | 0.0017 | vc33 1 | +0.39 ~ +0.59 | ~140–160s |
| 참고 v007 | 9b / 9b | 0.000 | 0 | (LLM -1 고정) | ~65s |
| 참고 v004 9b / v003 | | 0.0004 / 0.010 | vc33 1 / vc33 1 | | |

insights.md에 실제로 쌓인 검증 사실 (코드 측정, 효과/시도):

```
### ls20
- ACTION1: grows c3 90/94, shrinks c11 89/94        (이동 = 게이지 변화)
### sk48
- ACTION1: moves c6 (+0,-6) 51/65, moves c9 (+0,-6) 33/65
- ACTION6@c1: no change 15/15                        (클릭 무효)
### vc33
- ACTION6@c5: grows c4 72/72, shrinks c7 72/72       (어디를 클릭해도 카운터 변화)
```

- v007의 세 문제는 해결됨: 액션→효과 사실이 정확히 기록되고, 보상은 -1 붕괴
  없이 +0.4~0.6, 시도 안 한 액션에 대한 LLM 주장은 폐기(vc33 9건)
- 그러나 LLM(9b·35b 모두)이 게이지/카운터(c11, c7, c4)를 "목표 객체"로
  해석하고 "가장 효율적인 한 액션에 수렴"을 반복 권고 → 플래너가 같은 액션
  반복, 레벨 완료 조건 탐색이 줄어 v003(0.010)에는 못 미침
- General Insights는 근사 중복 필터를 뚫은 동의어 문장 12개로 포화. 기준
  개정은 게임 특정 색·수치로 오염(v12까지 bump)
- 35b 평가자: 문장만 매끈, 같은 오해. GPU 스왑으로 시간 2배 → 폐기

## 결론 / 다음 단계

- 채택: ActionModel(코드 측정 액션 사실)과 코드 보상. 9b 계열 최고(0.0058)
- 폐기: 35b 평가자 비대칭, LLM 자유 서술 general insight
- 다음: (1) 사실→계획을 코드로(이동 액션 판별 + 탐색), LLM은 가설만,
  (2) 매 액션 단조 변화하는 객체를 "게이지"로 라벨링해 목표에서 제외,
  (3) 기준 개정에 게임 특정 토큰 포함 시 거부, 중복 필터 강화
