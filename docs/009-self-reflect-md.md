# 009. v007 자기 성찰 마크다운 — 9b + 보상 평가자 루프

- **날짜**: 2026-09-22
- **관련 파일**: experiments/v007-self-reflect-md/ (agent.py, memory/insights.md)

## 목적

v004에서 9b 플래너는 계획 품질이 낮아 v003보다 못했다(0.0004). 모델을 키우지
않고, 몇 턴마다 "보상 평가자" 프롬프트(사용자 제공 템플릿)로 자기 행동을
채점하고 결과를 마크다운 파일에 누적·주입하면 처음 보는 게임에서도 이전
인사이트로 계획이 개선되는지 실험한다.

## 방법

- 루프: 25액션마다 최근 행동 창(액션×횟수, 화면 변화 수, 레벨/새 상태 증가)을
  평가자 템플릿에 채워 `qwen3.5:9b` 호출 (think=false, format=json)
- 출력 JSON: `reasoning`, `reward_score`, `template_update_suggestion`
  (needs_update → 기준 문구 추가 + 버전 bump), 확장 필드 `game_insights`,
  `general_insights`
- `memory/insights.md`: Evaluation Criteria(vN) / General Insights /
  Game Notes: <game> / Reward Log / Criteria History. 게임·런을 넘어 지속
- 플래너 system prompt에 `[INSIGHTS: general]`, `[INSIGHTS: this game]`,
  `[LAST EVAL]` 주입. 보상 ≤ -0.2 → 계획 폐기 + 즉시 재계획
- 실행: v004 벤치와 동일 조건 (ls20, lf52, sk48, vc33 × 400 스텝)

```bash
INSIGHTS_RESET=1 make exp-run NAME=v007 # run1: 빈 메모리에서 시작
make exp-run NAME=v007                  # run2: run1의 insights.md 이어받기
```

## 결과

| run | 메모리 | 종합 점수 | 레벨 | 평가자 오류 | 평균 보상 | 기준 버전 |
|---|---|---|---|---|---|---|
| run1 (fresh) | 빈 파일 | 0.000 | 0/4게임 | 0 | -0.79 ~ -1.00 | v1→v3 |
| run2 (carried) | run1 이어받음 | 0.000 | 0/4게임 | 0 | -0.79 ~ -1.00 | v3 유지 |
| 참고 v003 | — | 0.010 | vc33 1 | — | — | — |
| 참고 v004 9b | — | 0.0004 | vc33 1 | — | — | — |

배관은 정상(게임당 평가자 16회 + 플래너 12회, JSON 오류 0, LLM ~65초/게임)
이고 `memory/insights.md`도 게임·런을 넘어 누적된다. 그러나:

- 9b 평가자의 인사이트는 전부 "임의의 크기 조절/왕복 이동은 무의미" 류의
  부정 진술이며 액션→효과 규칙은 한 줄도 없음 (아래 실제 응답)
- 보상이 상수 -1로 붕괴(레벨 미완료 = -1) → 재계획 트리거가 매 창마다 발동해
  신호 가치 없음. 기준 개정도 자기 강화(v2·v3 동일 문구)
- 결과적으로 주입된 인사이트가 계획을 바꾸지 못함 → 0.0

실제 평가자 응답 예 (lf52, turn 50):

```json
{"reasoning": "외적 보상: 에이전트는 레벨 완료에 전혀 기여하지 않았습니다. ... 상태 변수(크기, 위치)만 반복적으로 조작 ...",
 "reward_score": -0.95,
 "template_update_suggestion": {"needs_update": false, "suggestion": "..."},
 "game_insights": ["Arbitrary manipulation of color/size without a clear sequence towards level completion results in severe stagnation and negative reward.",
                   "The goal is to complete levels, not to manipulate state variables like color/size for their own sake."],
 "general_insights": []}
```

발견한 버그: `INSIGHTS_RESET=1`을 에이전트 생성자에서 읽어 게임마다 파일을
지웠음(러너는 게임마다 새 에이전트) → `os.environ.pop`으로 프로세스당 1회로
수정. 수정 전 런은 폐기.

## 결론 / 다음 단계

- 현 형태(9b 자유 서술 평가자 → md → 프롬프트 주입)는 **폐기**. 배관과
  `InsightMemory`는 재사용 가능
- 다음: (1) 인사이트를 구조화된 액션→효과 사실로 강제하고 인코더 로그와 대조해
  검증된 것만 저장, (2) 보상은 코드로(새 상태·레벨업), LLM은 인사이트만,
  (3) 평가자만 35b로 두는 비대칭 구성 시험
