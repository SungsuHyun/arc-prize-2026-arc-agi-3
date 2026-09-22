# v007-self-reflect-md

## 가설

9b 플래너(v004)가 실패한 이유는 "계획 품질"이었다. 모델을 키우는 대신,
몇 턴마다 스스로의 행동을 **보상 평가자 프롬프트**(사용자 제공 템플릿)로
채점하고, 그 결과(보상·평가 기준 개정안·게임/범용 인사이트)를 **마크다운
파일(insights.md)** 에 누적한 뒤 다음 계획 프롬프트에 주입하면, 처음 보는
게임에서도 이전 게임에서 얻은 교훈으로 계획이 좋아질 것이다.

## 변경점 (v004 대비)

- `InsightMemory`: `memory/insights.md` 파싱/렌더/저장. 섹션 = Evaluation
  Criteria(vN, 개정 이력) / General Insights(≤12) / Game Notes: <game>(≤8) /
  Reward Log(≤20) / Criteria History
- `RewardEvaluator`(L3): REFLECT_EVERY=25 액션마다 최근 행동 창을 평가자
  템플릿에 넣어 9b 호출 → `reward_score`, `template_update_suggestion`
  (needs_update=true면 기준 문구 추가 + 버전 bump), `game_insights`,
  `general_insights`(출력 JSON에 두 필드 추가) → md 갱신·저장
- 플래너 system prompt에 `[INSIGHTS: general]` + `[INSIGHTS: this game]` +
  `[LAST EVAL]` 주입
- 보상 ≤ -0.2 이면 계획 큐 폐기 + 즉시 재계획(force_plan)
- metrics: reflect_calls/errors/time, rewards, mean_reward, criteria_updates,
  criteria_version, general/game insight 수, reflect_samples(raw JSON 일부)
- 파일은 게임·런을 넘어 지속(INSIGHTS_RESET=1 로 초기화). run1은 초기화 후,
  run2는 run1의 md를 이어받아 실행

## 관찰

동일 조건(ls20, lf52, sk48, vc33 × 400) 2런:

| run | 메모리 | 종합 점수 | 레벨 | 평가자 호출/게임 | 평균 보상 | 기준 버전 |
|---|---|---|---|---|---|---|
| run1-fresh-memory | 빈 파일에서 시작 | 0.000 | 0 | 15–16 (오류 0) | -0.79 ~ -1.00 | v1→v3 |
| run2-carried-memory | run1 파일 이어받음 | 0.000 | 0 | 15–16 (오류 0) | -0.79 ~ -1.00 | v3 (변화 없음) |

비교: v003(L1만) 0.010, v004 9b 0.0004(vc33 레벨 1), v004 35b 0.321.

- 파이프라인 자체는 안정: 게임당 평가자 16회 + 플래너 12회, JSON 파싱 오류 0,
  LLM 시간 ~60–70초/게임. `insights.md`가 게임·런을 넘어 정상 누적됨
  (버그 수정 후 — 처음엔 INSIGHTS_RESET이 게임마다 파일을 지워 게임별 리셋됐음)
- **인사이트 품질이 병목**: 9b 평가자가 쓴 노트는 거의 전부
  "arbitrary resizing/oscillation without a clear sequence is futile" 류의
  부정 진술. 액션→효과 규칙(예: "ACTION1은 c4 객체를 위로 1칸")은 한 줄도
  없음. lf52의 "ACTION6로 c1 블록을 클릭해 제거하라"가 유일한 행동 지침이나
  검증 안 됨(run2에서도 레벨 0)
- **보상이 상수 -1로 붕괴**: 레벨 미완료 = -1로 채점하는 경향 → 창마다 재계획
  강제(force_plan)가 매번 발동해 신호로서 무의미. 기준 개정 제안도 자기
  강화(v2와 v3가 같은 문구: "반복 조작에 극도로 낮은 보상")
- 주입된 인사이트가 계획을 바꾸지 못함: planned_actions 21–93/401, 나머지는
  L1 폴백. 부정 진술만 있으니 플래너가 "무엇을 할지"를 얻지 못함
- 근사 중복 필터(토큰 Jaccard ≥0.6)에도 유사 문장이 다수 통과 → 8개 한도가
  같은 말로 채워짐

## 결론

- **폐기(현 형태)**: 9b 평가자의 자유 서술 인사이트는 계획을 개선하지 못하고,
  보상은 정보량이 없음. 점수 0.0 (v004 9b/v003보다 나쁨)
- 살릴 것: InsightMemory(md 파싱/렌더/주입) 인프라, 평가자 호출 배관
- 후속 아이디어:
  1. 인사이트를 자유 문장이 아니라 **구조화된 사실**로 강제
     (`{"action": "ACTION1", "effect": "moves c4 up 1"}`) — 인코더의 [RECENT]에서
     프로그램적으로 추출한 것과 대조해 검증된 것만 저장
  2. 보상을 LLM이 아닌 코드로 계산(새 상태 수, 레벨업)하고 LLM은 "기준 개정 +
     인사이트"만 담당 — 붕괴 방지
  3. 평가자만 35b로 (계획은 9b) 하는 비대칭 구성 — 인사이트 품질이 병목이므로
  4. 기준 개정 제안에도 근사 중복 필터 적용
