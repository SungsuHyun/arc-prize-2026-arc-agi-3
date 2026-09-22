# v008-verified-insights

## 가설

v007의 병목은 9b 평가자가 쓰는 인사이트가 검증 안 된 부정 진술뿐이고 보상이
-1로 붕괴한 것. (1) 액션→효과를 **코드로 측정한 사실**을 프롬프트 최상단에
주고, (2) 보상도 코드로 계산해 트리거에 쓰고, (3) LLM 인사이트는 구조화해
실제 시도된 액션에 대한 것만 저장하면, 같은 9b로도 계획이 "무엇을 할지"를
얻어 나아질 것이다. 평가자만 35b로 바꾸는 비대칭 구성도 함께 본다.

## 변경점 (v007 대비)

- `ObservationEncoder.last_events`: 일반화 시그니처("moves c9 (-5,+0)",
  "grows c3", "removes c1", "no change")
- `ActionModel`: 액션 키별(ACTION6은 클릭 셀 색으로 `ACTION6@cN`) 시그니처
  카운트 → `facts()` = "ACTION3: moves c9 (-5,+0) 14/15". 스텝당 동일
  시그니처 중복은 1회
- `InsightMemory`에 "Verified Action Facts"(게임별, 덮어쓰기) + "Next
  Strategy" 섹션. `cross_game_facts()`로 "ACTION1: visible effect in 3/4
  games (typically moves)" 렌더. prompt_block 순서: ACTION MODEL → NEXT
  STRATEGY → INSIGHTS
- 코드 보상 `1.0*levelups + 0.4*novelty + 0.2*change - 0.5*(1-change)`.
  재계획 트리거(≤ -0.2)는 이 값만 사용. LLM reward_score는 로그 비교용
- 평가자 프롬프트에 `[Measured by code]` 블록. 출력 `game_insights`는
  `{action, fact}` 객체, 시도 안 된 액션은 폐기(dropped_insights).
  `next_strategy{action,target,why}` 추가
- 기준 개정 제안 근사 중복 필터(Jaccard ≥ 0.5 → 무시)
- `EVAL_MODEL` env로 평가자 모델 분리 (`_complete_raw(model=)`)

## 관찰

동일 조건(ls20, lf52, sk48, vc33 × 400), 빈 메모리에서 시작:

| run | 플래너/평가자 | 종합 | vc33 | 코드 보상 평균 | LLM 보상 평균 | LLM 시간/게임 |
|---|---|---|---|---|---|---|
| eval-9b-fresh | 9b / 9b | **0.0058** | 레벨 1 (0.023) | +0.39 ~ +0.62 | -0.20 ~ +0.51 | ~70–88s |
| eval-35b-fresh | 9b / 35b | 0.0017 | 레벨 1 (0.007) | +0.39 ~ +0.59 | -0.43 ~ +0.75 | ~140–160s |

비교: v007 0.000, v004 9b 0.0004, v003(L1만) 0.010, v004 35b 0.321.

- **검증된 액션 사실은 정확하고 유용**: ls20 "ACTION1~4: grows c3 / shrinks
  c11 ~95%"(이동 = 게이지 변화), sk48 "ACTION1: moves c6 (+0,-6)", "ACTION6@c*:
  no change"(클릭 무효), vc33 "ACTION6@c*: shrinks c7 / grows c4 100%"(어디를
  클릭해도 카운터 변화). v007엔 전혀 없던 종류의 정보
- **보상 붕괴 해소**: 코드 보상 +0.4~0.6, LLM 보상도 -1 고정에서 벗어남
  (9b: -0.2~+0.5). force_plan은 실제 정체 시에만 발동
- **그러나 해석이 틀림**: LLM이 게이지/카운터(c11, c7, c4)를 "목표 객체"로
  읽고 "가장 효율적인 한 액션에 수렴하라"는 전략을 반복 → 플래너가 같은
  액션을 반복(ls20 ACTION2, vc33 ACTION6@c7). 레벨 완료 조건(공간 목표)을
  찾는 탐색이 오히려 줄어듦
- 근사 중복 필터(0.6)가 긴 문장엔 약함: General Insights 12칸이 같은 뜻
  ("converge on the most efficient action")으로 채워짐
- 기준 개정이 게임 특정 색(c4, c7, 90%, 100px)으로 오염(9b: v12까지 bump)
- **35b 평가자는 이득 없음**: 인사이트 문장은 더 매끈하지만 같은 오해(게이지
  = 목표). GPU 모델 스왑으로 시간 2배. dropped_insights 9(vc33) — 시도 안
  한 액션 사실을 걸러낸 건 검증 로직이 작동한 증거

## 결론

- **부분 채택**: ActionModel(코드 측정 사실) + 코드 보상은 유지 가치 있음 —
  9b 계열 최고 점수(0.0058)이고 v007 대비 명확한 개선. 35b 평가자 비대칭은 폐기
- 여전히 v003(L1 탐색만, 0.010)보다 낮음: LLM 서술 인사이트 → "한 액션 반복"
  수렴이 탐색을 해침. 자유 서술 인사이트/next_strategy는 축소하거나 제거
- 후속:
  1. 사실 → 계획을 코드로: ActionModel에서 "이동 액션"(moves) 판별 후 목표
     후보(레벨업 직전 상태 등)로 BFS/탐색 — LLM은 가설 제안만
  2. 게이지 감지: 매 액션 같은 방향으로 grow/shrink하는 객체는 "카운터"로
     라벨링해 프롬프트에 명시(목표 아님)
  3. General Insights는 LLM 자유문 대신 cross_game_facts만; 기준 개정 제안에
     게임 특정 토큰(cN, px, %) 포함 시 거부
  4. 중복 필터 임계 0.6 → 0.35, 문장 길이 제한
