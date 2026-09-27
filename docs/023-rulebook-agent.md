# 023. 룰북 에이전트 — 가설 룰북 + 결정적 예측기 + 모델 선택 + 불일치 검토

- **날짜**: 2026-09-27
- **관련 파일**: `rulebook/` (새 패키지: env.py · book.py · predict.py · candidates.py · llm_io.py · agent.py · run.py),
  `scripts/run_rulebook.py`, `make rulebook`, 결과 `experiments/rulebook/results/`
- **선행 문서**: docs/021 §7(LLM 제안 → 기호적 귀납·검증 → 시뮬레이터 탐색), docs/022(0/1/2층 구조)

## 목적

기존 라인(매 스텝 LLM 행위자 `arcnav/agent.py`, 모델 없는 `explorer.py`, 매크로 자문 `advisor.py`)을 버리고,
사용자가 제시한 절차를 그대로 구현한 새 솔루션 라인을 만든다.

1. 게임 시작(레벨 1) 때 **룰북 가설**을 세운다 — 환경 사실 / 규칙 / 승리 조건을 항목 리스트로, Qwen thinking on.
2. 매 순간 **후보 액션(action1~N)마다 다음에 무슨 일이 일어날지를 프로그램이 결정적으로 예측**한다.
3. 예측이 붙은 후보 목록을 보고 **LLM이 어떤 액션을 할지 결정**한다(thinking off).
4. 액션을 수행하고 **기대와 다르면 룰북을 리뷰**한다(프로그램 채점 → LLM 수정, thinking on).
5. 증명된 가설은 따로 체크(confirmed), 틀린 가설은 그때마다 수정(refuted/edit).

## 다듬은 설계

### 룰북(`book.py`)

```
ENVIRONMENT  [E1][?] 아바타는 색 9/12의 5×5 블록 …
RULES        [R1][OK] UP moves the avatar by (dx=0, dy=-5) <move {"action":"UP","dx":0,"dy":-5}> [6 for / 0 against]
             [R4][X]  the avatar cannot enter colour 5 <wall {"color":5}> [1 for / 1 against] — evidence says …
WIN          [W1][?] reach the colour-12 door <reach {"reach":12}>
PLAN         …
```

- 항목 = {id, section(env/rules/win), text, kind, params, status(hypothesis/confirmed/refuted), support, counter, source(llm/harness), note}.
- **kind가 기계 검증 가능한 항목**(move/wall/noop/gauge/refill/collect/hazard/click, win: reach/collect_all/collect_reach/click_sequence)은
  params를 가지며, 하네스가 전이 기록으로 support/counter를 매기고 status를 **하네스가 결정**한다(support≥2·counter=0 → confirmed,
  counter≥support → refuted). LLM은 이런 항목을 confirm/refute할 수 없고 add/edit/remove만 가능하다.
- kind "other" 항목(자연어 가설)은 LLM 리뷰가 confirm/refute한다.
- `sync(facts)`: 하네스가 귀납한 사실과 룰북을 대조 — 같은 kind·키 params면 카운트 갱신, LLM 항목이 증거와 다른 params면
  refuted + 하네스 버전 추가, 룰북에 없는 사실은 `source=harness` 항목으로 추가.
- 편집 언어: `{"op":"add","section","text","kind","params"}`, `{"op":"confirm|refute|remove","id","note"}`, `{"op":"edit","id",...}`.
- 모든 변경은 `history`에 남고 `<game>.rulebook.json`으로 매번 저장된다.

### 결정적 예측기(`predict.py`)

전이 기록만으로 O(n)에 재구축되는 `Evidence`(옛 `arcnav.rules.induce`는 O(n²)라 버림):

| 사실 | 귀납 방법 | 예측 |
|---|---|---|
| move(action→dx,dy) | nav의 이동 로그(아바타 = 함께 움직이는 물체) | 아바타 bbox 이동 후 위치 |
| wall(colour) | 이동 실패 시 전방 셀의 다수 색(support), 같은 색에 성공적으로 진입한 횟수(counter) | blocked / 진입 성공 이력이 있으면 **undetermined**로 정직하게 보고 |
| hazard(colour) | 이동 후 시작 위치로 되돌아감 | "HAZARD" 예측 |
| collect(colour) | 아바타 도착 지점의 작은 물체 소멸 | 부수 효과로 보고 |
| gauge | 가장자리 띠의 단조 변화 | 액션당 변화량 |
| noop(action) | 보드 전체 무변화 | "nothing changes" |
| click: 물체 소멸 | 클릭한 물체의 셀이 모두 한 색으로 바뀌고 다른 플레이필드 셀은 불변 | 예측 보드(그 물체만 색 k로) |
| click: 위치 상대 효과 | `arcnav.effects.ClickModel`(오프셋별 전→후 부분 함수) | 예측 보드 |
| click: HUD만 / 무효 | 변화가 가장자리 3px 이내 또는 HUD 띠 뿐 | "HUD only" / "nothing" |

`check(prediction, transition)` → OK / MISMATCH / obs(예측 없음). 이동은 아바타 bbox 좌표 일치, 보드 예측은 HUD를 가린 보드 일치,
noop/HUD는 변화 분류(`change_class`: world / hud / none)로 판정.

### 후보와 결정(`candidates.py`, `llm_io.py`)

후보 = 이동 키 각각 / 키 8연타 / 상호작용 키 / 물체별 클릭(4×4 셀당 1개, 작은 것부터, 최대 24) / 타깃까지 걷기(goto) / 프론티어.
각 후보에 첫 스텝(걷기는 경로 시뮬레이션 종점)의 예측 문장이 붙고, 같은 세계 상태(가장자리 카운터 제외한 물체 목록 + 아바타 위치)에서
이미 시도한 후보는 `[tried here]` 표시. DECIDE 프롬프트 = 룰북 + 보드 + 후보표 + 최근 결과 10줄 → `{"choice","expect","edits"}`.
라벨이 무효면 결정적 대체 정책(미시도 우선, 우선순위순), 같은 라벨 3연속+이미 시도면 강제 교체.

### 검토 트리거(`agent.py`)

- MISMATCH: 즉시 후보 실행 중단 → 증거 재귀납·sync → REVIEW(thinking on, 변경 영역 전/후 크롭 포함). 레벨당 8회 상한, 이후는 프로그램 갱신만.
- 예측 없는 관찰(obs): sync만. 그 결과 LLM 항목이 refuted되면 REVIEW.
- 레벨 완료: `goals.infer`로 승리 사실 추출 → win 섹션 sync → REVIEW("어느 승리 조건이 증명됐나, 다음 레벨 계획").
- 게임 오버: 리셋 → REVIEW(위험 규칙 추가).

### 실행

```bash
make rulebook GAME=ls20,tn36 MINUTES=12 JOBS=2 TAG=x        # 모델 사용(vLLM :1234, qwen3.6-27b-awq)
.venv/bin/python scripts/run_rulebook.py --games ls20 --no-model --minutes 2   # 대체 정책만(스모크)
```

로그: `experiments/rulebook/results/logs/<run>/<game>.log`(모든 예측·판정·모델 호출), `<game>.rulebook.json`.

## 결과

로컬 vLLM(qwen3.6-27b-awq, 32k 컨텍스트), 게임당 12분, 레벨당 150액션, 2게임 동시.

| 실행 | ls20 | tn36 | 비고 |
|---|---|---|---|
| 1 (22:27 UTC, 첫 판) | 0레벨, 44액션, 검토 6회 | 0레벨, 80액션, DECIDE 13회 JSON 실패 | 검토(thinking) 60–145초 × 6 = 12분 중 10분 소모. `sync` 버그로 하네스 규칙이 중복 추가(항목 147개). 모델이 이미 시도한 클릭을 반복 |
| 2 (22:42 UTC, 수정 후) | 0레벨, 86액션, 검토 4회 | **레벨 1 완료 15액션(점수 3.571 = 만점)**, 24액션 | 승리 열 [1×6, 4×4, 11, 9]. 검토 thinking이 6000토큰 상한에 걸려 JSON 없음 → no-think 재시도(7초)가 대신 답함 |

실행 2에서 tn36 레벨 1을 깬 경위: INIT 룰북이 "색 1 = 도구/토큰, 색 9 삼각형 = 제출" 가설을 세움 → 색 1 클릭 6회(예측 "물체 소멸→색 5"
가 매번 OK로 검증) → 색 4 격자 클릭(예측 "HUD만 변함" OK) → 색 11 → 색 9 클릭으로 완료. 레벨 2에서는 같은 색 1 클릭이 다른 효과를
내어(4셀 차이) 불일치 5회 — 클릭 통계가 레벨 1 관측에 끌려간 것이 원인 → 현재 레벨 관측 우선으로 수정.

ls20은 두 실행 모두 미완료: 이동·벽·게이지·수집 규칙은 하네스가 전부 confirmed로 올렸지만(예측 OK 비율 높음), 모델의 계획이
"아래로/위로" 왕복이라 레벨 1(문 모양 맞추기)을 못 깸. 예측기는 "색 5 앞: 1회 막힘·1회 진입 — 조건 미결정"처럼 정직하게 보고한다.

### 실행 1→2 사이 수정

- `book.find`가 키만 같은 항목(모델의 틀린 params)을 먼저 잡아 하네스 규칙을 매번 새로 추가하던 버그 → 정확 일치 우선, 검증된 규칙이
  생기면 refuted된 모델 항목은 제거.
- 모델이 kind에 섹션명("rules")을 쓰고 진짜 kind를 params.action에 넣는 습관 → `normalize_kind`.
- 이미 시도했고 "변화 없음/HUD만"으로 예측된 후보를 다시 고르면 프로그램이 거부하고 미시도 후보로 대체.
- 클릭 후보를 색별 라운드로빈으로 뽑아(색 4 타일 36개가 목록을 독점하지 않게) 모델이 좌표로 쓴 라벨도 해석.
- 관찰(예측 없음)이 모델 항목을 refute할 때는 검토를 부르지 않음(하네스가 이미 고침).

### 실행 3 (6게임, 22:55 UTC, 게임당 12분·레벨당 150액션·2게임 동시)

| 게임 | 레벨 | 액션 | 점수 | 불일치/검토 | 관찰 |
|---|---|---|---|---|---|
| tn36 | 1/7 | 78 | 3.571 | 48/10 | 레벨 1을 **10액션**(만점). 레벨 2에선 색 1 클릭이 HUD만 바꾸는데 예측기가 레벨 1 통계("물체 소멸")를 계속 써서 불일치 반복 |
| lp85 | 1/8 | 96 | 2.778 | 5/7 | 레벨 1을 **16액션**(만점). 레벨 2에서 색 8 클릭 59회(효과 모델 없음) 후 게임오버 |
| r11l | 0/6 | 113 | 0 | 94/9 | 색 15 클릭 = 색 0 마커가 클릭 지점으로 이동 + 광역 변화. 오프셋 클릭 모델이 매번 틀림(94회) |
| vc33 | 0/7 | 52 | 0 | 9/9 | 색 9 클릭 40회 후 게임오버; 펌프 순서 퍼즐은 효과 모델 없음 |
| m0r0 | 0/6 | 52 | 0 | 1/1 | 거울 2체 게임: 예측은 맞지만(불일치 1) 모델의 계획이 진행을 못 만듦 |
| ls20 | 1/7 | 170 | 0.071 | 11/11 | 레벨 1을 155액션 만에 완료(게임오버 리셋 포함). 승리 사실 "색 5 물체에 진입" 추출 |

합계 1.07 (레벨 1 3/6, 레벨 2 0/6). 비교: arcnav 반복 14 같은 6게임 3.03(m0r0 레벨 2 자동조종 포함), 탐색 엔진 v6 레벨 1 6/25.

실행 3에서 드러난 예측기의 구멍과 수정(실행 4 전):
- 클릭 예측 순서: 현재 레벨에서 그 색 클릭이 HUD만 바꿨으면 그 사실이 이전 레벨의 "물체 소멸"보다 우선.
- 같은 색의 보드 예측이 한 레벨에서 2회 틀리면 그 색은 "예측 불가"로 보고(`distrust`) — 틀린 예측으로 검토를 소모하지 않음.
- 마커 규칙 추가: 클릭 후 어떤 색 물체가 클릭 지점으로 이동하면 "색 k 마커가 클릭 지점으로 이동" 예측·판정(r11l 색 6 클릭에서 OK 확인).

### 실행 4 (실행 3과 같은 6게임·조건, 위 수정 반영, 23:32 UTC)

| 게임 | 레벨 | 액션 | 점수 | 불일치(실행 3→4) | 관찰 |
|---|---|---|---|---|---|
| tn36 | 1/7 | 105 | 3.571 | 48 → 3 | 레벨 1 10액션. 레벨 2 95액션 미완료(예측은 맞으나 계획이 없음) |
| lp85 | 1/8 | 58 | 2.778 | 5 → 7 | 레벨 1 17액션 |
| r11l | **1/6** | 104 | 0.513 | 94 → 6 | 마커 규칙으로 예측이 맞기 시작, 레벨 1을 66액션에 완료 |
| vc33 | 0/7 | 81 | 0 | 9 → 2 | 미완료 |
| m0r0 | 0/6 | 96 | 0 | 1 → 1 | 미완료(거울 2체 계획 부재) |
| ls20 | 0/7 | 99 | 0 | 11 → 24 | 이번엔 미완료(실행 3의 155액션 완료는 리셋 포함 우연에 가까움) |

합계 **1.14**(실행 3: 1.07), 레벨 1 3/6, 레벨 2 0/6. 예측기 수정으로 불일치 총합 168 → 43. 검토 호출은 게임당 1–8회, 각 7초(no-think).

### 실행 2 이후 수정(실행 3 전)

- 검토 thinking은 레벨 완료 때만(`review_think: level`); 불일치·게임오버 검토는 no-think(7초).
- 클릭 통계·클릭 효과 모델을 현재 레벨 관측 우선으로.

## 결론 / 다음 단계

**되는 것.** 룰북 루프는 설계대로 돈다: INIT(thinking) 룰북 → 후보별 결정적 예측 → 모델 선택 → 액션별 검증 → 불일치 검토. 클릭 게임의
레벨 1은 기준 액션 이내로 깬다(tn36 10, lp85 16–17 = 만점). 룰북은 하네스가 채점한 사실(confirmed/refuted, for/against 수)과
모델의 자연어 가설이 한 문서에 공존하고, 모델의 틀린 params는 증거로 자동 refute된다. 예측이 맞는 비율이 높아질수록(실행 4) 검토
호출이 줄고 처리량이 는다(게임당 12분에 60–105 결정).

**안 되는 것.** 레벨 2는 0/6. 원인은 예측이 아니라 **계획**이다 — 모델(27B, no-think)은 "무엇을 순서대로 할지"를 룰북에서 끌어내지
못하고 국소 탐색을 반복한다(tn36 레벨 2 95액션, m0r0 거울 2체). ls20처럼 조건부 규칙(문 모양 일치)은 룰북 항목이 "미결정"으로만
남는다.

**다음.**
1. 계획 후보를 프로그램이 만든다: 확정된 승리 조건(reach/collect_all/click_sequence)에 대해 예측기 위에서 BFS/탐욕으로 짧은 계획을
   만들어 "plan(...)" 후보로 제시 — 모델은 실행 여부만 고른다(구 arcnav의 1층 계획기를 룰북 술어 위로 옮기는 것).
2. 레벨 완료 검토(thinking)에 "이전 레벨 승리 열의 일반화"를 요구하고, 그 결과를 click_sequence 규칙으로 하네스가 재생·검증.
3. 조건부 규칙 kind 추가(`wall_unless`: 색 c는 조건 X일 때만 통과) — 미결정 관측을 모델이 조건 가설로 쓰고 하네스가 채점.
4. 25게임 오프라인 측정(시드 고정 불가: 모델 온도 0.6 → 3회 반복)과 Kaggle 노트북 통합은 레벨 2가 1개라도 나온 뒤.
