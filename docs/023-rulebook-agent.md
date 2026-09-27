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

## 레벨 2 실패 분석 (실행 4, 2026-09-27)

레벨 2에 도달한 세 게임(tn36·lp85·r11l)의 로그·룰북을 게임 소스(environment_files, 공개 게임)와 재생 실험으로 대조했다.
공통 결론을 먼저 쓰고 게임별 근거를 붙인다.

### 공통 결론

1. **레벨 1은 "진짜 규칙"을 모른 채 깨졌다.** 세 게임 모두 레벨 1 해법은 진짜 규칙의 퇴화 사례였고, 룰북은 그 퇴화 사례를 규칙으로
   일반화했다(tn36 "색 1을 모두 없애고 9 클릭", r11l "색 15를 모두 수집", lp85 "위 줄 색 5 표식에 맞추기"). 하네스의 승리 사실
   추출(`goals.infer`: 사라진 색·클릭 색 순서)도 같은 상관관계를 사실로 승격했다(W: collect_all 1 / click_sequence [15…]).
2. **룰북의 존재론이 색 단위다.** 세 게임의 레벨 2는 모두 "물체의 역할"(템플릿 vs 편집 가능 표식, 앵커 vs 조각 vs 구멍, 틀 vs 블록)로
   결정되는데, 룰북 항목·하네스 사실·후보 라벨이 모두 `colour k`로만 말한다. 같은 색이 두 역할을 가지면(tn36의 색 1, r11l의 색 15)
   규칙이 서로 모순되어 predictor는 "예측 불가"로 물러나고 모델은 색 이름만 보고 반복한다.
3. **목표를 만족했는지 알 길이 없다.** 승리 조건은 항상 "레벨이 끝났다"로만 관측된다. tn36 레벨 2에서 4번의 올바른 토글 뒤 제출을
   눌러야 했는지, lp85에서 블록이 틀에 들어갔는지, r11l에서 조각이 구멍에 닿았는지를 하네스가 판정하지 못하므로 모델은 진행
   신호 없이 예산을 태운다(tn36: 96액션 중 70이 HUD만 바뀐 무효 클릭, 67회가 읽기 전용 패널).
4. **검토가 룰북을 고치지 못했다.** 불일치 검토(no-think, 7초)는 잘못된 항목에 조건을 덧붙이거나("some clicks may trigger a move"),
   이미 하네스가 확정한 항목을 다시 쓰는 데 그쳤다. 새 개념("이 패널은 읽기 전용", "색 3 덩어리는 다른 앵커")을 만든 적이 없다.
   레벨당 검토 8회 상한과 distrust 이후 predictor 침묵으로, 레벨 2 중반부터는 배우는 것이 없다.
5. **탐색이 아니라 반복이다.** 세 게임 모두 레벨 2에서 한 종류의 후보만 눌렀다(tn36 색 1 91/96, lp85 버튼 8 32/42, r11l 색 15 39/39).
   "이미 시도했고 효과 없음" 거부는 같은 세계 상태에서만 작동하는데, 무효 클릭 사이에 다른 토글이 끼면 상태가 바뀌어 거부가 풀린다.

### tn36 — 디지트 프로그램 퍼즐

**진짜 규칙(소스+실험).** 하단 두 패널의 표식(색 1=켜짐, 5=꺼짐)은 7세그먼트식 숫자 셀이고, 각 숫자는 명령이다(소스의 op 표:
1=왼쪽 이동, 2=오른쪽, 3=아래, 5/6/7=회전). 왼쪽 하단(색 2 배경)은 **읽기 전용** 프로그램(태그 reooao → 클릭이 거부됨), 오른쪽 하단
(색 0 배경)이 편집 가능한 프로그램. 상단 왼쪽 패널의 색 4 조각은 프로그램 실행으로 움직이고(하단 작은 색 11 버튼 (58,8) 클릭 →
조각 (17,23)→(17,7)로 이동을 실험으로 확인), 승리 판정은 **조각의 x·y·scale·rotation·상태가 목표와 같은가**(`vklyonlcrw`). 상단 오른쪽
패널의 색 11 도형이 목표 포즈. 색 9 큰 버튼 = 실행/제출. 왼쪽 상단 패널은 클릭마다 1px씩 왼쪽으로 흐르는 타이머로, 벗어나면 게임오버.

**레벨 1이 왜 깨졌나.** 레벨 1의 편집 패널은 한 줄, 필요한 프로그램은 "표식 6개를 끄기"였다. 승리 열 [11,1,1,1,1,1,1,5,4,9] 중 11·5·4
클릭은 무효(HUD만), 실제 해법은 표식 6개 토글 + 9. 룰북은 이를 "색 1 물체를 모두 수집한 뒤 색 9 클릭"으로 기록했다.

**레벨 2에서 일어난 일.** 룰북(레벨 2 시작 시점)의 계획은 "남은 색 1을 전부 수집 → 9". 처음 4번(편집 패널 첫 줄 토글)은 예측
"물체 소멸→색 5"가 OK로 검증됐다. 5번째부터 읽기 전용 패널의 색 1을 눌렀고(HUD만 변함 → 불일치 → 검토 2회 → distrust), 이후 91번
중 67번을 읽기 전용 패널에 썼다. 편집 패널의 다른 줄도 24번 토글해 프로그램을 망가뜨렸다. 제출(9)은 4번 눌렀으나 프로그램이
틀려 무효. 재생 실험: 첫 4토글 직후 제출해도 통과되지 않았다 → "두 패널 일치"도 정답이 아니며, 목표 포즈를 만드는 프로그램을
찾아야 한다(레벨 2는 두 줄 이상의 명령).

**룰북이 틀린 지점.** E4 "색 11 = 아바타/버튼"(맞는 건 하단 작은 11 = 실행 버튼, 상단 11 = 목표 도형), E5 "색 1 = 수집품"(표식),
R2/R6/R10 "색 1 클릭 = 제거"(토글이며 읽기 전용 패널은 무효), W4 "색 1 전부 수집 후 9"(프로그램 = 목표 포즈). 검토는 R10에
HUD 서술을 덧붙이고 E8(HUD 색 3 설명), E9("아바타 = 색 4 블록, 색 11로 순간이동")를 추가했을 뿐 "읽기 전용 패널"을 끝내 못 봤다.

### lp85 — 순환 컨베이어 퍼즐

**진짜 규칙.** 2×2 블록들이 이름 붙은 경로(맵)를 따라 놓여 있고, 버튼(색 8 = 한 방향, 색 14 = 반대 방향)을 누르면 그 버튼의
맵에 속한 블록 전부가 경로상 다음 칸으로 한 칸씩 이동(순환)한다. 승리: 색 11 모서리 틀(스프라이트 bghvgbtwcb)의 안에 색 11 2×2
블록("goal")이, 색 12 틀 안에 색 12 블록이 있을 것. 실험: (17,20)의 8 버튼은 하단 루프(37–44행)의 블록 6개를 한 칸씩 돌렸고 (26,14)·
(35,14)의 8 버튼은 각각 다른 루프를 돌렸다(같은 색 블록이 많아 diff는 "이동"을 못 잡고 33–41셀 변화로만 보임).

**레벨 1.** 루프 하나·버튼 두 개. 승리 열 [11,11,11,8,14,8,8,8,8,14,8,8]: 11 클릭 3회는 무효, 나머지는 8을 눌러 돌리다 블록이 틀에
들어간 순간 완료. 룰북은 "8 = 줄을 오른쪽으로 4칸 이동, 14 = 왼쪽", "색 11 조각은 정적 표식·무효", W1 "위 줄 색 5 표식(사실은
레벨 진행 HUD)에 블록 맞추기"로 기록.

**레벨 2.** 루프 3개·버튼 6개·틀 2개·색 11 블록 2개, 행동 예산 60. 42액션 동안 8을 32번, 14를 10번 눌렀고 검토 5회는 R14/R16/R17/
R19/R20/R23 사이에서 "줄 이동" 문구를 고쳐 쓰는 데 소모됐다(하네스 오프셋 모델은 순환 이동을 표현하지 못해 2회 후 distrust).
필요한 것은 (a) 틀 ↔ 같은 색 블록의 짝, (b) 버튼별 순열(각 블록의 다음 칸 — 한 번 누르면 물체 추적으로 얻어짐), (c) 예산 안에서
두 블록을 동시에 틀에 넣는 짧은 누름 수열 탐색. 셋 다 룰북·predictor에 없다.

### r11l — 앵커 무게중심 퍼즐

**진짜 규칙.** 색 3 12px 덩어리들이 **앵커**(선택된 앵커는 색 0으로 표시), 앵커 무리의 무게중심에 **조각**(색 12 또는 15 다이아몬드,
중심 색 6)이 놓이며 앵커–조각 사이에 색 1 점선(tether)이 그려진다. 앵커 덩어리를 클릭하면 선택이 바뀌고, 빈 곳을 클릭하면 선택된
앵커가 그리로 이동(색 10 벽과 충돌하면 거부)하고 조각은 그만큼의 평균만큼 움직인다. 승리: 각 조각이 같은 색 점선 다이아몬드
(구멍)에 정확히 놓일 것. 실험: (35,45) 덩어리 클릭 → 선택 표시 (6,17)→(35,45); (30,45) 클릭 → 앵커 -5행 이동, 조각 (41,49)→(39,49).

**레벨 1.** 앵커 2개·조각 1개·구멍 1개. 구멍의 점(색 15)을 계속 클릭하면 선택된 앵커가 구멍 위로 가고, 결국 무게중심(조각)이
구멍에 들어가 완료 — "색 15를 모두 클릭"처럼 보였다. 룰북: E5 "색 15 = 수집품", R14(하네스) "색 15 클릭 → 색 0 마커가 클릭 지점으로
이동"(절반만 맞음: 마커 = 선택된 앵커), R5/R13 "색 3은 무관"(**refuted로 지워 버린 것이 바로 다른 앵커**), W5 "색 15 전부 수집".

**레벨 2.** 조각 2개(앵커 2개 + 3개), 벽 미로. 39액션 전부가 오른쪽 위 구멍의 색 15 점 클릭 — 선택된 앵커 하나만 점 사이를
오갔고 조각은 진동. 다른 앵커(색 3 덩어리)를 한 번도 선택하지 않았다(룰북이 색 3을 무관하다고 확정했기 때문). 두 앵커를 모두
구멍 위로 옮기면 되는 문제였으므로, 레벨 1 해법의 올바른 일반화("앵커마다 선택 후 구멍으로")를 룰북이 담았다면 풀렸다.

### 무엇을 바꿔야 하나 (분석에서 직접 나오는 것)

- **역할 단위 항목**: 룰북·사실·후보를 "색 k"가 아니라 "물체 집단(위치·크기·패널)"로 쓴다. 같은 색이라도 패널이 다르면 다른 항목.
  하네스는 클릭 효과 통계를 (색, 영역) 단위로 나눠 "읽기 전용 패널"을 스스로 발견할 수 있다(tn36: 왼쪽 패널 색 1은 67/67 무효).
- **승리 조건은 실행으로 검증되는 술어여야 한다**: 레벨 완료 직전 보드에서 "A가 B 안에 있다", "조각이 구멍 윤곽과 겹친다", "두 영역이
  같다" 같은 관계 술어 후보를 뽑고, 레벨 2에서 그 술어가 만족되는 순간을 progress 신호로 모델에 보여 준다.
- **레벨 1 해법의 일반화를 검증하는 검토**: 레벨 완료 검토(thinking)에 "승리 열에서 무효 클릭을 빼고, 남은 클릭이 어떤 물체를 어떻게
  바꿨는지"를 하네스가 표로 주고, 모델이 "다음 레벨에서도 통하는 절차"를 쓰게 한다. 무효 클릭이 승리 열에 섞이는 것을 하네스가
  먼저 걸러야 한다(tn36의 11·5·4, lp85의 11×3).
- **순열형 효과 모델**: 클릭 후 물체별 변위 표(어떤 물체가 어디로)를 predictor에 추가(lp85 버튼, r11l 앵커 이동은 이것으로 정확히
  예측된다). 오프셋 모델은 국소 토글 전용으로 남긴다.
- **반복 거부를 세계 상태가 아니라 "그 물체에 대한 같은 시도"로 판정**하고, 한 레벨에서 무효 클릭이 N회를 넘으면 후보에서 뺀다.

### 부록: 레벨 2 종료 시점 룰북 (실행 4)

#### tn36

```
ENVIRONMENT (what is on the board):
  [E1][?] Colour 5 forms the outer border and background.
  [E2][?] Colour 0 defines the main playfield area.
  [E3][?] Colour 4 tiles form a grid of interactive cells.
  [E4][?] Colour 11 objects act as the player avatar or primary interactive buttons.
  [E5][?] Colour 1 objects are collectible targets or items.
  [E6][?] Colour 9 at the bottom is the goal zone.
  [E7][?] Colour 9 strip at row 1 is an action counter or HUD.
  [E8][?] Colour 3 on the HUD strip (row 1) represents the remaining action count or collected items, shifting as the counter (colour 9) changes. It is part of the HUD, not the playfield.
  [E9][OK] The avatar is the colour-[4] block, 4x4 px. It can be moved by clicking colour 11 objects (teleport). [2 for / 0 against] — induced from transitions
RULES (what actions do):
  [R1][X] Clicking a colour 11 object activates a state change or resets state, decrementing the HUD counter. It does not move an avatar. <click {"color":11}> [1 for / 0 against] — Clicking colour 11 does not just change the HUD; it teleports the avatar (colour 4) to the clicked position. Evidence: a105 clicked 11 at (58,8) and avatar moved from (17,23) to (17,7) (approximate cl
  [R2][OK] Clicking a colour 1 object collects it or removes it from the board. <collect {"color":1}> — Clicking colour 1 removes the object from the board.
  [R3][?] Clicking a colour 4 tile changes its colour or state. <click {"color":4}> [1 for / 0 against] — Clicking colour 4 also decrements the HUD counter, similar to clicking colour 11 or 1. It is an action that consumes a move.
  [R4][?] Colour 5 blocks movement or interaction. <wall {"color":5}>
  [R5][OK] The top colour 9 HUD decreases by 1 per action. <gauge {"color":9,"per_action":-1}> [3 for / 0 against] — HUD decreased by 1 after clicking colour 11, confirming it tracks actions.
  [R6][OK] Clicking a colour 1 object removes it from the board and decrements the HUD counter by 1. The HUD strip (row 1) shifts: the block of 9s shrinks by 1 pixel from the right, and the block of 3s grows by 1 pixel from the left (or shifts right). Note: Some clicks on colour 1 may trigger a 'move' effect o <collect {"color":1}> [6 for / 0 against] — induced from transitions
  [R7][?] clicking colour 5 changes only the HUD/counter (1 clicks) <click {"color":5,"changes":[]}> [1 for / 0 against] — induced from transitions
  [R8][OK] clicking colour 9 changes only the HUD/counter (3 clicks) <click {"color":9,"local":136,"changes":[]}> [3 for / 0 against] — induced from transitions
  [R10][OK] Clicking a colour 1 object removes it from the board (turns to 5 or disappears) and decrements the HUD counter by 1. The HUD strip (row 1) shifts: the block of 9s shrinks by 1 pixel from the right, and the block of 3s grows by 1 pixel from the left (or shifts right), maintaining the total width of t <collect {"color":1,"effect":"remove","hud_change":true,"hud_shift":{"row":1,"color_9_delta":-1,"color_3_delta":1,"direction":"right"}}> [5 for / 0 against] — induced from transitions
  [R11][OK] clicking colour 1 changes 1 cells around the click point (1->5x1) <click {"color":1,"local":1}> [10 for / 0 against] — induced from transitions
  [R12][?] Clicking a colour 4 tile decrements the HUD counter by 1 (same as other actions). <click {"color":4,"hud_change":true}>
  [R13][?] The game ends in failure if the HUD counter (colour 9) is depleted to zero before winning. <gauge {"color":9,"per_action":-1,"limit":0,"effect":"game_over"}>
  [R14][?] Clicking a colour 11 object teleports the avatar (colour 4) to the clicked location and decrements the HUD counter. <click {"color":11,"effect":"teleport_avatar","hud_change":true}>
WIN CONDITIONS (how a level is completed):
  [W1][X] Win by collecting all colour 1 objects. <collect_all {"collect":1}> — Winning sequence ended with clicking colour 9, not just collecting 1s. Collecting 1s was a prerequisite, but the final click on 9 triggered the win.
  [W2][X] Win by moving the avatar to the bottom colour 9 goal. <reach {"reach":9}> — No avatar movement occurred. The win was triggered by a click action on the goal zone (colour 9), not by reaching it with an avatar.
  [W3][X] Win by clicking colour 11, then colour 1, then colour 9 in sequence. <click_sequence {"sequence":[11,1,9]}> — Refuted by harness evidence W4.
  [W4][?] Win by clicking the colour 9 goal zone after all colour 1 objects are collected, provided the HUD counter (colour 9) has not reached zero. Note: Avatar position is irrelevant; only collection status and counter matter. <collect_reach {"collect":1,"reach":9,"condition":"gauge > 0"}> [1 for / 0 against] — induced from transitions
PLAN: 1. Identify all remaining colour 1 objects on the board. 2. Click each colour 1 object to collect them. Monitor the HUD counter (colour 9 width). 3. If the avatar is far from a target colour 1, click a nearby colour 11 to teleport the avatar closer (if movement is required for some reason, though clicking seems global). 4. Ensure the HUD counter does not reach zero before all colour 1 objects are collected. 5. Once all colour 1 objects are removed and the counter is still > 0, click the large colour 9 strip (goal zone) to win the level.
```

#### lp85

```
ENVIRONMENT (what is on the board):
  [E1][?] Colour 4 forms the main walkable floor area.
  [E2][?] Colours 3 and 14 form static borders and walls.
  [E3][?] Colours 1,2,9,10,11,15 are 4x4 puzzle blocks.
  [E4][?] Colour 5 marks target positions at row 1. These markers shift right by 4 columns when colour 8 is clicked, and left by 4 columns when colour 14 is clicked.
  [E5][?] Left column (col 0) is a HUD/gauge strip in colour 14.
  [E6][?] Player avatar is not visible; likely selected by first MOUSE click.
  [E7][OK] the avatar (what the movement keys move) is the colour-[2] block, 2x2 px, now at rows 23-24 cols 23-24 <avatar {"colors":[2]}> [2 for / 0 against] — induced from transitions
  [E8][?] Colour 11 objects are small 4x4 fragments scattered on the board. They do not move when rows/columns shift (or move with the grid if they are part of the background, but evidence shows they stay or vanish). In Level 1 win, they were clicked 3 times at the start.
RULES (what actions do):
  [R1][?] Clicking a 4x4 block of colour 1 shifts the entire row containing the block to the right, wrapping around. Blocks move to the next empty slot in the row. <click {"color":1,"shift":"row_right"}> [1 for / 0 against] — Clicking colour 1 does nothing; it is a movable block, not a controller.
  [R2][?] Clicking a 4x4 block of colour 2 shifts the entire row containing the block to the right, wrapping around. <click {"color":2,"shift":"row_right"}> [1 for / 0 against] — Clicking colour 2 does nothing; it is a movable block, not a controller.
  [R3][?] Clicking a 4x4 block of colour 10 shifts the entire row containing the block to the right, wrapping around. <click {"color":10,"shift":"row_right"}> [1 for / 0 against] — Clicking colour 10 does nothing; it is a movable block, not a controller.
  [R4][?] Clicking a 4x4 block of colour 15 shifts the entire row containing the block to the right, wrapping around. <click {"color":15,"shift":"row_right"}> [1 for / 0 against] — Clicking colour 15 does nothing; it is a movable block, not a controller.
  [R5][?] Clicking colour 3 or 14 (walls) produces no change. <wall {"color":3}>
  [R6][?] Clicking colour 4 (floor) may select the active piece or do nothing. <click {"color":4}>
  [R7][?] Clicking colour 5 (top markers) may trigger a rotation or level transition. <click {"color":5}>
  [R8][OK] Clicking colour 11 (4px fragments) does nothing; they are static or inactive markers. <click {"color":11,"effect":"noop"}> [3 for / 0 against]
  [R9][?] Clicking a 4x4 block of colour 9 shifts the entire row containing the block to the right, wrapping around. <click {"color":9,"shift":"row_right"}> [1 for / 0 against] — Clicking colour 9 does nothing; it is a movable block, not a controller.
  [R11][OK] the gauge (colour 14, edge strip) changes by -1 per action; about 20 actions left <gauge {"color":14,"per_action":-1}> [3 for / 0 against] — induced from transitions
  [R12][?] clicking colour 14 changes 24 cells around the click point (9->15x8, 10->9x8, 2->10x8) <click {"color":14,"local":24}> [1 for / 0 against] — induced from transitions
  [R13][X] Clicking a 4x4 block of colour 11 shifts the entire column containing the block downwards, wrapping around. <click {"color":11,"shift":"col_down"}> — Clicking colour 11 does nothing; it is a static/movable marker, not a controller.
  [R14][X] Clicking colour 8 shifts the entire row containing the clicked block to the right by 4 columns, wrapping around. It also shifts the top row markers (colour 5) to the right by 4 columns. The gauge (colour 14) decreases by 1. <click {"color":8,"shift":"row_right_4","side_effect":"shift_top_markers_right_4","gauge_cost":1}> [4 for / 0 against] — Clicking colour 8 does not shift all blocks globally. It only shifts the specific row containing the clicked colour 8 block to the right by 4 columns, wrapping around. It also shifts the top row marke
  [R15][OK] Clicking colour 14 (the special block) shifts all movable blocks in the same column downwards. <click {"color":14,"shift":"col_down_all"}> [6 for / 0 against] — Clicking colour 14 does not shift columns down; it shifts rows left.
  [R16][X] Clicking colour 14 (the special block at col 58) shifts all movable 4x4 blocks in the same row to the left, wrapping around. <click {"color":14,"shift":"row_left_all"}> — Duplicate/incorrect. Replaced by R20 refinement.
  [R17][X] Clicking colour 14 shifts all movable 4x4 blocks in the same row to the left, wrapping around. <click {"color":14,"shift":"row_left_all"}> — Duplicate/incorrect. Replaced by R20 refinement.
  [R18][?] Clicking colour 11 (4px fragments) does nothing. It is likely a target marker or static object. <click {"color":11,"effect":"noop"}>
  [R19][X] Clicking colour 8 shifts all movable 4x4 blocks in the same row to the right by 4 columns, wrapping around. It also shifts the top row markers (colour 5) to the right by 4 columns. <click {"color":8,"shift":"row_right_4","side_effect":"shift_top_markers_right_4"}> — Clicking colour 8 shifts rows right, but the previous entry lacked gauge cost and marker shift details found in R14/R20. R14 is also partially wrong about 'all blocks globally' but correct about the r
  [R20][OK] Clicking colour 14 shifts the entire row containing the clicked block to the left by 4 columns, wrapping around. It also shifts the top row markers (colour 5) to the left by 4 columns. The gauge (colour 14) decreases by 1. <click {"color":14,"shift":"row_left_4","side_effect":"shift_top_markers_left_4","gauge_cost":1}> — Consistent row-left-4 shift for colour 14 clicks confirmed by previous actions a35, a37, a39.
  [R21][?] Clicking colour 1, 2, 9, 10, 11, 15 does nothing; they are movable puzzle blocks. <click {"colors":[1,2,9,10,11,15],"effect":"noop"}>
  [R22][?] Clicking a movable block (colours 1, 2, 9, 10, 11, 15) does nothing; they are not controllers. <click {"colors":[1,2,9,10,11,15],"effect":"noop"}>
  [R23][OK] Clicking colour 8 shifts the entire row containing the block to the right by 4 columns, wrapping around. It also shifts the top row markers (colour 5) to the right by 4 columns. The gauge decreases by 1. Note: The specific row shifted is the one containing the clicked colour 8 block. — Clicking colour 8 shifts the specific row containing the clicked block right by 4 columns. Confirmed by consistent observations in a41, a44, a45, a46, a47, a48, a49, a50.
WIN CONDITIONS (how a level is completed):
  [W1][?] Level completes when all movable blocks (colours 1, 2, 9, 10, 11, 15) are aligned with the target markers (colour 5) in the top row. The markers move with row shifts, so the goal is to position blocks under the markers. <collect_reach {"collect":[1,2,9,10,11,15],"reach":"match_top_row_markers"}>
  [W2][?] Win by collecting all blocks of colour 1. <collect_all {"collect":1}>
PLAN: 1. Identify the current positions of the target markers (colour 5) in the top row. 2. Identify the rows containing movable blocks that need to be aligned with these markers. 3. Use colour 8 blocks to shift rows right and colour 14 blocks to shift rows left. Each click shifts the row by 4 columns and moves the markers by 4 columns in the same direction. 4. Calculate the required number of shifts for each row to align the blocks with the markers. Note that shifting a row also shifts the markers, so the relative alignment depends on the net shift. 5. Monitor the gauge (colour 14) to ensure action
```

#### r11l

```
ENVIRONMENT (what is on the board):
  [E1][?] Avatar is colour 6 at (47,17).
  [E2][?] Walls are colour 2.
  [E3][?] Floor/background is colour 5.
  [E4][?] HUD counter is colour 0 in column 0.
  [E5][?] Collectibles are colour 15 (f).
  [E6][?] Path guide is colour 1.
  [E7][?] Goal/exit is colour 3 cluster at (59,27).
  [E8][X] Key/item is colour 0 cluster at (36,7). — Colour 0 cluster is not a key/item; it is part of the avatar's trail or state.
  [E9][OK] the avatar (what the movement keys move) is the colour-[1, 6, 12] block, 5x7 px, now at rows 13-17 cols 36-42 <avatar {"colors":[1,6,12]}> [2 for / 0 against] — induced from transitions
  [E10][?] Remaining collectibles are at (22,36), (22,42), (23,37), (23,41), (24,38), (24,40). <observation>
RULES (what actions do):
  [R1][?] Colour 2 blocks movement. <wall {"color":2}>
  [R2][?] Clicking colour 15 collects it. <collect {"color":15}>
  [R3][X] Clicking colour 1 moves avatar along the path. <click {"color":1}> — Clicking colour 1 does not move avatar; avatar moves by clicking its own colour (6).
  [R4][OK] Clicking colour 6 moves the avatar to the clicked location. Clicking colour 15 also moves the avatar and causes significant board changes. <click {"color":6,"effect":"teleport_avatar_to_click"}> [7 for / 0 against] — Clicking colour 6 does not just select/avatar; it moves the avatar to the clicked location.
  [R5][X] Clicking colour 3 triggers the win condition. <click {"color":3}> — Clicking colour 3 does not trigger win; it swaps with 15 or is irrelevant.
  [R6][X] Clicking colour 0 cluster picks up a key. <click {"color":0}> — Clicking colour 0 does not pick up a key; colour 0 is part of the avatar's trail/state or irrelevant to movement.
  [R7][?] Clicking colour 5 floor does nothing. <noop {"action":"MOUSE"}>
  [R8][X] Mouse clicks move avatar to adjacent cells. — Mouse clicks teleport avatar to clicked coordinates, not adjacent cells.
  [R9][OK] the gauge (colour 0, edge strip) changes by -1 per action; about 25 actions left <gauge {"color":0,"per_action":-1,"limit":0,"effect":"game_over"}> [3 for / 0 against] — induced from transitions
  [R10][OK] Clicking the avatar (colour 6) moves it to the clicked location. If a collectible (colour 15) is at the target, it is collected. <move {"action":"CLICK","target_color":6,"effect":"teleport_to_click","collect_if_present":15}> — Clicking the avatar (colour 6) moves it to the clicked location, as seen in previous actions a1 and a2.
  [R11][X] The avatar carries the key (colour 0) if it was previously collected/adjacent, or the key moves with the avatar. — Avatar does not carry a key; colour 0 objects are separate trail markers or state indicators.
  [R12][?] When the avatar moves to a cell containing a collectible (colour 15), the collectible is removed from the board. <collect {"color":15,"condition":"avatar_moves_to_cell"}>
  [R13][X] Clicking colour 3 triggers the win condition if all collectibles are collected. <click {"color":3,"effect":"win_if_all_collected"}> — Clicking colour 3 does not trigger win condition.
  [R14][OK] clicking colour 15 moves the colour-0 marker to the clicked cell (35/37 clicks); clicking colour 15 changes 18 cells around the click point (5->0x12, 15->0x8, 1->0x6, 5->1x5) <click {"color":15,"local":18,"marker":0}> [37 for / 0 against] — induced from transitions
  [R16][?] Collecting a colour 15 object is done by clicking it, which also moves the avatar. <collect {"color":15,"method":"click"}>
  [R17][OK] Clicking a collectible (colour 15) teleports the avatar to that location, removes the clicked collectible, and causes a global board update where the path (colour 1) and remaining collectibles (colour 15) change positions. <click {"color":15,"effect":"teleport_avatar_and_global_update","moves_path":true,"moves_collectibles":true}> [33 for / 0 against] — Previous hypothesis was too weak. The effect is a global shuffle, not just a teleport and update.
  [R18][?] The game ends in failure if the gauge (colour 0 strip) empties. <hazard {"color":0,"condition":"empty","effect":"game_over"}>
  [R19][?] Clicking a collectible (colour 15) teleports the avatar to that location, removes the collectible, consumes 1 gauge unit, and triggers a global board shuffle where paths (1), remaining collectibles (15), and other objects (12, 3, 6) move to new positions. <click {"color":15,"effect":"teleport_avatar_and_global_shuffle","moves_path":true,"moves_collectibles":true,"moves_other":true,"consumes_gauge":1}>
WIN CONDITIONS (how a level is completed):
  [W1][X] Win condition: Collect all colour 15 objects. The level ends immediately upon collecting the last 15, regardless of avatar position or goal (3) status. <collect_all {"collect":15}> — Win condition is collecting all 15s, not reaching 3.
  [W3][X] Win condition: Collect all colour 15 objects. Reaching colour 3 is irrelevant. Gauge must remain > 0 until the last 15 is collected. <collect_reach {"collect":15,"reach":3,"constraint":"gauge > 0"}> — Reaching 3 is not required; collecting all 15s is sufficient.
  [W5][?] Win condition: Collect all colour 15 objects. Confirmed by Level 1 completion sequence [15, 15, 15, 15, 15, 15]. <collect_all {"collect":15}> [1 for / 0 against] — induced from transitions
PLAN: 1. Identify all remaining colour 15 objects on the board. 2. Click each colour 15 object exactly once. 3. Each click will teleport the avatar and shuffle the board (moving paths, other collectibles, and other objects). 4. After each click, re-scan the board to find the new locations of the remaining colour 15 objects. 5. Continue until all colour 15 objects are collected. 6. Monitor the gauge (colour 0 strip) to ensure it does not reach 0 before the last 15 is collected. 7. Ignore colour 3, colour 12, and other objects unless they interfere with visibility.
```

