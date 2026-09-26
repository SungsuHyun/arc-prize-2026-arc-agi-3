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

(아래에 실행 결과를 기록)
