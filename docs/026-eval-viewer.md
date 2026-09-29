# 026 — 로컬 평가 뷰어 (게임 → 레벨 → 스텝 재생)

- 날짜: 2026-09-28
- 종류: 인프라
- 관련: docs/023 (룰북 에이전트), docs/024 §재생 기반 검증

## 목적

룰북 실행 결과를 게임별·레벨별로 클릭해 가며 "현재 솔루션이 각 레벨에서 어떻게 움직였는지"를
보드 그림과 로그로 한 스텝씩 확인하는 별도 로컬 웹 서버. 기존 벤치마크 사이트
(experiments/site, GitHub Pages)와 분리되며 공개하지 않는다.

```bash
make eval-site            # http://0.0.0.0:8090/  (모든 인터페이스, EVAL_PORT=로 포트 변경, --host로 제한 가능)
```

## 데이터가 어디서 오는가

| 화면 | 출처 |
|---|---|
| run 목록, 게임·레벨 요약(완료/액션/점수/기준 액션) | `experiments/rulebook/results/run-<id>.json` |
| 스텝별 액션·예측·실제·룰북 변화·계획·리뷰 | `results/logs/<id>/<game>.log` (`[t a N Lk]` 태그 줄 단위로 묶음) |
| 스텝별 64×64 보드 | 저장돼 있지 않음 → **오프라인 엔진에 액션을 재생해서 복원** |
| 액션 시퀀스(새 실행) | `results/logs/<id>/<game>.actions.jsonl` — `rulebook/env.py`가 매 reset/step을 기록, 프레임 sha1 포함 |

## 검증

- 재생 결정성: 6개 게임(lp85, tn36, ar25, bp35, cd82, cn04)에서 재생 레벨 수가 원본과 일치.
  스텝당 약 1 ms, 게임당 0.1~0.4 s. 결과는 `results/cache/<run>/<game>.json.gz`에 캐시(gitignore).
- 로그 파싱 재생의 한계: coder 모드 일부 실행에서 액션 몇 개가 표준 액션 줄로 남지 않아
  ar25 235/239, cn04 360/361로 어긋남(레벨 결과는 동일). 이후 실행은 actions.jsonl이 있어 파싱 불필요.
- actions.jsonl 경로: `make rulebook GAME=lp85 MINUTES=1 NOMODEL=1` 스모크에서 80 스텝 재생, 해시 불일치 0.

## 구조

- `scripts/eval_viewer/replay.py` — 로그 파싱, actions.jsonl 로드, 재생, 2단 캐시, run 요약
- `scripts/serve_eval.py` — 표준 라이브러리 HTTP 서버 + JSON API (`/api/runs`, `/api/runs/<id>`,
  `/api/runs/<id>/games/<game>[?level=N]`, `.../rulebook`, `.../log`)
- `scripts/eval_viewer/index.html` — 단일 페이지: 왼쪽 게임/레벨 트리, 가운데 canvas 보드
  (변경 셀 외곽선, 클릭 위치, 타임라인, 재생, ←/→/Space), 오른쪽 스텝 상세와 그 뒤의 로그 이벤트.
  URL 해시 `#run/game/level/step`로 특정 스텝 공유 가능.

## 자동 재시작

`make eval-site`는 감시 프로세스가 서버를 자식으로 띄우고, `scripts/serve_eval.py`, `scripts/eval_viewer/`,
`rulebook/`, `arcnav/`의 .py/.html이 바뀌면 자식을 재시작한다(0.5초 폴링, 의존성 없음). 문법 오류로 죽으면
다음 수정까지 기다렸다가 다시 띄운다. 페이지는 2초마다 `/api/health`의 시작 시각을 확인해 서버가 바뀌면
스스로 새로고침하고 URL 해시로 같은 스텝으로 돌아온다. 디스크 재생 캐시는 유지된다.
단일 프로세스로 띄우려면 `--no-reload`.

## 부팅 시 자동 시작 (systemd user service)

`make eval-site-install`은 `scripts/systemd/arc-eval-site.service`를 `~/.config/systemd/user/`에 설치하고
`systemctl --user enable --now`로 바로 띄운다(포트는 `EVAL_PORT=`, 기본 8090). 사용자에게
`loginctl enable-linger`가 켜져 있으면 로그인 없이 서버 부팅 직후 시작되고, 죽으면 5초 뒤 재시작된다.
서비스도 위의 감시 프로세스를 그대로 쓰므로 소스를 고치면 자동 반영된다. 로그는
`journalctl --user -u arc-eval-site -f`, 제거는 `make eval-site-uninstall`. 같은 포트에 `make eval-site`를
따로 띄우면 바인드 충돌이 나므로 서비스가 있을 때는 수동 실행 대신 `systemctl --user restart arc-eval-site`.

## 직접 플레이 모드 (2026-09-29 제거)

처음에는 헤더의 "직접 플레이"로 environment_files 게임을 오프라인 엔진에서 사람이 직접 둘 수 있었다
(`POST /api/play/...`, 세션은 메모리에만). 뷰어를 기록된 실행의 재생 전용으로 두기 위해 이 모드와 API
(`/api/play/*`, `/api/games`)를 뺐다. 사람 플레이 로그가 필요하면 `make pbg-human-log GAME=..`(터미널)를 쓴다.

## 2026-09-29 추가: 이름 Replay, pbg 라인, 진행 중 실행

- 뷰어 이름을 **Replay**로 바꿨다(페이지 제목·헤더·systemd 설명). make 타깃(`eval-site`)과 포트는 그대로.
- **pbg 실행도 같은 화면에서 본다.** `pbg/env/wrapper.py`의 `ArcadeEnv`가 rulebook과 같은 형식의
  `experiments/pbg/results/logs/<run>/<game>.actions.jsonl`(reset/step, 액션 라벨, 레벨, 시도, 프레임 sha1, 벽시계)을
  쓰고, 뷰어(`replay.py`)는 `EXPERIMENTS = {rulebook, pbg}` 두 results 폴더를 함께 훑는다. pbg 로그는
  `HH:MM:SS 메시지` 형식이라 액션 카운터가 없으므로 기록의 `clock`과 대조해 스텝에 붙인다. 완료 실행의
  레벨 요약은 pbg의 `level_actions` 딕셔너리에서 만든다.
- **진행 중인 실행이 보인다.** 두 러너(`rulebook/run.py`, `pbg/harness/online_runner.py`)가 첫 액션 전에
  `logs/<run>/run.json`(태그·게임·파라미터·pid)을 남기고, 요약 `run-<id>.json`이 아직 없는 로그 폴더는
  pid가 살아 있으면 `running`, 아니면 `aborted`로 목록에 오른다(`● 진행 중` / `✕ 중단`). 진행 중 실행의
  게임·레벨 요약은 actions.jsonl을 집계해 만들고, 페이지는 5초마다 다시 읽어 보드가 늘어나면 제자리에서
  이어 붙인다(마지막 스텝을 보고 있었으면 따라간다). 요약이 생기면 목록 라벨이 완료로 바뀐다.
  이 변경 전에 시작한 실행은 run.json·actions.jsonl이 없어 목록에는 뜨지만 초기 보드만 재생된다.
- **게임 실행마다 고유 hash.** `scripts/eval_viewer/ids.py`의 `game_hash(run_id, game_id)` = sha1("<run id>/<game id>")
  앞 10자리. run id가 유일하고 한 run에서 게임은 한 번 실행되므로 실행마다 유일하고, 두 문자열만 있으면 과거 실행에도
  같은 값이 나온다. 두 러너가 요약의 게임 항목(`"hash"`)과 `logs/<run>/run.json`(`"hashes"`)에 기록한다.
  뷰어는 게임 목록과 레벨 상세에 hash를 보여 주고(클릭 = 복사), 헤더의 "hash로 찾기" 상자나 URL `#h/<hash>`,
  API `GET /api/find/<hash>`(6자 이상 접두어 허용)로 해당 run·게임으로 바로 이동한다.
- 검증: `pbg --games ar25 --minutes 1 --no-llm --budget 40` 스모크를 진행 중에 열어 18스텝 재생, 해시 불일치 0,
  planner 로그 4건이 마지막 스텝에 붙음; 종료 후 같은 run id가 완료 상태로 전환.

## 다음

- 레벨 시점의 룰북 스냅샷(현재는 최종본 + REVIEW 로그로 대체)
- arcnav 실행 로그 지원(형식이 달라 별도 파서 필요)
