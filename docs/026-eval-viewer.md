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
스스로 새로고침하고 URL 해시로 같은 스텝으로 돌아온다. 디스크 재생 캐시는 유지되고 플레이 세션은 사라진다.
단일 프로세스로 띄우려면 `--no-reload`.

## 부팅 시 자동 시작 (systemd user service)

`make eval-site-install`은 `scripts/systemd/arc-eval-site.service`를 `~/.config/systemd/user/`에 설치하고
`systemctl --user enable --now`로 바로 띄운다(포트는 `EVAL_PORT=`, 기본 8090). 사용자에게
`loginctl enable-linger`가 켜져 있으면 로그인 없이 서버 부팅 직후 시작되고, 죽으면 5초 뒤 재시작된다.
서비스도 위의 감시 프로세스를 그대로 쓰므로 소스를 고치면 자동 반영된다. 로그는
`journalctl --user -u arc-eval-site -f`, 제거는 `make eval-site-uninstall`. 같은 포트에 `make eval-site`를
따로 띄우면 바인드 충돌이 나므로 서비스가 있을 때는 수동 실행 대신 `systemctl --user restart arc-eval-site`.

## 직접 플레이 모드

헤더의 "직접 플레이"를 누르면 왼쪽에 environment_files의 25개 게임이 뜨고, 고르면 오프라인 엔진으로
새 판이 시작된다. 조작은 화살표·Space·7(ACTION7)·R(리셋) 키 또는 화면 버튼, MOUSE가 허용된 게임은
보드 클릭(클릭 좌표 → row/col 변환). 액션 배너와 오른쪽 "내 액션" 목록에 매 액션의 결과(바뀐 셀 수,
LEVEL COMPLETED, GAME OVER)가 표시된다. 시간은 재지 않고 아무것도 기록하지 않는다.
서버 API: `POST /api/play/new {game_id}`, `POST /api/play/<sid>/step {action}`, `POST /api/play/<sid>/reset`
(세션은 메모리에만, 최대 32개).

## 다음

- 레벨 시점의 룰북 스냅샷(현재는 최종본 + REVIEW 로그로 대체)
- arcnav 실행 로그 지원(형식이 달라 별도 파서 필요)
