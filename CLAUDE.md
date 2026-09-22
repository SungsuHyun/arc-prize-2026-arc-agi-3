# ARC Prize 2026 — ARC-AGI-3 프로젝트

Kaggle [arc-prize-2026-arc-agi-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
참가 프로젝트. 공식 스타터 킷(upstream: arcprize/ARC-AGI-3-Kaggle-Starter) 기반.
인터랙티브 에이전트 벤치마크 — 에이전트가 처음 보는 게임(64×64 그리드, 값 0–15)을
탐험하며 레벨을 완료해야 함.

## 자동 커밋 & 푸시 (중요)

**중요한 분기점마다 묻지 말고 커밋하고 origin/main에 푸시한다.** 분기점 기준:

- 새 실험 버전 생성 또는 의미 있는 실험 결과 기록 (`experiments/` 변경)
- `docs/NNN-*.md` 새 문서 추가
- 스크립트/Makefile/인프라 추가·수정, 버그 수정
- `agent/my_agent.py` 전략 변경
- Kaggle 제출 직전 (제출한 코드가 반드시 커밋에 남도록)

규칙:
- 커밋 메시지: 영어, 한 줄 요약 + 필요시 본문. 실험 결과 커밋은
  `exp(vNNN): ...`, 인프라는 `infra: ...`, 문서는 `docs: ...` 프리픽스
- 사소한 중간 수정은 모아서 분기점에 한 번에 커밋 (커밋 스팸 금지)
- push 실패(네트워크 등) 시 커밋만 남기고 계속 진행, 마지막에 사용자에게 알림
- `.kaggle/`(토큰), `vendor/`, `environment_files/`, `recordings/`는
  gitignore됨 — 절대 강제 추가하지 말 것

## 핵심 명령

```bash
make play-local [GAME=ls20] [STEPS=200]   # agent/my_agent.py 로컬 실행
make exp-new NAME=title [FROM=vNNN]       # 새 실험 버전 스캐폴드
make exp-run NAME=vNNN [GAME=..] [STEPS=..] # 실험 실행 + results/ JSON 기록
make exp-summary                          # 실험 비교 + experiments/summary.json
make bench [GAME=..] [STEPS=..] [ONLY=..]  # 전 버전 동일조건 일괄 실행 + Pages 자동 배포
make site-publish                         # 벤치마크 사이트를 GitHub Pages로 수동 배포
make site                                 # 로컬 미리보기 (localhost:8080, 자동 재생성)
make dashboard                            # experiments/site/ 정적 파일만 재생성
make serve                                # 로컬 REST 서버 (:8001, 공식 API 동일)
make submit && make status                # Kaggle 푸시 (리더보드 제출은 웹에서 수동)
```

벤치마크 결과 웹사이트: **https://sungsuhyun.github.io/arc-prize-2026-arc-agi-3/**
(gh-pages 브랜치, `make bench` 후 자동 갱신 — 공개 페이지이므로 비공개 정보 금지)

## 컨벤션

- **experiments/vNNN-제목/**: 실험 1버전 = 1폴더 (agent.py 스냅샷 + config.json +
  notes.md + results/). 폴더는 불변(notes.md 제외), 수정 대신 `FROM=`으로 파생.
  세부 규칙은 experiments/README.md
- **docs/NNN-제목.md**: 테스트·발견을 순차 번호로 기록 (한국어,
  docs/TEMPLATE.md 형식), docs/README.md 표에 한 줄 추가
- 실험은 데이터(experiments), 서사는 문서(docs) — 큰 발견은 양쪽 모두

## 도메인 지식 (자주 잊는 것들)

- **점수는 레벨 완료에서만 발생**: 레벨당 `(기준액션/사용액션)²×100` (상한 115),
  미완료 레벨 0점. 랜덤 에이전트가 0점인 이유
- 액션: ACTION1–5, 7 = 단순 / ACTION6 = 복합(x,y 클릭, 0–63)
- 프레임: `latest_frame.frame`(list of 64×64 grids), `available_actions`,
  `levels_completed`, `win_levels`, `state`(NOT_PLAYED/NOT_FINISHED/WIN/GAME_OVER)
- RESET 후 액션은 전체 버전 게임 id + guid 필요 (HTTP 사용 시)
- Kaggle 재실행은 인터넷 차단 — 외부 API 불가, 모델 가중치는 Kaggle
  Dataset/Model로 첨부해야 함
- `vendor/ARC-AGI-3-Agents/.env`는 make clean/재클론 시 사라짐 — 재생성 필요
  (OPERATION_MODE=normal, ENVIRONMENTS_DIR=<프로젝트 루트 절대경로>/environment_files)
- 제출물에는 `agent/my_agent.py` 본문만 포함됨 (테스트 인프라와 완전 분리)
