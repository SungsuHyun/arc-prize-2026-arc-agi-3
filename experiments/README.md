# 실험 버전 관리 (experiments/)

에이전트 실험을 버전별 폴더로 기록합니다. 각 폴더는 **자체 완결적**입니다 —
에이전트 코드 스냅샷 + 설정 + 실행 결과가 함께 있어, `agent/my_agent.py`가
바뀌어도 과거 실험을 언제든 재현할 수 있습니다.

```
experiments/
├── _template/                 # 새 실험의 config/notes 원형
├── summary.json               # 전체 실험 집계 (자동 생성, dashboard 입력)
└── vNNN-제목/
    ├── agent.py               # 이 버전의 에이전트 (스냅샷, 독립 실행)
    ├── config.json            # games, max_steps, params, description
    ├── notes.md               # 가설 / 변경점 / 관찰 / 결론 (사람이 작성)
    └── results/
        └── run-<시각>.json    # 실행 1회당 1개 (자동 생성, 누적)
```

## 워크플로

```bash
make exp-new NAME=greedy-search        # 새 버전 스캐폴드 (자동 번호 부여)
make exp-new NAME=tweak FROM=v002      # 기존 실험 기반으로 파생
# → experiments/vNNN-greedy-search/agent.py 수정
make exp-run NAME=v002                 # 실행 + results/에 JSON 기록 (prefix 매칭)
make exp-run NAME=v002 GAME=ls20 STEPS=100   # 일부 게임만 빠르게
make exp-summary                       # 전체 실험 비교 테이블 + summary.json 갱신
make bench GAME=ls20,vc33 STEPS=400    # 모든 버전 일괄 실행 + GitHub Pages 자동 배포
make site-publish                      # Pages 수동 배포 (gh-pages 브랜치)
make site                              # 로컬 미리보기 (localhost:8080)
make dashboard                         # experiments/site/ 정적 파일만 재생성
```

## 결과 JSON 스키마 (dashboard 입력)

`results/run-*.json` 하나가 실행 1회입니다. 주요 필드:

| 필드 | 내용 |
|---|---|
| `experiment`, `run_id`, `started_at` | 식별자 |
| `git.commit`, `git.dirty` | 재현용 코드 상태 |
| `config` | 실행에 실제 사용된 games/max_steps/params |
| `aggregate.score` | 스코어카드 종합 점수 (리더보드 지표와 동일 계산) |
| `games[]` | 게임별 levels_completed / win_levels / actions / score |
| `scorecard` | 엔진 스코어카드 전체 덤프 (레벨별 상세) |

`make exp-summary`가 만드는 `experiments/summary.json`은 모든 실험×실행을
한 파일로 모은 것이고, `experiments/site/`가 이를 시각화합니다 — `index.html`(전체 개요 + 벤치마크
카드 목록)과 벤치마크 1회당 1개의 `bench-<시각>.html` 상세 페이지.
`make bench`가 모두 자동 갱신하고 GitHub Pages로 배포함:
**https://sungsuhyun.github.io/arc-prize-2026-arc-agi-3/**

## 규칙

- 실험 폴더는 생성 후 **수정하지 않는 것**이 원칙 (notes.md 제외).
  코드를 바꾸려면 `make exp-new FROM=vNNN`으로 새 버전을 파생
- 좋은 버전을 제출하려면 해당 `agent.py`를 `agent/my_agent.py`로 복사 후
  `make play-local` → `make submit`
- 큰 발견은 `docs/NNN-*.md`에도 기록 (docs는 서사, experiments는 데이터)
