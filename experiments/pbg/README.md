# experiments/pbg — pbg 시스템 실행 결과

`make pbg` (pbg/harness/online_runner.py)가 쓰는 결과 폴더. 스키마는 rulebook/arcnav 결과와 같다
(`results/run-<시각>.json`, `aggregate.score`, `games[]`).

- `results/run-*.json` — 실행 1회 = 1파일. `games[].hypotheses/goals/semantics/level_actions/llm_calls` 포함
- `results/logs/<run>/<game>.log`, `<game>.events.jsonl` — 상태 전이 이벤트(원인 귀속 입력), gitignore
- `memory/` — 게임별 에피소드/의미 메모리 + LLM 응답 캐시(gitignore, `--fresh`로 게임별 초기화)

지표·병목 귀속: `make pbg-metrics RUN=experiments/pbg/results/run-....json` (홀드아웃 5게임은 pbg/data/holdout.txt, 분리 보고).
