# 004. 실험 버전 관리 구조 구축

- **날짜**: 2026-09-22
- **관련 파일**: experiments/, scripts/run_experiment.py, scripts/exp_summary.py, scripts/new_experiment.py, Makefile

## 목적

에이전트 실험을 여러 버전으로 진행하면서 각 버전의 코드·설정·결과를
재현 가능하게 기록하고, 추후 dashboard로 비교할 수 있는 데이터 구조를 만든다.

## 방법

- `experiments/vNNN-제목/` 폴더 구조: agent.py 스냅샷 + config.json +
  notes.md + results/run-*.json (실행 1회당 1개 누적)
- `make exp-new NAME=...` — 자동 번호로 스캐폴드 (`FROM=vNNN`으로 파생 가능)
- `make exp-run NAME=vNNN` — 실행 후 구조화된 JSON 기록
  (git 커밋, 게임별 레벨/액션/점수, 스코어카드 전체 덤프 포함)
- `make exp-summary` — 전체 실험 비교 테이블 출력 + `experiments/summary.json`
  생성 (dashboard가 읽을 단일 입력 파일)
- v001-random-baseline 생성 후 ls20, vc33 50스텝으로 전체 흐름 검증

## 결과

```
experiment                   runs    best    last  last run
----------------------------------------------------------------------
v001-random-baseline            1   0.000   0.000  20260922-055228
```

- 결과 JSON에 git commit(`2a1e65c`, dirty 여부), 게임별
  `levels_completed/win_levels/actions/score`, aggregate, 스코어카드
  덤프까지 저장되는 것 확인

## 결론 / 다음 단계

- 실험 폴더는 불변 원칙 (notes.md 제외) — 수정 대신 `FROM=`으로 새 버전 파생
- 점수 계산은 리더보드와 동일한 스코어카드 로직 사용
- dashboard는 `experiments/summary.json` 하나만 읽으면 됨 — 추후 정적
  HTML(Artifact)이나 로컬 웹앱으로 시각화 예정
- 다음: v002부터 실제 전략 실험 시작
