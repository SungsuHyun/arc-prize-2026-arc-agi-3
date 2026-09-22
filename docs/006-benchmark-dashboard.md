# 006. 일괄 벤치마크 + 웹 대시보드

- **날짜**: 2026-09-22
- **관련 파일**: scripts/benchmark.py, scripts/build_dashboard.py, Makefile(bench/dashboard), experiments/dashboard.html

## 목적

모든 실험 버전을 동일 조건(같은 게임 셋, 같은 스텝 수)에서 한 번에 실행해
공정하게 비교하고, 결과를 웹페이지로 볼 수 있게 한다.

## 방법

- `make bench [GAME=..] [STEPS=..] [ONLY=v001,v002]` — 전 버전을 서브프로세스로
  순차 실행, 공통 태그(`bench-<시각>`)를 부여해 그룹으로 비교 가능하게 기록.
  종료 후 summary.json과 dashboard.html 자동 재생성
- `make dashboard` — 임의 시점에 사이트만 재생성
- `experiments/site/` — **벤치마크 결과마다 별도 웹페이지** (자체 완결형,
  데이터 내장, file://로 바로 열림):
  - `index.html` — 요약 타일, 벤치마크 리포트 카드 목록(클릭 → 상세),
    점수 추이 라인 차트, 전체 실행 기록 표
  - `bench-<시각>.html` — 벤치마크 1회당 1페이지: 버전별 점수/레벨 바 차트,
    게임별 상세 표, 해당 벤치마크의 실행 목록
  - 라이트/다크 모드 (OS 설정 따름 + 수동 토글), 호버 툴팁

## 결과

- 첫 벤치마크 `bench-20260922-061016`: v001/v002 × 4게임(ls20,lf52,sk48,vc33)
  × 400스텝 — 둘 다 0점 (레벨 완료 없음, 예상 범위)
- 헤드리스 Chromium으로 index/벤치마크 페이지 라이트·다크 렌더링 확인
- 버그 수정: `experiments/site/` 폴더가 실험으로 잘못 집계되던 것을
  `agent.py` 존재 여부로 판별하도록 수정 (exp_summary.py)
- 팔레트는 dataviz 검증된 기본 팔레트(카테고리 8색 고정 순서) 사용

## 결론 / 다음 단계

- 앞으로 버전 추가 후 `make bench` 한 번이면 새 벤치마크 페이지가 생기고
  index 카드 목록에 자동 등록됨
- 점수가 계속 0인 동안은 '게임별 상세'의 액션 수와 (추후 추가할) 탐색
  커버리지 지표가 더 유용 — v003에서 unique states visited 같은 진행
  지표를 결과 JSON에 추가하는 것 고려
