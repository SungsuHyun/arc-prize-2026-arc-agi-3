# 테스트 기록 (docs/)

이 폴더는 프로젝트에서 수행한 테스트·실험을 **순차 번호**로 기록합니다.

## 규칙

- 파일명: `NNN-짧은-제목.md` (예: `004-greedy-agent-baseline.md`)
- 새 테스트마다 다음 번호를 부여하고 아래 목록에 한 줄 추가
- 형식은 [TEMPLATE.md](TEMPLATE.md) 참고: 날짜 / 목적 / 방법 / 결과 / 결론(다음 단계)
- 에이전트 점수 실험은 가능하면 `make play-local` 출력(레벨/액션 수)을 그대로 붙여넣기

## 목록

| # | 문서 | 날짜 | 요약 |
|---|---|---|---|
| 001 | [환경 구축 및 검증](001-setup-verification.md) | 2026-09-22 | 스타터 킷 설치, verify-local 통과 |
| 002 | [로컬 테스트 서버](002-local-test-server.md) | 2026-09-22 | make serve 구축, HTTP 게임 루프 검증 |
| 003 | [제출 파이프라인 점검](003-submission-pipeline.md) | 2026-09-22 | 토큰/규칙/노트북 빌드 확인, 제출 준비 완료 |
| 004 | [실험 버전 관리 구조](004-experiment-versioning.md) | 2026-09-22 | experiments/ 구조 + exp-new/run/summary, v001 검증 |
| 005 | [Qwen LLM 에이전트 설계](005-qwen-llm-agent-design.md) | 2026-09-22 | 3계층 하이브리드 설계, v002 스켈레톤 구현 |
| 006 | [일괄 벤치마크 + 대시보드](006-benchmark-dashboard.md) | 2026-09-22 | make bench, 벤치마크별 웹페이지(site/) + index |
| 007 | [v003 관측 인코더](007-observation-encoder.md) | 2026-09-22 | 객체+변화로그 텍스트 인코딩, 첫 비영점 점수 |
| 008 | [v004 Qwen 플래너](008-qwen-planner.md) | 2026-09-22 | L2 가동, 35b로 vc33 레벨2 · 점수 0.321 최고 기록 |
| 009 | [v007 자기 성찰 마크다운](009-self-reflect-md.md) | 2026-09-22 | 9b + 보상 평가자 루프, insights.md 누적·주입 실험 |
| 010 | [v005 Kaggle vLLM 패키징](010-kaggle-vllm-packaging.md) | 2026-09-22 | wheels 커널+모델 첨부, 백엔드 교체형 플래너, RTX6000 스모크 |
| 011 | [v006 내비게이션+기억](011-nav-memory-ls20.md) | 2026-09-22 | 아바타/맵/BFS goto, ls20 레벨1 최초 완료 (17액션) |
| 012 | [v008 검증된 액션 사실 + 코드 보상](012-verified-insights.md) | 2026-09-22 | v007 보완: 코드 측정 액션 모델·코드 보상, 9b 계열 최고 0.0058, 35b 평가자 이득 없음 |
| 013 | [v009/v010 클릭 스위프+LLM 게이트](013-click-sweep-llm-gate.md) | 2026-09-23 | 25게임 0.408 (LLM 없음), LLM은 stuck 시에만 |
| 014 | [방법론 재검토](014-methodology-review.md) | 2026-09-23 | 레벨 가중 채점·400액션 상한·노이즈 평가·규칙 미학습 진단, 규칙 합성 방향 |
| 015 | [방법론 탐구](015-methodology-survey.md) | 2026-09-23 | 월드모델/튜닝/하네스 비교, 공개 결과 정리, REPL 툴 하네스(Duck) 주력 판단 |
| 016 | [로드맵 v2](016-roadmap-v2.md) | 2026-09-23 | finding별 방향 D1~D11, 실행 순서 v011~v014 |
| 017 | [Duck 하네스 재현 + nav 하이브리드](017-duck-harness-nav-hybrid.md) | 2026-09-23 | 로컬 5090에서 Duck 재현, nav 헬퍼 주입, 27B vs 35B-A3B 비교 |
| 018 | [Duck 자문 앙상블](018-duck-advisor-ensemble.md) | 2026-09-23 | 작은 Qwen 자문 ×N → 27B 결정 모델, 017 하이브리드와 같은 조건 비교 |
| 019 | [v012 솔버 합성](019-solver-synthesis-v012.md) | 2026-09-23 | propose_solver + 자동 실행, 샌드박스 nav 수정, 오프라인 ls20 19액션 |
| 021 | [문제의 근본 정의와 분석 계획](021-problem-definition.md) | 2026-09-25 | T/G/B/L 미지 요소, 하위 문제 P1~P6 진단, 시뮬레이터+탐색으로 재정의, 진단 실험 D1~D6 |
| 020 | [arcnav 리빌드](020-arcnav-rebuild.md) | 2026-09-24 | Duck 코드 없는 자체 하네스, 반복 0~7, 리더보드 진단, gpt-oss 실험 |
| 022 | [아키텍처 v2 — 프로그램이 운전하고 모델은 자문](022-architecture-v2.md) | 2026-09-26 | 0층 탐색 엔진·1층 규칙 귀납/계획·2층 자문의 역할과 상태, 채점 공식 함의, 마일스톤 M1–M4 |
