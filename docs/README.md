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
