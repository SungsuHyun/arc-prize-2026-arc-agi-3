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
