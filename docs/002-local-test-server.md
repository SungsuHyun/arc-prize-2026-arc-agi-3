# 002. 로컬 테스트 서버 구축

- **날짜**: 2026-09-22
- **관련 파일**: scripts/serve_local.py, Makefile(`serve` 타깃), vendor/ARC-AGI-3-Agents/.env

## 목적

Kaggle에 제출하지 않고 로컬에서 공식 API(`three.arcprize.org`)와 동일한
게임 서버를 띄워, HTTP 레벨에서 에이전트를 테스트할 수 있게 한다.

## 방법

1. `arc-agi` 패키지에 내장된 Flask 앱(`arc_agi.server.create_app`)을 발견 —
   공식 API와 동일한 라우트 제공
2. `scripts/serve_local.py` 작성 + `make serve` 타깃 추가 (기본 포트 8001 —
   공식 에이전트 프레임워크의 기본값과 일치)
3. curl로 전체 게임 루프 검증: 게임 목록 → 스코어카드 open → RESET →
   ACTION1 → 스코어카드 조회
4. 공식 프레임워크(`vendor/ARC-AGI-3-Agents`)의 random 에이전트를 로컬
   엔진으로 실행

## 결과

- `/api/games` → 25개 게임 반환 (ls20, vc33, lf52, sk48, cn04, tr87, …)
- RESET → 64×64 그리드 프레임, `guid`, `available_actions: [1,2,3,4]`,
  `win_levels: 7` (ls20 기준) 정상 반환
- ACTION1 → 상태 갱신 정상. 스코어카드에 레벨별 점수·기준 액션 수 기록됨
- 프레임워크 random 에이전트: 80액션 플레이, `recordings/`에 녹화 저장

### 발견한 주의점

1. **게임 id 규칙**: RESET은 짧은 id(`ls20`) 허용, 이후 ACTION1~7은
   전체 버전 id(`ls20-9607627b`) + RESET이 준 `guid` 필요
2. **응답에 score 필드 없음**: `levels_completed` / `win_levels`로 진행도
   추적, 점수는 `/api/scorecard/<card_id>`에서 레벨별 기준 액션 수 대비 계산
3. **로컬 서버에는 `/api/games/<id>/source` 엔드포인트 없음**: 프레임워크의
   `ARC_BASE_URL`을 localhost로 바꾸면 게임 소스 다운로드가 404남
4. **프레임워크 `.env.example`은 `OPERATION_MODE=online`(원격) 기본값**:
   로컬 실행용 `vendor/ARC-AGI-3-Agents/.env`를 별도 생성
   (`OPERATION_MODE=normal`, `ENVIRONMENTS_DIR`은 프로젝트 루트 절대경로).
   `make clean`이나 재클론 시 이 `.env`는 사라지므로 재생성 필요

## 결론 / 다음 단계

- HTTP 레벨 테스트, 타 언어/프로세스에서 엔진 구동, Kaggle gateway 재현이
  필요하면 `make serve` 사용
- 에이전트 코드 반복 개발은 여전히 `make play-local`(in-process)이 가장 빠름
- Kaggle 재실행 환경도 동일 구조(gateway 사이드카가 `http://gateway:8001`
  서빙)라서 로컬 검증이 그대로 통용됨
