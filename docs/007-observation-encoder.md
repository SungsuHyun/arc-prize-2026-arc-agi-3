# 007. v003 관측 인코더 구현

- **날짜**: 2026-09-22
- **관련 파일**: experiments/v003-obs-encoder/, scripts/dump_observations.py, scripts/run_experiment.py(metrics 병합)

## 목적

docs/005 로드맵의 v003 단계: 64×64 원시 그리드(4천+ 토큰)를 LLM 플래너가
읽을 수 있는 수백 토큰의 구조화 텍스트로 압축하는 관측 인코더를 만든다.

## 방법

- v002에서 파생한 v003에 `ObservationEncoder` 구현:
  `[PROGRESS]` 레벨/상태 · `[AVAILABLE]` 가용 액션 · `[OBJECTS]` connected
  component 객체(색·크기·bbox·중심, 큰 순 24개) · `[RECENT]` 액션별 변화
  로그(이동/생성/소멸/셀 수, 최근 8개) · `[ACTION_STATS]` 액션 효과 통계
- 에이전트 `metrics`(unique_states, avg/max_obs_chars)를 결과 JSON에 병합
  하도록 run_experiment.py 확장, 대시보드 게임별 셀에 상태 수 표시
- `scripts/dump_observations.py vNNN --game ls20` — 실제 플레이 중 수집한
  인코딩 샘플을 출력해 오프라인 검수

## 결과

- **프로젝트 첫 비영점 점수**: v003이 vc33 레벨 1 완료, 점수 0.040
  (클릭 타깃을 인코더의 객체 중심 목록으로 바꾼 효과로 추정)
- 관측 크기: 평균 ~1.9K chars(~620토큰), 최대 ~2.1K — 27B 플래너 호출
  예산 내
- **변화 로그가 게임 규칙을 노출**함을 확인 (ls20):
  ACTION1/2 = 아바타(c9s15+c12s10) y축 ±5 이동, ACTION3/4 = x축 ±5,
  하단 c3/c11 바가 이동마다 ±2 증감(액션 예산 게이지로 추정)
  → LLM이 이 텍스트만으로 조작 체계를 추론할 수 있는 수준
- 벤치마크(4게임×400스텝): v001 0.116(sk48 레벨 1 — 시간 시드라 운),
  v002 0.0, v003 0.010(vc33 레벨 1)
- 참고: v001은 비결정적(시간 시드), v002/v003은 seed=1337 고정 —
  버전 비교 시 v001 수치는 노이즈로 볼 것
- 벤치마크 중 v001이 1회 일시 실패(three.arcprize.org 401 → 캐시 폴백
  타이밍 문제로 추정), 재실행으로 해소

## 결론 / 다음 단계 (v004)

- 인코더 개선: 같은 색·위치의 크기 변화를 "resized c3 60→62"로 통합,
  동일 변화 반복은 "×N" 압축
- Qwen 연결: QWEN_ENDPOINT(OpenAI 호환) → 관측 텍스트 → JSON 계획,
  T4+8B 경로부터 실측 (docs/005 리스크 섹션)
