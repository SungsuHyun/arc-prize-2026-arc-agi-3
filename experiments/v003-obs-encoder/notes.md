# v003-obs-encoder

## 가설

64×64 원시 그리드(4천+ 토큰)를 객체 목록 + 변화 로그로 압축하면 수백 토큰
으로 게임 상태와 규칙 단서를 LLM에 전달할 수 있다. (v004 Qwen 연결의 전제)

## 변경점 (v002 대비)

- ObservationEncoder 추가: [PROGRESS]/[AVAILABLE]/[OBJECTS](connected
  component 색·크기·bbox·중심)/[RECENT](액션별 이동·생성·소멸·셀변화)/
  [ACTION_STATS] 텍스트 인코딩
- ACTION6 클릭 타깃을 인코더의 객체 중심 목록으로 통일
- metrics(unique_states, avg/max_obs_chars)를 결과 JSON에 노출
  (run_experiment.py가 agent.metrics 병합)
- 관측 샘플 확인: scripts/dump_observations.py v003 --game ls20

## 관찰

- **첫 레벨 완료**: vc33 레벨 1 클리어, 점수 0.040 (프로젝트 첫 비영점).
  클릭 타깃이 배경 제외 24개 최대 객체로 바뀐 효과로 추정
- 관측 크기: 평균 ~1.9K chars(~620토큰), 최대 ~2.1K — 27B 플래너 예산 내
- ls20 변화 로그가 메커니즘을 그대로 노출: ACTION1/2=아바타 y±5,
  ACTION3/4=x±5, 하단 c3/c11 바가 이동마다 ±2 증감(액션 예산 게이지로 추정)
  → LLM이 규칙을 추론할 수 있는 형태
- vc33은 available_actions=[6] (클릭 전용 게임)

## 개선 여지 (v004로)

- 같은 색·같은 위치의 크기 변화가 +/− 쌍으로 나옴 → "resized c3 60→62"로
  통합하면 로그가 더 짧고 명확해짐
- 동일 변화 반복 시 "×N회 반복" 압축

## 결론

채택 / 폐기 / 후속 버전 아이디어.
