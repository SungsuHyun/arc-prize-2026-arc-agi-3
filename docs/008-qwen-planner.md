# 008. v004 Qwen 플래너 연결 — L2 가동

- **날짜**: 2026-09-22
- **관련 파일**: experiments/v004-qwen-planner/, scripts/run_experiment.py(OFFLINE 폴백)

## 목적

docs/005 로드맵 v004: 로컬 Qwen을 L2 플래너로 연결해 3계층(반사/탐색/계획)
전체를 가동하고, 모델 크기별 latency·계획 품질을 실측한다.

## 방법

- 로컬 ollama 발견 (GPU 구동): qwen3.5:9b(6.6GB), qwen3.5:35b(24GB, MoE)
  — docs/005에서 가정한 "~27B급 Qwen" 자리에 35b가 실물로 존재
- QwenPlanner: ollama /api/chat, **think=false + format=json**이 핵심
  (thinking 모드가 토큰 예산을 소진해 빈 응답 → 비활성화 필수)
- 트리거: stuck(15액션 무진전) | 40액션 경과 | 레벨업, 게임당 12회 예산,
  계획 실행 중 무변화 3연속이면 재계획
- 인코더 개선: resized 이벤트, 반복 ×N 압축
- 러너에 OFFLINE 폴백 추가 (three.arcprize.org DNS 불안정 시 캐시 사용)

## 결과

| 설정 | 결과 |
|---|---|
| latency | 호출당 ~2초 (9b/35b 웜 상태 유사, format=json) |
| 안정성 | 게임당 9~12회 호출, 오류 ≈0, 폴백 정상 |
| 9b 플래너 | 4게임 종합 0.0004 — v003(0.010)보다 **나쁨** (액션 낭비) |
| **35b 플래너** | **vc33 레벨 2, 점수 0.321 — 프로젝트 최고** |

교훈: 관측 인코딩이 같아도 플래너 품질이 점수를 좌우. 9b는 커버리지만
늘리고 계획이 부정확해 (기준/사용액션)² 공식에서 오히려 손해.

## 결론 / 다음 단계

- 35b(=Kaggle에서는 RTX6000 + 30B급 4bit)를 기본 경로로 확정
- v005: Kaggle 패키징 — QwenPlanner._complete()를 in-process vLLM으로 교체,
  가중치/wheel 데이터셋 첨부, build_notebook.py 확장
- v006: ls20류(이동+게이지) 공략 — 레벨별 기억, 프롬프트에 게이지 힌트
