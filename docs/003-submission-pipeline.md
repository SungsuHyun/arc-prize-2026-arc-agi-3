# 003. 제출 파이프라인 점검

- **날짜**: 2026-09-22
- **관련 파일**: .kaggle/access_token, scripts/build_notebook.py, notebooks/submission.ipynb

## 목적

별도 설정 없이 `make submit`으로 바로 제출 가능한 상태인지 확인한다.

## 방법

1. 신형 `KGAT_` 토큰을 `.kaggle/access_token`에 저장 후 인증 테스트
   (`kaggle competitions list`, `kaggle competitions files`)
2. `make notebook`으로 제출 노트북 생성 검증

## 결과

- 토큰 인증 정상, 대회 데이터 API 접근 가능 → 규칙 동의 완료 상태 확인
- `notebooks/submission.ipynb` 정상 생성: 5셀, 가속기 T4
- 제출물에 포함되는 사용자 코드는 `agent/my_agent.py` 본문뿐 —
  테스트 서버(002) 관련 파일은 제출물과 완전 분리

## 결론 / 다음 단계

- 제출 절차: `make submit` → `make status`가 complete → kaggle.com 커널
  페이지에서 **"Submit to Competition"** 클릭 + `submission.parquet` 선택
  (이 클릭이 하루 5회 제한 대상)
- 랜덤 스타터 상태로 1회 제출해 전체 파이프라인(Phase A/B) 확인 권장
- 비ML 에이전트 동안은 `scripts/build_notebook.py`의 `ACCELERATOR`를
  `"cpu"`로 낮춰 GPU 쿼터 절약 고려
