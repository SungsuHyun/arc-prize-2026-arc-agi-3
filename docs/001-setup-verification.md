# 001. 환경 구축 및 검증

- **날짜**: 2026-09-22
- **관련 파일**: Makefile, notebooks/kernel-metadata.json, .venv/

## 목적

ARC Prize 2026 — ARC-AGI-3 대회 참가를 위한 로컬 개발 환경을 구축하고,
에이전트가 실제 게임 엔진 위에서 동작하는지 end-to-end로 확인한다.

## 방법

1. 공식 스타터 킷 [arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)를
   프로젝트 디렉터리에 클론 (원격은 `upstream`으로 이름 변경)
2. `notebooks/kernel-metadata.json`의 사용자명을 `monggree`로 설정 (→ 2026-09-22 정정: 실제 계정은 `sungsuhyun`)
3. `make setup PYTHON=/usr/bin/python3.12` — venv 생성, `arc-agi 0.9.9`,
   `kaggle 2.2.4` 설치, `vendor/ARC-AGI-3-Agents` 프레임워크 클론
4. `make verify-local` — ls20, vc33 두 게임에 50스텝 스모크 테스트

## 결과

```
========= SUMMARY =========
  ls20     levels=  0  actions=   51  state=GameState.NOT_FINISHED
  vc33     levels=  0  actions=   51  state=GameState.NOT_FINISHED

Aggregate scorecard score: 0.0
```

- 게임 소스가 `environment_files/`에 다운로드·캐시됨 (이후 오프라인 동작)
- 랜덤 스타터라 점수 0.0은 정상 (파이프라인 검증이 목적)
- 로컬 엔진 속도: 초당 수백 액션 (fps 300~500)

## 결론 / 다음 단계

- 파이프라인 정상. 수정 대상은 `agent/my_agent.py` 하나
  (`MyAgent.choose_action` / `is_done`)
- 발견: 구형 `~/.kaggle/kaggle.json` 토큰은 401 → 신형 `KGAT_` 토큰을
  프로젝트 로컬 `.kaggle/access_token`에 저장하는 방식으로 전환 (003 참고)
