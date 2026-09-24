# 020. arcnav — Duck 코드 없이 우리 솔루션으로 리빌드

- **날짜**: 2026-09-24
- **관련 파일/커밋**: `arcnav/` (frame, nav, sandbox, llm, prompts, solver, agent, runner),
  `scripts/run_arcnav.py`, `scripts/build_arcnav_notebook.py`, `notebooks/arcnav/`, `make arcnav`

## 목적

017~019의 하이브리드(vendored Duck 하네스 + 우리 nav 헬퍼 + 솔버 합성)는 성능은 좋았지만
"Duck 복제 + 패치"라는 정체성 문제가 있었다. 사용자 요청("Duck이란 이름을 없애고 전체적으로
리빌딩해서 우리만의 솔루션으로 탈바꿈")에 따라, 검증된 아이디어만 남기고 코드를 처음부터
다시 작성했다. Duck 코드는 한 줄도 쓰지 않는다(아이디어 계보만 README/NOTICE에 남김).

## 설계

| 모듈 | 역할 | 비고 |
|---|---|---|
| `frame.py` | 그리드→ascii(hex 한 글자/셀), 연결요소 세그멘테이션(id/색/픽셀/bbox/center/hash/children/hud), 인접 그래프, 턴 전후 diff 요약 | 순수 python, 의존성 없음 |
| `nav.py` | v006~v012의 NavHelper 이식. 이번에 **반대 방향 추론** 추가(UP 학습 시 DOWN 델타 가정) | 우리 코드 |
| `sandbox.py` | 게임당 1개의 `python -I` 자식 프로세스, JSON-lines 프로토콜. `action()`, `propose_solver()`, `nav`, `transitions`, `level_transitions`. 프로토콜 채널을 fd 1에서 분리해 `os.system` 등이 오염 못 함. 타임아웃(SIGALRM)·메모리 제한(4GB) | 모델 코드의 `propose_solver`/`action` 재정의 거부 |
| `llm.py` | OpenAI 호환 chat 클라이언트(urllib만 사용, Kaggle 재실행 컨테이너에서 추가 패키지 불필요). tool_calls, reasoning 필드, 컨텍스트 초과 예외 | |
| `prompts.py` | 우리 문구의 시스템 프롬프트 + nav/click 솔버 템플릿 | nav 템플릿: 미학습 키 먼저 → 미방문 타깃 → 가장 오래된 타깃 → frontier |
| `solver.py` | 저장된 solve()의 자동 실행 정책: 턴당 12액션, 무변화 2턴, 무진전 80액션, 동일 보드 6회(순환) 시 실패 보고. 주기적 제안 요구는 비활성(v012e/f 결론) | |
| `agent.py` | GameSession: env 스텝, 레벨 완료/게임오버(자동 리셋) 처리, 모델 턴(유저 메시지 = 레벨/예산/직전 결과 diff/nav 요약/솔버 상태/현재 보드), 컨텍스트 트리밍(최근 3턴만 보드 유지, 오래된 툴 출력 축약, 80% 초과 시 오래된 턴 삭제), 솔버 턴, 전역 데드라인 | 로그: `<game>.log`, `<game>.jsonl` |
| `runner.py` | 스레드 풀로 게임 병렬 실행, experiments 형식 결과 JSON(`experiments/arcnav/results/`), 스코어카드 병합 | 대회 모드(gateway) 지원 |

Kaggle: `scripts/build_arcnav_notebook.py`가 `arcnav/` 소스를 노트북에 내장하고, 공개
wheelhouse(vllm 0.19)와 Qwen3.6-27B-FP8 HF 스냅샷을 붙여 노트북 안에서 vLLM 서버를 띄운 뒤
재실행이면 gateway의 전 게임을(동시 12, 전역 470분 캡), 커밋이면 2게임 15분 스모크를 돌린다.

## 방법

```bash
make arcnav GAME=ls20 MINUTES=15 JOBS=1 TAG=smoke-ls20          # 첫 라이브 실행
make arcnav GAME=ls20,cn04,s5i5,vc33 MINUTES=25 JOBS=2 TAG=4games-a   # v012f 4게임 세트와 비교
```

오프라인(모델 없음) 검증: 샌드박스 프로토콜 테스트(액션/제안/타임아웃/메모리/오염 방지),
nav 템플릿만으로 ls20 레벨 1 완료(23~25액션, 기준 22).

## 결과

(아래 표는 실행 종료 후 채움)

## 결론 / 다음 단계

