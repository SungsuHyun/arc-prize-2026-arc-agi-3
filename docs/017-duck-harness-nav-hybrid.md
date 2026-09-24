# 017. Duck 하네스 로컬 재현 + nav 헬퍼 하이브리드 (로드맵 D5/D8)

- **날짜**: 2026-09-23
- **관련 파일**: harness/duck/ (MIT, Tufalabs/duck-harness 사본),
  harness/duck/ARC3-Inference/inference/agent/nav_helpers.py, python_tool_sandbox.py,
  tool_agent.py, prompts.py; scripts/smoke_llm.py; .venv-llm(vllm 0.30 서버)

## 목적

리더보드 1위 하네스(Duck: 27B + 파이썬 REPL 툴)를 로컬 5090에서 돌려 기준선을 잡고,
우리 강점(이동 게임 내비게이션)을 헬퍼 `nav`로 넣었을 때 얻는 이득을 같은 조건에서
측정한다.

## 방법

1. **재현 환경**: `uv sync --locked`(Python 3.12.12)로 하네스 venv, 모델 서버는
   우리 `.venv-llm`의 `vllm serve`(OpenAI 호환, tool parser qwen3_coder, reasoning
   parser qwen3, 32k 컨텍스트). NVML 불일치 우회를 venv 전역 `.pth` 심(`vllm_nvml_shim`)
   으로 옮겨 서버 프로세스에도 적용. 설정 `configs/local5090.json`
   (게임 1개, 1패스, 동시 1, 게임당 20분).
2. **모델**: (a) Qwen3.5-35B-A3B GPTQ-int4(우리 Kaggle 모델), (b) Duck 원본에
   가까운 Qwen3.6-27B AWQ(Kaggle 미러 `bachhg/qwen3-6-27b-awq`, 19GB; 5090에서 KV
   95k 토큰).
3. **nav 헬퍼**: v006~v011의 인코더/내비게이터를 `transitions`(전/후 프레임)에서
   매 호출 다시 만드는 무상태 클래스로 이식. `nav.summary()`, `nav.map()`,
   `nav.path_to(row,col)`, `nav.targets()`, `nav.frontier()`, `nav.gauge()`.
   샌드박스 전역에 주입하고, 시스템 프롬프트 addendum + 매 턴 유저 프롬프트에
   "Navigation helper summary" 블록을 호스트 측에서 계산해 삽입.
   오프라인 테스트: ls20에서 방향 6회 탐색 → 수정 타일 경로 6 → 자물쇠 경로 7,
   총 19액션으로 레벨 1 완료.

## 결과

| 설정 | ls20 | m0r0 | vc33 | 비고 |
|---|---|---|---|---|
| Duck 저자 example-run (Qwen3.6-27B FP8, 20패스 평균) | 0.35 (0.2 lv) | 0.05 (0.1 lv) | 3.33 (1.6 lv) | 25게임 평균 1.60±0.45, 라벨 "0-history-turns" |
| Duck 원본 + 35B-A3B GPTQ (로컬, 1패스 20분) | 0.00 (0 lv, 972액션, 72턴) | – | – | 마지막 턴 UP×20 연타, 월드모델 텍스트 비어 있음 |
| Duck 원본 + Qwen3.6-27B AWQ (로컬, 1패스 20분) | 0.00 (0 lv, 31액션, 28턴) | – | 0.00 (0 lv, 19액션) | 월드모델 텍스트는 정확(5×5 블록 아바타, 통로/벽, 이동 5칸)하나 사고 토큰으로 64 tok/s → 20분에 액션 31개뿐 |
| **Duck + nav (하이브리드) + 27B AWQ (로컬, 1패스 20분)** | **3.57 (lv1을 20액션, 기준 22)** + lv2 81액션 진행 중 타임아웃 | **0.36 (lv1을 109액션, 기준 30)** + lv2 진행 | 0.00 (lv1에 102액션) | 같은 모델·시간에서 이동 게임 0→레벨 1 |

관찰: 35B-A3B는 빠르지만(180 tok/s, 972액션) 월드모델을 못 쓰고 UP 연타로 빠지고, 27B는
정확하지만 로컬 단일 게임에선 너무 느리다. Duck의 Kaggle 설정은 28게임 동시·게임당 132분이라
처리량 문제를 동시성으로 푼다. 로컬 반복 실험은 35B-A3B, 최종 확인은 27B로 나눈다.

우리 프로그램형 v011(LLM 없음)은 ls20 3.57, m0r0 4.19, vc33 0.4±로, Duck이 약한
이동 게임에서 강하고 Duck이 강한 클릭 게임(vc33 3.33)에서 약하다 — 상보적.

### 하이브리드 25게임 (27B AWQ, 1패스, 게임당 20분, 동시 4, 2h20m)

- **25게임 평균 0.52**, 득점 5게임: tn36 3.57(lv1 10액션/기준 32), s5i5 2.78(20/20),
  sb26 2.78, ls20 2.77(25/22), su15 1.05. 나머지 20게임 0 (전부 20분 만료 `gave_up`)
- 동시 4개가 64 tok/s 서버를 나눠 써 게임당 액션 30~170. 단일 실행에서 풀리던 m0r0는
  34액션에서 만료(0) — 예산 부족과 nav 효과가 섞여 있음
- 대조군(원본 Duck, 동일 설정)은 다른 세션의 GPU 실험 뒤에 실행 예정. 참고치:
  우리 v011(LLM 없음) 0.37, Duck 저자 실행(게임당 100~200액션, 20패스) 1.60

### 대조군: 원본 Duck 25게임 (동일 조건, 16:10–18:30)

- **25게임 평균 0.92**, 득점 7게임: sp80 4.31, sc25 3.86, ls20 3.57, tn36 3.57, lp85 2.78,
  sb26 2.78, su15 2.22. 총 2,133액션, 694k 토큰
- 같은 예산의 "하이브리드"(실제로는 nav 요약 텍스트만, 아래 정정 참조) 0.52·5게임보다 높다.
  단일 런 비교(n=1)라 유의성은 없지만, **nav 요약 텍스트를 프롬프트에 넣는 것만으로는
  이득이 없거나 손해**일 수 있다(프롬프트 길이 증가 → 턴당 토큰·시간 증가). 단일 게임에서
  "Duck 단독 ls20 0"이었던 것도 예산 노이즈였다(여기선 46액션에 lv1)
- 따라서 nav의 가치는 도구(`nav.path_to`)로 실제 호출될 때 재야 하고, 그 측정이 v012
  4게임 검증(진행 중)이다

### 정정: 샌드박스 안의 `nav`는 None이었다 (2026-09-23 15:20 발견)

Duck의 파이썬 툴 샌드박스는 `python -I`(격리)로 뜨므로 `from inference.agent.nav_helpers import
build_nav`가 조용히 실패해 **모든 실행에서 샌드박스의 `nav`가 None**이었다. 하이브리드 25게임
트랜스크립트 25개 중 20개에 `'NoneType' object has no attribute 'path_to'` 오류가 있고, 모델은
`nav.path_to`를 1,700회 넘게 시도했다. 실제로 동작한 것은 호스트가 매 턴 유저 프롬프트에 넣은
"Navigation helper summary" 텍스트뿐이다. 따라서 위 표의 "Duck + nav" 결과(단일 게임 ls20 3.57,
m0r0 0.36, 25게임 0.52)는 **"Duck + nav 요약 텍스트"**의 결과이며 `nav.path_to`를 도구로 쓴
결과가 아니다. 수정(nav 소스를 초기 페이로드로 전달해 샌드박스에서 exec)은 v012 브랜치에 있고,
nav가 실제로 살아 있는 하이브리드 재측정이 필요하다. 요약 텍스트만으로도 이동 게임이 0→레벨 1이
된 점은 유효한 관찰이다.

### Kaggle 패키징 실행 (2026-09-23 밤) — 커널 `sungsuhyun/taaf-duck-nav-solver`

- `make -C harness/duck/ARC3-Inference kaggle-duck …`로 소스 데이터셋
  (`sungsuhyun/taaf-kaggle-source-nav-solver`, 1.8MB, 우리 nav/solver 포함)과 커널을 푸시.
  공개 wheelhouse·FP8 모델 스냅샷 첨부, machine_shape NvidiaRtxPro6000
- 노트북 템플릿의 커스터마이즈 훅에 쿼터 가드 추가: 비재실행(커밋)이면 경쟁 데이터셋의
  `environment_files`로 2게임만 15분씩(v1은 `__auto__` env-dir 오류로 실패 → v2에서 수정)
- **v2 커밋 런 성공**: RTX PRO 6000에서 wheelhouse 설치 → vLLM 서버(Qwen3.6-27B FP8) 기동·
  스모크 응답 → 우리 하네스가 tn36·lf52를 15분 플레이(0레벨, 59 tok/s, 동시 2) → 정리.
  재실행(제출)은 게이트웨이 Arcade·전체 설정(동시 28, 게임당 132분)을 그대로 사용 →
  **제출 가능한 커널 버전 확보(v2 = v012d 코드)**

### 라이선스·표기 검토 (2026-09-24)

- 번들 코드에 "duck"(9파일)·"Tufa"(16파일) 표기, 노트북 첫 셀은 TAAF 생성 문구, 커널 슬러그도
  `taaf-duck-…`였다. 채점·제출에는 무관(커널 비공개)하나 표기·기여 구분이 약했다
- 라이선스: GitHub 원본에 LICENSE 파일 없음(이슈 #6), pyproject에 MIT 분류자만. 같은 팀의
  Kaggle 소스 번들 `jeroencottaar/taaf-kaggle-source-share`는 **MIT** 명시 + 공개 오픈소스 선언
  → MIT 조건(고지 유지)으로 사용. ARC Prize 약관은 제3자 코드 조항 없이 "권리 침해 금지"만
- 조치: `harness/duck/NOTICE.md`(출처·MIT 근거·우리 기여 목록) 추가, 노트북 첫 셀을 우리
  솔루션 설명 + TAAF 출처로 교체, 커널/데이터셋 슬러그를 `arc3-nav-solver` /
  `arc3-nav-solver-source`로 변경, 우리 파일의 "duck" 표현 정리. **새 커널 `sungsuhyun/arc3-nav-solver`
  v1 커밋 런 성공**(11:30–12:15, 2게임 스모크) → 다음 제출은 이 커널로

### Kaggle 패키징 경로 (조사)

Duck의 `make kaggle-duck`는 (1) 소스 번들 데이터셋(우리 harness/duck 스냅샷)을 올리고,
(2) 공개 데이터셋 `driessmit1/arc3-vllm-h100-wheelhouse-v3`(5.2GB, vllm 0.19 + torch 2.10 +
flashinfer 0.6.6, RTX PRO 6000 검증)와 `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`(36GB)를
첨부한 커널을 푸시한다. 노트북 안에서 vLLM 서버를 띄우고(64k 컨텍스트) 재실행이면
게이트웨이 Arcade를, 아니면 번들 오프라인 게임을 플레이. 인증은 KAGGLE_USERNAME +
KAGGLE_API_TOKEN으로 우리 토큰과 호환. 주의: 커밋(비재실행) 런도 기본 540분 캡으로
게임을 다 돌리므로 주간 GPU 쿼터(6h)를 넘긴다 → 노트북 커스터마이즈 훅에서
비제출 모드일 때 게임 2개·15분으로 제한해야 한다.

## 결론 / 다음 단계

- **nav 헬퍼는 Duck의 약점(이동 게임)을 메운다**: 같은 27B·같은 20분에서 ls20 0→3.57
  (사람 기준 22액션에 20액션), m0r0 0→0.36. 클릭 게임(vc33)은 영향 없음(예상대로).
  n=1이므로 25게임·시드 반복으로 확인 필요
- 다음: (1) 하이브리드 25게임 동시 4·20분 1패스 → 원본 Duck 동일 조건과 비교,
  (2) Kaggle 패키징(Duck의 `make kaggle-duck` 경로: 소스 데이터셋 + vLLM wheelhouse +
  모델 스냅샷), (3) 클릭 게임용 헬퍼(우리 ClickSweeper의 게이지 인식·버킷 기록)를
  `clicks`로 추가하는 것은 데이터가 더 나온 뒤 결정
