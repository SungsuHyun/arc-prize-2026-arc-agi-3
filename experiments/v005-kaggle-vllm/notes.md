# v005-kaggle-vllm

## 가설

v004의 전략을 바꾸지 않고 L2 호출 경로만 교체하면, 로컬(ollama 35b)에서 얻은
점수를 Kaggle(인터넷 차단, RTX6000)에서 재현할 수 있다. 핵심은 (1) 가중치를
Kaggle Model로 첨부, (2) vLLM을 오프라인 wheel로 설치, (3) 프로세스당 1회
모델 로드.

## 변경점 (v004 대비)

- QwenPlanner 백엔드 분리: `ollama`(HTTP) / `vllm`(in-process, JSON 스키마
  구조화 출력) / `hf`(transformers generate, 폴백) / `none`. `QWEN_BACKEND`
  또는 자동 감지(`QWEN_MODEL_PATH` 또는 /kaggle/input/models 아래 체크포인트)
- 모델 엔진 모듈 전역 캐시 — 게임마다 새 MyAgent가 생겨도 재로드 없음
- JSON 파싱 강건화 (`<think>` 제거, 첫 `{`부터 raw_decode)
- metrics에 llm_backend / llm_load_s / llm_last_error 추가
- 인프라: `notebooks/wheels/`(인터넷 켠 커널에서 `pip download vllm` →
  출력 3.6GB, 196 wheel), `build_notebook.py`에 MODEL_SOURCES/KERNEL_SOURCES/
  스모크 테스트 셀, `scripts/kaggle_log.py`(커널 로그 조회)

## 관찰

- 로컬 ollama 백엔드: v004와 동일 동작. 4게임 벤치(bench-20260922-071959,
  400스텝): vc33 레벨 2 · 0.321, 나머지 0 → 종합 0.080 (v004 재현)
- Kaggle 스모크 1차(T4×2): vllm 0.30 오프라인 설치 OK(269s), 모델 경로 자동
  탐색 OK, `from vllm import LLM`에서 pillow `_Ink` ImportError
- Kaggle 스모크 2차: pillow 제거·재설치로 import 통과(vllm 0.30 / transformers
  5.17 / torch 2.13+cu130 / cuda True, 설치 223s). 그러나 `--accelerator
  NvidiaRtx6000`이 무시되어 여전히 T4×2(29GiB) → GPU 용량 가드가 35GiB
  모델 로드를 건너뛰고 L2 비활성(backend=none)으로 안전 폴백

## 결론

- 오프라인 LLM 스택(wheel 캐시 + 모델 첨부 + in-process 백엔드)은 Kaggle에서
  동작 확인. 남은 것은 RTX6000 배정 방법(machine_shape 값) 확인 후 실제
  vLLM 로드/latency 실측
- 전략은 v004 그대로이므로 점수 변화 없음. 후속은 v006(내비게이션)
