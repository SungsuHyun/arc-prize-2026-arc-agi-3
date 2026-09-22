# 010. v005 Kaggle 패키징 — 오프라인 vLLM + 모델 첨부

- **날짜**: 2026-09-22
- **관련 파일/커밋**: experiments/v005-kaggle-vllm/, scripts/build_notebook.py,
  notebooks/wheels/, scripts/kaggle_log.py, 커밋 79f0e76

## 목적

로컬(ollama qwen3.5:35b)에서 0.321을 낸 v004 플래너를 인터넷이 차단된 Kaggle
재실행 환경에서 그대로 돌릴 수 있게 패키징한다. 전략 코드는 건드리지 않고
`QwenPlanner._complete()`의 백엔드만 교체하는 것이 목표.

## 방법

1. **모델**: Kaggle Models에서 `wukeneth/qwen3-5-35b-a3b-fp8` (Qwen3.5-35B-A3B-FP8
   공식 미러, 37.5GB) 발견 — 로컬 ollama의 qwen3.5:35b와 같은 계열. 이전 세션의
   "Qwen 3.8 27B" 요청도 Kaggle에 `Qwen3.8-27B-FP8` 미러가 여럿 있어 교체 가능
   (dense 27B라 MoE 35B-A3B보다 느림 → 일단 35B-A3B 유지)
2. **vLLM wheel**: 경쟁 데이터셋의 `arc_agi_3_wheels/`에는 vllm/torch가 없음.
   인터넷을 켠 별도 커널(`notebooks/wheels/`, `make wheels`)에서
   `pip download vllm --only-binary=:all:` → 출력(196 wheel, 3.6GB: vllm 0.30,
   torch 2.13+cu130, transformers 5.17)을 제출 노트북에 `kernel_sources`로 첨부
3. **에이전트(v005)**: 백엔드 `ollama | vllm | hf | none`. `QWEN_MODEL_PATH`
   (또는 /kaggle/input/models 자동 탐색)가 있으면 vllm→hf 순, 없으면 ollama.
   모델은 모듈 전역 캐시(프로세스당 1회 로드). vLLM은 JSON 스키마 구조화 출력,
   thinking 비활성. GPU 총 메모리의 90%보다 큰 체크포인트면 로드 시도 없이 L2
   비활성(T4 안전장치)
4. **노트북(build_notebook.py)**: `MODEL_SOURCES`/`KERNEL_SOURCES`/`SMOKE_TEST`
   상수. 설치 셀이 wheel 캐시에서 vllm 오프라인 설치, 환경 셀이
   `QWEN_MODEL_PATH` 내보내기, 커밋(비재실행) 모드에서는 실제 GPU로 플래너
   1회 호출 스모크 테스트를 수행하고 더미 submission.parquet 기록
5. `make kaggle-log` (scripts/kaggle_log.py): 출력 파일을 내려받지 않고 커널
   로그만 REST로 조회. 커널 푸시 시 `--accelerator NvidiaRtx6000`으로 가속기 지정
   (kernel-metadata의 `enable_gpu`만으로는 T4×2가 배정됨)

## 결과

| 항목 | 결과 |
|---|---|
| 로컬 ollama 백엔드 | v004와 동일 동작 (ls20 60스텝 스모크, 오류 0) |
| Kaggle 계정 | 실제 사용자명은 `sungsuhyun` (docs/001의 `monggree`는 오기) |
| 스모크 1차 (T4×2, 기본 GPU) | vllm 0.30 설치 성공(269s), 모델 경로 자동 탐색 OK. `from vllm import LLM`에서 `PIL._typing._Ink` ImportError — Kaggle 이미지의 pillow 잔재와 충돌 → 설치 셀에 pillow 제거·재설치 추가 |
| 스모크 2차 (`--accelerator NvidiaRtx6000`) | pillow 재설치로 `from vllm import LLM` 통과 (vllm 0.30 / transformers 5.17 / torch 2.13+cu130 / cuda True, 설치 223s). 그러나 가속기 지정이 무시되어 여전히 **T4×2(29GiB)** → GPU 용량 가드가 35GiB 모델 로드를 건너뛰고 `backend=none`으로 안전 폴백 (호출 0, 오류 0) |
| 로컬 4게임 벤치 (bench-20260922-071959) | v004 재현: vc33 lv2 0.321, 종합 0.080 |

## 결론 / 다음 단계

- 오프라인 LLM 스택은 Kaggle에서 import까지 검증됨. 미해결: RTX6000 배정 —
  kernel-metadata `machine_shape` 값(샘플은 `NvidiaTeslaT4`)의 RTX6000 명칭을
  확인해야 실제 vLLM 로드·latency 실측 가능. 값이 틀리면 조용히 T4로 배정됨
  → 확인: `machine_shape: "NvidiaRtxPro6000"` (build_notebook.py가 동기화). 3차 스모크 진행 중
- T4×2 경로가 필요하면 Qwen3.5-9B(bf16 18GB, TP=2) 또는 4bit 35B(GPTQ 미러
  `awooooo/qwen3-5-35b-a3b-gptq-int4`, ~20GB)로 교체 — 단 9b는 v004에서
  점수가 L1보다 나빴음
- 전략 개선은 v006(docs/011)으로
