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

### 로컬 GPU 스모크 (RTX 5090 31GB, 2026-09-22 오후)

RTX6000 대기열을 기다리지 않고 같은 코드 경로를 로컬에서 검증 (`make llm-venv`,
`make smoke-local MODEL_PATH=~/models/qwen3.5-35b-a3b-gptq-int4`,
scripts/smoke_llm.py). FP8 35B(37GB)는 31GB에 안 들어가 같은 아키텍처의
GPTQ-int4 35B-A3B(Kaggle 미러 `awooooo/…/other/gptq-int4/1`, 22.8GB) 사용.

| 시도 | 실패 원인 | 조치 |
|---|---|---|
| 1 | vLLM이 NVML로 GPU를 탐지하는데 드라이버/라이브러리 불일치(재부팅 전)로 플랫폼 미탐지 | 스모크 스크립트에서 non-NVML CUDA 플랫폼 강제 + 엔진 in-process(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) |
| 2 | `max_num_seqs=256 > Mamba cache blocks 164` (Qwen3.5 하이브리드 DeltaNet) | **에이전트**: `max_num_seqs=4` (QWEN_MAX_SEQS) |
| 3 | FlashInfer 샘플러 JIT가 nvcc/ninja 요구, 빌드 실패 | **에이전트**: `VLLM_USE_FLASHINFER_SAMPLER=0`, GDN prefill `triton` (QWEN_GDN_PREFILL) |
| 4 | — | **성공**: 로드 91s(가중치 21GB, 15s + 컴파일/그래프), 호출당 **0.6s**, 구조화 JSON 계획 3/3 정상 |

2·3은 Kaggle에서도 그대로 터질 수 있는 문제라 에이전트 코드(v005/v006/my_agent)에
반영. wheel 캐시에 ninja 추가(커널 v2).

### Kaggle 스모크 3차 (커널 v4, `machine_shape: NvidiaRtxPro6000`)

- **RTX PRO 6000 Blackwell Server Edition, 97,887 MiB** 배정 확인 — 48GB가 아니라
  96GB (docs/005의 가정 수정). FP8 35B는 물론 bf16 35B(70GB)도 올라가는 크기
- 설치 128s, vllm import 통과, 모델 경로/백엔드(vllm) 선택 정상
- 플래너 호출은 `PIL._typing._Ink` ImportError로 실패 — 원인은 노트북 커널
  프로세스가 이미지의 옛 PIL을 이미 import한 상태(matplotlib inline)에서 pillow를
  재설치했기 때문. 경쟁 재실행은 `python main.py` 새 프로세스라 영향 없음.
  스모크 셀도 서브프로세스로 돌리게 수정(커널 v5)
- **리더보드 제출 완료**: 커널 v4 → submission 56466876 (2026-09-22 14:19 UTC,
  "v006 nav-memory + Qwen3.5-35B-A3B FP8 in-process vLLM"), → **public 0.15**
  (2026-09-23 채점 완료, 첫 리더보드 점수). CLI 메시지 "0 submissions remaining today"는 제출 후 남은 일일
  한도(1회/일) 안내였음

## 결론 / 다음 단계

- 오프라인 LLM 스택은 Kaggle에서 import까지, 로컬 GPU에서 생성까지 검증됨. 미해결: RTX6000 배정 —
  kernel-metadata `machine_shape` 값(샘플은 `NvidiaTeslaT4`)의 RTX6000 명칭을
  확인해야 실제 vLLM 로드·latency 실측 가능. 값이 틀리면 조용히 T4로 배정됨
  → 해결: `machine_shape: "NvidiaRtxPro6000"` (build_notebook.py가 동기화), 96GB 배정 확인
- T4×2 경로가 필요하면 Qwen3.5-9B(bf16 18GB, TP=2) 또는 4bit 35B(GPTQ 미러
  `awooooo/qwen3-5-35b-a3b-gptq-int4`, ~20GB)로 교체 — 단 9b는 v004에서
  점수가 L1보다 나빴음
- 전략 개선은 v006(docs/011)으로
