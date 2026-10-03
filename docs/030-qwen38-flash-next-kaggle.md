# 030. Qwen3.8-Flash-Next를 Kaggle 단일 RTX Pro 6000에서 서빙하기까지 (v6→v17)

## 목적

사용자 요청으로 pbg 제출 커널의 LLM을 Qwen3.6-27B-FP8에서 **Qwen3.8-Flash-Next**(125B MoE, 아키텍처 `qwen4_exp`,
멀티모달, 51B 파라미터 PLE n-gram 임베딩)로 교체한다. Kaggle 재실행은 인터넷 차단·단일 GPU(RTX Pro 6000 Blackwell,
96 GB)라, 가중치와 런타임을 모두 Kaggle 입력으로 첨부해 오프라인으로 띄워야 한다. 이 문서는 커널 v6부터 v17까지
12번의 commit 런(각 ~15–65분 + 큐 대기)에서 만난 블로커를 **하나씩** 로그로 확정하고 해결한 기록이다.

## 방법

- 변수는 한 번에 하나만 바꾸고, 매 런의 `vllm-server.log`와 papermill 로그를 내려받아 실패 지점을 확정한 뒤 다음 수정.
- v14부터 준비 대기 루프에 5분 간격 하트비트(마지막 vLLM 로그 줄)를 넣어 "조용한 멈춤"의 위치를 특정.
- 각 블로커는 웹에서 동일 하드웨어 사례·vLLM 이슈로 교차 확인한 뒤에만 수정 적용(1시간짜리 런을 추측으로 태우지 않음).
- 빌더: `scripts/build_arcnav_notebook.py`의 프리셋 `qwen38fn`(runtime="image") + `scripts/build_pbg_notebook.py`의
  전용 셀 `image_vllm_cell`. arcnav/qwen27b 라인은 회귀 체크로 불변 확인.

## 결과

### 블로커와 해결 (버전별)

| 버전 | 도달 지점 | 실패 | 원인 | 해결 |
|---|---|---|---|---|
| v6 | pip | `accelerate==1.14.0` 없음 | 휠하우스의 lock이 루트에, 휠은 `wheels/wheelhouse/` 하위(find-links 비재귀) | 휠 디렉터리 자동탐지 |
| v7 | pip | `vllm` 없음 | 〃 | 〃 |
| v8 | vLLM 기동 | `model type qwen4_exp ... Transformers does not recognize` | **transformers 5.15.1 + vllm 0.27.1조차 qwen4_exp 미지원** → 어떤 pip 휠하우스로도 불가 | 공식 Docker 이미지 `vllm/vllm-openai:qwen38-flash-next`의 레이어 blob(`keithtyser/...runtime-v1`)을 런타임에 추출 |
| v9 | 이미지 추출 | vllm 못 찾음 | 이미지는 `dist-packages`에 설치(`site-packages`만 탐색) | 탐색 범위 확대 |
| v10 | 모델 로드 | OOM 95.37 GiB (`vocab_parallel_embedding`) | **51B PLE n-gram 테이블**이 experts와 함께 96 GB에 안 들어감 | `VLLM_PLE_CPU_OFFLOAD=1`(host RAM으로 오프로드) |
| v11 | MoE dummy run | `No supported CUDA architectures for major versions [12]` | sm120 FlashInfer JIT가 Kaggle의 구버전 nvcc를 집음 | `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `FLASHINFER_FORCE_SM=120f`, `TORCH_CUDA_ARCH_LIST=12.0`, 이미지의 cuda-13.0 nvcc를 CUDA_HOME/PATH에 |
| v12–v13 | 엔진 init 완료 | 그래프 캡처 후 30–37분 무로그, 포트 미개방 → 타임아웃 | (v14에서 autotune OFF로도 재현 → autotune은 멈추는 *위치*일 뿐) | 대기 연장은 무의미했음 |
| v14 | 〃 | 〃 (`Free memory on device` 줄에서 30분 정지) | **vllm#53960**: 단일 GPU 기본 executor가 PLE 오프로드 워커를 스폰하지 않아 warmup forward가 영원히 대기 | `--distributed-executor-backend mp`, `VLLM_PLE_OFFLOAD_READY_TIMEOUT=5400`, `--no-enable-flashinfer-autotune`(sm120 출력 손상 방지) |
| v15 | PLE 워커 가중치 적재 | `no module or parameter named 'ngram_embedding.weight_scale'` | **vllm#54765(미머지, 우회법 없음)**: 공식 NVIDIA NVFP4의 PLE 테이블은 FP8 양자화 → 오프로드 경로에 `weight_scale` 슬롯 없음 | PLE가 **BF16**인 체크포인트로 교체: `lordhansolo/swift-1-5-qwen3-8-flash-next-nvfp4/pyTorch/hf-nvfp4/1`(NVFP4 experts, `hf_quant_config`가 `layers.1.ple*` 제외; 81 샤드, 단일 디렉터리, index.json) |
| v16 | PLE↔GPU 워커 핸드셰이크 | `pidfd_getfd: Operation not permitted` | 두 원인: ① 내가 v11에 추측으로 넣은 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 CUDA IPC를 fd 핸들로 바꿈(pytorch#165685; 메모리는 75.7/88 GiB로 불필요했음) ② Yama ptrace_scope=1은 형제 프로세스의 ptrace를 거부 | ① 제거 ② PYTHONPATH 앞에 `sitecustomize.py` 주입 → 모든 vLLM 프로세스가 `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`(peakcrosser7/vllm#12 "ptrace opt-in") |
| **v17** | **COMPLETE** | — | — | 처음으로 노트북 완주 |

### v17 측정값

커널 v17(커밋 cb51ae4), 2026-10-03, commit 모드 smoke(2게임 × 12분, jobs=2):

- **기동**: `vLLM ready after 945s`(15.8분; 하트비트 — 5분 54%·10분 80% 샤드 로드, 15분에 `shm_broadcast ... No available
  shared memory broadcast block found in 60 seconds` 1회 = 워밍업 중 일시 대기였고 교착 아님). 노트북 시작 기준 ≈ 추출 3분 + 16분.
- **smoke 채팅**: `'Greetings to you! 👋'` — 모델이 실제 응답.
- **PLE 워커**: `PleOffload: spawning worker` → gloo 분산 초기화 → PLE 가중치 탐색 → GDN 커널 선택까지 정상, pidfd/EPERM/OOM/ERROR 없음.
- **2게임 smoke**: `smoke errors: []`, `done in 30.0 min`.

  | 게임 | 레벨 완료 | 점수 | 액션 | LLM 호출 | 종료 |
  |---|---|---|---|---|---|
  | r11l | 1 | 4.762 | 63 | 84 | timeout(12분 캡) |
  | vc33 | 1 | 0.280 | 57 | 84 | timeout |

  합계 2.521, level≥2 0/2. LLM 84회/12분 ≈ 7회/분/게임 — pbg 가설 루프를 구동할 만한 응답 속도.
- `submission.parquet`(commit 모드 더미)와 `pbg-results/run-20261003-025821-65.json` + logs 생성.
- **비교**: 27B 커널의 smoke(v3–v5)도 "r11l L1 + vc33 L1, 에러 없음"이었다 → 이 smoke 기준으로는 **동일한 L1 클리어**.
  우열은 미측정이며, 리더보드 전이도 미지(27B v5는 리더보드 0.04).

### 머신·체크포인트 사실 (로그로 확정)

- Kaggle RTX Pro 6000 박스: VRAM 96 GB(94.97 GiB 가용), **host RAM 131–177 GiB**(런마다 다름), 체크포인트는 느린 NFS
  (샤드 로드 9–24분, 런마다 2–3배 편차; 123 GiB 체크포인트가 RAM 90%를 넘어 자동 프리페치 생략).
- 가중치 로드 후 GPU 사용 ~75.7 GiB(experts NVFP4), KV 캐시 10.83 GiB(max-model-len 32768 기준 375k 토큰, 11.46x 동시성),
  CUDA 그래프 0.46 GiB. BF16 PLE는 host RAM ~95–100 GiB.
- 공식 `nvidia/qwen3-8-flash-next-nvfp4`는 **단일 GPU에서 영구히 불가**(FP8 PLE + #54765 미머지). 다시 시도하지 말 것.
- 이미지 런타임: 6개 레이어 tar.gz ≈ 17 GB 추출(멤버별 추출로 디바이스/whiteout 항목 무시), python 3.12로 커널과 ABI 일치,
  nvcc는 `/tmp/vllm-image/usr/local/cuda-13.0/bin/nvcc`.

### 과정상의 실수와 재발 방지

- v17 커밋 체인에서 검증 히어독 뒤에 `&&` 없이 `git …`을 이어 써, 검증이 실패했는데도 커밋·푸시가 진행되고 커밋 메시지가
  실행되지 않은 검증을 주장했다(검증 실패 자체는 주석을 매칭한 오탐이었고, 이후 AST 기반 재검증은 전부 통과).
  → `scripts/verify_pbg_notebook.py` + `make pbg-verify`를 `pbg-submit`의 선행 조건으로 연결(빌드→검증→푸시;
  긍정·부정 테스트로 게이트 동작 증명).

## 결론 / 다음 단계

- Qwen3.8-Flash-Next는 Kaggle 단일 RTX Pro 6000에서 **서빙 가능**하되, 조건이 전부 필요하다: 공식 vllm-openai 이미지 런타임,
  BF16-PLE 체크포인트, PLE host-RAM 오프로드, `mp` executor, sm120 arch 명시 + 이미지 nvcc, autotune OFF,
  expandable_segments 금지, ptrace opt-in.
- 비용: 기동이 길다(추출 + 로드 + 워밍업). 경쟁 재실행(540분 한도, `RERUN_TOTAL_MINUTES=470`)에서 게임 시간이 그만큼 줄어든다
  — 재실행 예산 재점검 필요.
- 리더보드 제출은 수동(`kaggle competitions submit -k sungsuhyun/arc3-pbg -v 17`); 이 문서 작성 시점에 제출하지 않았다.
  27B 대비 실제 점수 차이는 미측정.
