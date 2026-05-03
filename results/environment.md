# 실행 환경 점검 보고서

작성일: 2026-05-03
프로젝트: TabNet + SHAP + LLM 기반 XAI-RAG 신용 평가 (석사학위 논문)

---

## 1. 시스템 정보

| 항목 | 값 |
|---|---|
| OS | Windows 10 Pro (10.0.19045) |
| Shell | bash (MSYS) + PowerShell |
| Working Directory | `D:\paper` |

## 2. Python 환경

| 항목 | 값 |
|---|---|
| Python | 3.10.11 |
| 인터프리터 경로 | `C:\Users\ws059\AppData\Local\Programs\Python\Python310\python.exe` |
| 가상환경 활성화 | **없음** (시스템 Python 사용 중) |
| `VIRTUAL_ENV` | (unset) |
| `CONDA_DEFAULT_ENV` | (unset) |

> 계획서에서는 Python 3.11을 가정. 3.10.11도 호환 문제는 없음. 다만 venv 없이 시스템 파이썬에 설치하면 패키지 충돌 위험이 있어 venv/conda 권장.

## 3. GPU / CUDA

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti (Mobile/Desktop 추정) |
| 총 VRAM | **6,144 MiB (≈6 GB)** |
| 현재 사용 중 VRAM | 251 MiB (idle) |
| 드라이버 버전 | **460.89** (2020년 출시, 매우 구버전) |
| CUDA 런타임 (드라이버 기준) | 11.2 |
| 컴퓨트 캐퍼빌리티 | 7.5 (Turing) |

### ⚠️ 호환성 이슈
- 최신 PyTorch 2.x는 CUDA 11.8 / 12.1을 권장하며, Windows에서 CUDA 11.8 사용 시 드라이버 **452.39 이상** 필요(엄밀히는 PyTorch 빌드별 상이). 현재 드라이버 460.89는 CUDA 11.8 forward compatibility로 동작할 가능성이 있으나 **공식 보장 영역이 아님**.
- CUDA 12.x는 드라이버 525+ 권장 → 현재 드라이버로는 사실상 불가.
- 해결 옵션:
  1. **드라이버 최신화** (가장 안전, 권장)
  2. **PyTorch CUDA 11.8 설치 후 동작 테스트** (작동할 가능성 있음)
  3. **CPU 모드 진행** (TabNet/XGBoost/LightGBM 모두 CPU 가능. 시간 소요 증가하나 본 프로젝트 데이터 규모(≈300K행)에서는 수용 가능)

## 4. 패키지 설치 현황 (2026-05-03 업데이트)

가상환경: `D:\paper\.venv\` (Python 3.10.11)

### 설치 완료
| 패키지 | 버전 | 비고 |
|---|---|---|
| torch | 2.5.1+cu121 | CUDA 12.1 빌드 |
| pytorch-tabnet | 4.1.0 | |
| xgboost | 3.2.0 | |
| lightgbm | 4.6.0 | |
| shap | 0.49.1 | |
| optuna | 4.8.0 | |
| pandas | 2.3.3 | |
| scikit-learn | 1.7.2 | |
| seaborn | 0.13.2 | |
| numpy | 2.2.6 | |
| matplotlib | 3.10.9 | |
| pyarrow | 24.0.0 | parquet 지원 |
| category_encoders | 2.8.1 | 타깃 인코딩 |
| tqdm | 4.67.3 | |
| kaggle | 1.7.4.5 | API 토큰 필요 시 |
| anthropic | 0.97.0 | LLM 옵션 1 |
| google-genai | 1.74.0 | LLM 옵션 2 |
| python-dotenv | 1.2.2 | .env 로딩 |

전 패키지 import 스모크 테스트 통과.

## 5. PyTorch CUDA 인식 결과

```
torch.cuda.is_available() → False
경고: NVIDIA driver on your system is too old (found version 11020).
```

**원인:** 드라이버 460.89(CUDA 11.2)가 너무 구버전. PyTorch 2.5.1+cu121은 드라이버 525+ 권장.
**해결:** 사용자가 NVIDIA 드라이버 업데이트 후 동일 명령으로 재테스트:
```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0))"
```
업데이트 안 하면 CPU 모드로 작업 가능 (TabNet도 CPU 학습 됨, 시간만 좀 더 걸림).

## 6. 데이터 폴더 상태

`D:\paper\data\home_credit\` — **현재 비어있음**.

Home Credit Default Risk 데이터셋(Kaggle) 다운로드 필요. 본 프로젝트는 메인 테이블 `application_train.csv`(약 166 MB, 약 307K 행, 122개 컬럼)만 사용 예정.

## 7. 프로젝트 폴더 구조 (생성 완료)

```
D:\paper\
├─ paper/             # 계획서 docx 사본
├─ data/home_credit/  # 데이터 (다운로드 필요)
├─ src/               # 소스 코드
├─ thesis/            # CLAUDE.md, 논문 작업 노트
├─ results/           # 실험 결과 (이 파일 포함)
├─ figures/           # 시각화 산출물
└─ 석사학위_논문계획서_오현택_A70067.docx (원본)
```

## 8. 권장 batch_size (6GB VRAM 기준)

| 모델 | 권장 `batch_size` | `virtual_batch_size` (TabNet) |
|---|---|---|
| TabNet (n_d=n_a=8~16) | 1,024 | 128 |
| TabNet (n_d=n_a=32~64) | 256~512 | 128 |
| MLP (DNN baseline) | 512~1,024 | — |
| XGBoost / LightGBM (GPU) | n/a (CPU 권장) | — |

> Home Credit 메인 테이블 처리 시 RAM 사용량은 200~500 MB 수준이라 VRAM이 병목.
> Llama 3 8B/70B 로컬 실행은 6 GB VRAM에서 **불가능** → LLM은 OpenAI API 등 외부 호출 권장.
