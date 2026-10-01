# 🏆 Qwen3-VL 기반 한국어 Scene-Text Multiple Choice VQA 성능 개선

> **SSAFY AI Challenge 프로젝트 (2026.09)**
> 
> **Kaggle Public Score: 0.96127 달성 (최고 점수)**
> 
> *"단순 모델 스케일업보다 효과적이었던 OCR Localization & 선택적 보정 전략"*
> 

---

## 📌 1. Project Overview

본 프로젝트는 이미지와 질문, 4개의 객관식 선택지가 주어졌을 때 정답(`a / b / c / d`)을 예측하는 멀티모달 VQA(Visual Question Answering) 모델 개발 과제입니다.

* **과제 특성**: 일반 사물 인식보다 간판 상호명, 전화번호, 가격표, 도로 표지판, 메뉴판 등 이미지 속 미세 글자를 정확히 읽어내야 하는 **Scene-Text OCR 중심 과제**입니다.


* **라벨 분포**: Train 데이터의 정답 위치 분포는 a(25.3%), b(24.5%), c(25.6%), d(24.6%)로 균등하여, 라벨 편향보다는 **시각 해독 역량이 승부처**였습니다.


* **핵심 가설**: "단순 모델 파라미터 확장보다, 미세 글자 영역의 고해상도 시각 가이드 확보와 Attention 유도가 성능의 핵심이다"라는 가설을 세우고 파이프라인을 구축했습니다.



---

## 📂 2. Directory Structure

대용량 데이터셋과 모델 가중치 파일은 `.gitignore`에 등록하여 형상 관리에서 제외하고, 코드 및 파이프라인 노트북을 중심으로 관리합니다.

```text
AI2_Challenge/
├── data/                                 # [Git LFS / 제외] 대회 데이터셋 디렉터리
│   ├── train/                            # 학습용 이미지 폴더 (6,714개)
│   ├── dev/                              # 검증용 이미지 폴더 (2,683개)
│   ├── test/                             # 평가용 이미지 폴더 (6,714개)
│   ├── train.csv                         # 이미지, 질문, 선택지, 정답 라벨
│   ├── dev.csv                           # 복수 Annotator 응답 데이터
│   ├── test.csv                          # 테스트 질문 및 선택지
│   └── sample_submission.csv
├── downloads/                            # [제외] 사전 다운로드 모델 및 캐시
│   ├── models/                           # HuggingFace 사전학습 가중치 (Qwen3-VL 등)
│   └── libs/                             # 오프라인 패키지 파일
├── output/                               # 학습 체크포인트 및 추론 제출 파일
│   ├── qwen2_5_vl_3b_lora/               # [제외] 모델 체크포인트
│   ├── submission.csv                    # 최종 제출 파일 (Score: 0.96127)
│   ├── submission_baseline.csv 
│   └── submission_7b.csv
├── notebooks/                            # 단계별 실험 및 파이프라인 노트북
│   ├── (260827)_baseline_desktop...ipynb # 로컬 베이스라인 테스트
│   ├── (260908)_baseline_colab.ipynb     # Colab 학습 환경 구축
│   ├── (260911)_download_libs_models.ipynb # 모델 및 라이브러리 프리패치
│   └── Qwen3VL_VQA_FINAL2_ChoicePerm.ipynb # 최종 TTA 및 Vote-Switch 파이프라인
├── requirements.txt                      # 환경 구성 의존성 목록
├── .gitignore                            # 대용량 바이너리/데이터셋 제외 설정
└── README.md

```

(참고: 레포지토리 내 실제 폴더 구조를 기반으로 구성되었습니다.)

---

## ⚙️ 3. Environment & Data/Model Preparation Guide

### 3.1. 의존성 설치

```bash
pip install -r requirements.txt

```

### 3.2. 대회 데이터셋 준비 (`data/`)

대회 규정 및 보안 유지를 위해 원천 데이터는 레포지토리에 포함되지 않습니다. 공식 대회 플랫폼에서 다운로드한 데이터를 아래 구조로 `data/` 경로에 배치해야 합니다:

* `data/train.csv` (6,714건)


* `data/dev.csv` (2,683건)


* `data/test.csv` (6,714건)


* `data/train/`, `data/dev/`, `data/test/` 이미지 폴더



### 3.3. 사전학습 모델 다운로드 (`downloads/models/`)

HuggingFace로부터 사전학습 모델 가중치를 다운로드하여 `downloads/models/` 경로에 배치합니다:

* 모델 가중치 파일(`.safetensors`, `.bin`)은 대용량 파일이므로 `.gitignore`를 통해 자동 제외됩니다.


* 주요 활용 모델: `Qwen/Qwen3-VL-8B-Instruct`


---

## 🚀 4. Key Engineering & Methodologies

### 4.1. Pipeline Redesign (베이스라인 취약점 개선)

* **Full-Train 전환**: 초기 베이스라인의 200개 샘플 학습에서 벗어나 6,714개 전체 데이터 학습으로 전면 전환했습니다.


* **Answer-Only Loss**: 전체 프롬프트 토큰에 Loss를 주던 방식에서 벗어나, 질문과 선택지 토큰은 `-100`으로 마스킹하고 정답 토큰(`a/b/c/d`)에만 CrossEntropy Loss를 집중 적용했습니다.


* **Direct Choice Logit Scoring**: `model.generate()` 출력 파싱 시 발생하는 형식 오류를 0%로 제거하고, 모델 출력 로짓에서 `a / b / c / d` 토큰 로짓을 직접 비교하여 추론 속도와 정답 수렴도를 극대화했습니다.



### 4.2. Resolution Tuning (최적 Visual Token 규명)

* 무조건적인 해상도 증가가 성능 향상으로 이어지지 않음을 확인했습니다.


* 1024 토큰(0.95770) 대비 1536 토큰(0.95859)으로 확장 시 성능이 향상되었으나, 2048 토큰(0.95770) 세팅에서는 불필요한 연산량만 증가하고 점수가 하락하여 **1536 Visual Tokens을 Sweet Spot**으로 채택했습니다.



### 4.3. OCR 활용 패러다임 전환: Prompt Injection vs BBox Crop

* **실패 (OCR Text 직접 주입: 0.95442)**: EasyOCR로 추출한 텍스트를 프롬프트에 직접 삽입했으나, OCR 자체 오인식 노이즈(예: `1699` → `1689`)가 VLM의 정상적인 시각 추론을 방해하여 점수가 급락했습니다.


* **성공 (V3 Question-Aware BBox Crop: 0.95918)**: OCR의 역할을 텍스트 생성기가 아닌 관심 영역 탐색기(Locator)로 재정의했습니다. 질문·선택지와 관련성이 높은 영역만 BBox Crop하여 원본 이미지와 함께 다중 입력(Multi-Image)함으로써 모델 자체 판독력을 극대화했습니다.



### 4.4. Model Scaling Analysis (8B Dense vs 30B MoE)

* 동일 파이프라인(1536 + OCR Crop) 조건에서 Qwen3-VL-8B (0.95918)와 Qwen3-VL-30B-A3B MoE (0.95233)를 비교 검증했습니다.


* 30B MoE 모델은 Dev 지표 일부가 상승했으나 Test 일반화에 실패하고 Public Score가 급락했습니다. 이를 통해 **단순 모델 스케일업보다 도메인 맞춤형 시각 가이드 설계가 훨씬 우수함**을 입증했습니다.



### 4.5. Choice Permutation TTA (선택지 위치 편향 상쇄)

* 객관식 문제에서 모델이 가지는 보기 위치 편향(Position Bias)을 해결하기 위해 선택지를 4가지 순서로 순환 회전(Cyclic Permutation)시켰습니다.


* 회전된 각 선택지를 원본 보기 기준 확률로 역매핑(Re-mapping)하여 위치 편향을 제거했습니다.


* Base V3: 0.95918


* Permutation Conservative: 0.96008


* Permutation Balanced: 0.96067





### 4.6. Winning Methodology: Vote-Switch Selective Correction

* 모든 예측값을 단순 평균하지 않고, 강력한 Baseline(V3)을 보존하면서 불확실한 소수만 타겟팅하는 선택적 보정 전략을 수립했습니다:


1. **다수 합의 조건**: 3개의 Permutation 결과 중 2개 이상이 동일한 대안을 선택


2. **확신도 격차 조건**: 대안의 평균 확률과 기존 V3 답 확률의 차이가 $P(\text{대안}) - P(\text{V3}) \ge 0.08$ 이상으로 명확한 경우




* 전체 6,714개 테스트 문제 중 **단 35개(약 0.52%)의 난제만 선별 수정**하여 **최종 최고 점수 0.96127**을 달성했습니다.



---

## 📊 5. Experiment Results & Kaggle Progression

| Version | 모델 및 핵심 기법 | 주요 특징 및 비고 | Public Score |
| --- | --- | --- | --- |
| **V1** | Qwen3-VL-8B (1024 Tokens) | Full Image 기준 모델

 | 0.95770

 |
| **V1.1** | Qwen3-VL-8B (1536 Tokens) | Visual Token 확장 (Sweet Spot 규명)

 | 0.95859

 |
| **V2** | Qwen3-VL-8B (2048 Tokens) | 고해상도 연산량 증가 대비 성능 정체

 | 0.95770

 |
| **V2 OCR** | 2048 Tokens + OCR Text Prompt | OCR 오인식 노이즈 유입으로 실패

 | 0.95442

 |
| **V3** | **1536 Tokens + Question-Aware OCR Crop** | **OCR Locator 전환 및 멀티 이미지 입력 (핵심 돌파구)**<br> | **0.95918**<br> |
| **V4** | Probability Ensemble | 동일 계열 모델(V1.1 + V3) 오류 패턴 중복으로 하락

 | 0.95740

 |
| **V5** | 4-Way CrossEntropy | Dev 과적합으로 Public Score 하락

 | 0.95740

 |
| **V6** | Qwen3-VL-30B-A3B (MoE) | 모델 스케일업 가설 기각 (과적합 발생)

 | 0.95233

 |
| **Top-2 Crop** | V3 + 2nd Candidate Crop TTA | 두 번째 Crop 정보 과잉 및 노이즈 추가

 | 0.95859

 |
| **TTA-Perm** | Choice Permutation Balanced | 선택지 위치 편향 상쇄

 | 0.96067

 |
| **Final** | **V3 + Vote-Switch (2/3 합의 선별 보정)** | **6,714건 중 35건(0.52%) 정밀 수정 (최고 성적 🏆)**<br> | **0.96127**<br> |

---

## 🖥️ 6. Hardware & Training Specs

* **GPU 인프라**: NVIDIA RTX PRO 6000 Blackwell Server Edition (VRAM 96GB)


* **정밀도 및 튜닝**: BF16 + LoRA 파인튜닝


* **배치 최적화**: Batch Size 4, Gradient Accumulation 2 (Effective Batch: 8) 세팅을 통해 1 Epoch당 약 1.83시간 소요 (Peak VRAM: 69~71GB)



---

## 💡 7. Key Takeaways

1. **OCR = Locator**: OCR 모델을 텍스트 추출기가 아닌 VLM이 주목해야 할 Bounding Box 제안 도구로 활용할 때 가장 강력했습니다.


2. **Sweet Spot = 1536**: 무조건적인 해상도 증가보다 모델과 문제에 최적화된 토큰 수 탐색이 중요했습니다.


3. **Selective Correction**: 전체 예측을 무리하게 흔들지 않고, 모델의 확신도와 Permutation TTA 합의를 기반으로 불확실한 소수(0.52%)만 정밀 교정하는 방식이 최고 성능(0.96127)을 견인했습니다.