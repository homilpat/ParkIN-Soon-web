# ParkIN Soon (박인순) - 파킨슨병 조기 선별 키오스크 및 AI 파이프라인

AI 기반 비침습적 기술을 사용하여 파킨슨병 관련 전구 증상(후각, 배변) 및 운동 증상(손그림 그리기)을 종합 분석하고 위험도를 평가하는 스크림릿(Streamlit) 웹 키오스크 애플리케이션 및 모델 학습 파이프라인입니다.

---

## 📂 프로젝트 구조

```text
├── kiosk.py                      # 메인 Streamlit 키오스크 애플리케이션
├── ParkinsoonAI.py               # 사이드바 상주형 AI 도우미 (음성 및 RAG 연결)
├── rag_chatbot.py                # RAG(Retrieval-Augmented Generation) 챗봇 엔진
├── requirements.txt              # 파이썬 의존성 패키지 목록
├── character.png                 # 인트로/챗봇용 안내 캐릭터 이미지
├── hospital_data.csv / .xlsx     # 고위험군 추천용 전국 신경과 좌표 데이터
├── LLM논문/                      # RAG 지식베이스 구축용 파킨슨 관련 의학 논문 PDF 폴더
│
├── modeling/                     # 학습 완료된 모델 및 벡터 DB 저장소
│   ├── olf_con_model.pkl         # 후각 + 배변 판별용 XGBoost/LR 모델
│   ├── drawing_cnn_model.h5      # 나선 손그림 이미지 분석용 CNN(MobileNetV2) 모델
│   ├── drawing_kinematic_model.pkl # 손 움직임(속도/가속도) 분석용 RF 모델
│   ├── fusion_config.json        # 후기 융합(Late Fusion) 가중치 설정 파일
│   └── vector_db.pkl             # 논문 텍스트 임베딩 벡터 DB 파일
│
└── 파이프라인 스크립트 (모델 학습 및 검증)
    ├── PPMI.py                   # 후각 및 배변 설문 데이터 분석 및 모델 학습 파이프라인
    ├── drawing_pipeline.py       # 나선 손그림 이미지 분류 CNN 학습 파이프라인 (Grad-CAM 포함)
    └── kinematic_pipeline.py     # 나선 그리기 좌표 데이터 피처 추출 및 RF 학습 파이프라인
```

---

## 🛠️ 설치 및 설정 방법

### 1. 가상환경 구축 및 패키지 설치
프로젝트 루트 폴더에서 터미널을 열고 필요한 패키지들을 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (`.env`)
프로젝트 루트 폴더에 `.env` 파일을 생성하고 아래와 같이 설정합니다. (**.env 파일은 보안을 위해 절대 Git에 올리지 마십시오.**)
```env
OPENAI_API_KEY="your-openai-api-key-here"
# 필요시 모델 폴더 경로를 커스텀할 수 있습니다 (기본값은 './modeling')
MODEL_DIR="./modeling"
```

### 3. RAG 로컬 벡터 DB 구축
의학 논문 PDF를 기반으로 로컬 벡터 데이터베이스(`vector_db.pkl`)를 최초 1회 생성해야 챗봇 상담 기능이 활성화됩니다. (키오스크 화면 내의 버튼으로도 생성 가능합니다.)
```bash
python rag_chatbot.py --build
```

---

## 🚀 키오스크 애플리케이션 실행

터미널에서 아래 명령어로 스트림릿 앱을 실행합니다.
```bash
streamlit run kiosk.py
```
실행이 완료되면 자동으로 브라우저 창(`http://localhost:8501`)이 열리며 검사 화면이 나타납니다.

---

## 🧠 머신러닝 & 딥러닝 파이프라인 설명

본 프로젝트는 세 가지 개별 인공지능 모델을 통합하여 종합적인 분석 결과를 도출합니다.

### 1. 후각 + 배변 분석 파이프라인 (`PPMI.py`)
* **개요**: PPMI 데이터셋을 가공하여 12가지 후각 문항(BSIT) 및 3가지 배변 불편 설문(SCOPA-AUT)으로 파킨슨 증상 유무를 분류합니다.
* **불균형 해소**: 클래스 불균형 문제를 해소하기 위해 `BorderlineSMOTE`를 파이프라인에 통합해 오버샘플링을 진행하며, XGBoost, Random Forest, SVM, Logistic Regression 성능을 교차 검증(5-Fold Stratified CV)으로 비교합니다.
* **최종 모델**: 학습 완료 후 `modeling/olf_con_model.pkl` 경로로 저장됩니다.

### 2. 손그림 이미지 분류 파이프라인 (`drawing_pipeline.py`)
* **개요**: 사용자가 그린 나선형 그림 이미지의 미세한 왜곡 및 흔들림 형태를 탐지합니다.
* **아키텍처**: `MobileNetV2` 백본 모델을 로드하여 가중치를 동결한 채 Classifier를 먼저 학습(Stage 1)한 후, 상위 30개 레이어를 해제하여 미세 조정(Stage 2, Fine-tuning)을 진행합니다.
* **설명 가능성 (XAI)**: Grad-CAM 알고리즘을 사용해 AI가 판단 시 비정상 패턴으로 주목한 이미지 내 픽셀 영역을 히트맵 형태로 시각화합니다.
* **최종 모델**: 학습 후 `modeling/drawing_cnn_model.h5` 경로로 저장됩니다.

### 3. 손그림 움직임(운동학) 피처 분석 파이프라인 (`kinematic_pipeline.py`)
* **개요**: 캔버스에서 실시간 수집된 그리기 궤적의 시계열 X, Y 좌표와 타임스탬프를 물리적 수치로 가공합니다.
* **피처 엔지니어링**: 총 이동거리, 그리기 소요 시간, 속도 평균/표준편차/변동계수(CV), 가속도 평균/표준편차/최대치, 저크(Jerk, 가속도의 변화율) 등을 추출합니다.
* **학습**: UCI 및 PaHaW 공개 데이터셋을 각 소스별로 정규화한 뒤, Random Forest Classifier로 학습시킵니다.
* **최종 모델**: 학습 완료 후 `modeling/drawing_kinematic_model.pkl` 경로로 저장됩니다.
