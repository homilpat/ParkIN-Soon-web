[시연영상](https://youtu.be/iWVYuI1OL_8)
# ParkIN Soon: 파킨슨병 위험군 조기 선별 서비스

후각·배변·나선 그리기 검사를 이용해 **파킨슨병 위험군을 조기에 선별하고 병원 상담을 안내하는 키오스크 서비스**입니다. 의료 진단이나 질병 발생 확률을 제공하지 않습니다.

## 프로젝트 개요

| 구분 | 내용 |
|---|---|
| 진행 기간 | 2026.04 ~ 2026.05 팀 프로젝트, 2026.08 개인 재작업 |
| 진행 인원 | 4인 (ParkINSoon팀), 2026.08 재작업은 개인 단독 |
| 담당 역할 | Late Fusion 설계·구현, RAG 상담 챗봇 구현 / 재작업: 피험자 단위 검증 재설계, HandPD 데이터 확보, HSV 전처리, 재모델링, 로컬 RAG 전환 |

팀 프로젝트 종료 후, 이미지 단위 5-fold로 검증하던 나선 이미지 모델은 같은 피험자의 그림이 학습과 검증에 함께 들어갈 수 있었습니다. 2026.08 재작업에서 피험자 단위 StratifiedGroupKFold로 검증을 교체하고, 공개 데이터셋 HandPD를 더해 개발 표본을 65명·376장으로 늘린 뒤 모델과 RAG를 다시 평가했습니다. 날짜별 기록은 `DEVELOPMENT_ROADMAP.md`에 있습니다.

## 핵심 기능

- 후각 12문항과 배변 3문항 기반 위험 신호 분석
- 나선 이미지 5-fold 앙상블과 그리기 운동학 분석
- 입력 품질과 불확실성을 반영한 동적 Late Fusion
- 유효한 하위 검사 임계값 다수결에 따른 단계 판정
- 유효 검사 2개 미만 시 판정 보류와 재검사 안내
- 논문·질병관리청 근거 기반 RAG 설명과 병원 위치 안내
- 약물 오복용·응급 증상·낙상 위험을 우선 처리하는 고령 사용자 안전 응답

## 설계 포인트

- 서로 성격이 다른 설문·이미지·운동학 신호를 독립 모델로 분석한 뒤 품질과 불확실성을 반영해 결합합니다.
- OpenCV 전처리와 여러 도메인을 이용한 이미지 앙상블로 실제 키오스크 입력 차이에 대응합니다.
- 로컬 LLM과 논문·공공기관 근거를 연결한 RAG로 출처가 있는 설명을 제공하고, 의료 안전 질문은 결정 규칙으로 먼저 처리합니다.

주요 기술은 Python, Streamlit, scikit-learn, TensorFlow/Keras, OpenCV, LM Studio 기반 로컬 RAG입니다.

## 검증 기준

| 하위 검사 | 검증 | AUC | 단계 임계값 |
|---|---|---:|---:|
| 후각+배변 | 내부 nested CV | 0.853759 | 0.448809 |
| 나선 이미지 | 개발 피험자 단위 OOF | 0.825203 | 0.389826 |
| 나선 운동학 | 5-fold CV | 0.761500 | 0.541800 |

수치는 내부 또는 개발 데이터 검증 결과이며 외부 임상 검증 결과가 아닙니다. 종합 점수는 `파킨슨병 위험군 선별 보조 점수`입니다.

나선 이미지 모델은 도메인 증강 MobileNetV2 5-fold 앙상블이며 개발 OOF 기준 민감도 0.829, 특이도 0.792입니다. 출처별 AUC는 HandPD 0.805, 기존 데이터 0.850이고 seed를 바꾼 3회 재실행에서도 0.825·0.814·0.833으로 유지됐습니다.

### RAG 검색 평가

| 평가셋 | source-page Recall@3 |
|---|---|
| 개발셋 30문항 (튜닝에 사용) | 기본 검색 0.600 → 용어 확장·수치 재순위화 0.833 → Cross-Encoder 결합 0.867 → 표·수치·캡션 전용 색인 0.933 |
| 블라인드 잠금셋 (튜닝에 미사용) | 1차 15문항 0.400 · 2차 15문항 0.533 |

개발셋 수치는 실패 분석과 보강에 사용한 회귀 성능이므로 독립 테스트 성능이 아닙니다. 챗봇은 사용자의 주의 단계·검사 점수를 프롬프트에 포함하므로 개인 건강정보가 외부 API로 나가지 않도록 생성·임베딩·채점을 로컬 LLM(Qwen3-8B, BGE-M3, Gemma)으로 수행합니다.

## 실행

고정 환경은 WSL2 Ubuntu 24.04, Python 3.12입니다.

```bash
wsl.exe -d Ubuntu-24.04 -- /home/user/venvs/parksoon-tf/bin/python -m pip install -r requirements.txt
wsl.exe -d Ubuntu-24.04 -- bash run_kiosk_wsl.sh --check
wsl.exe -d Ubuntu-24.04 -- bash run_kiosk_wsl.sh
```

브라우저에서 `http://127.0.0.1:8501`을 엽니다. 로컬 LM Studio를 사용할 때는 `parkinsoon-eval`, `parkinsoon-judge`, `text-embedding-bge-m3` 모델을 준비합니다.

## 프로젝트 구조

```text
kiosk.py                 Streamlit 검사·결과·병원 안내
ParkinsoonAI.py          결과 화면 RAG 챗봇 UI
late_fusion.py           동적 통합 점수와 다수결 단계 판정
spiral_preprocessing.py  나선 이미지 정렬·전처리
rag_*.py                 운영 RAG 검색·근거·안전 모듈
modeling/                배포 모델·설정·벡터 DB·모델 메타데이터
tests/                   단계 판정과 RAG 안전 회귀 테스트
analysis_archive/        학습·실험·이전 평가·롤백 자료
```

## 테스트

```bash
wsl.exe -d Ubuntu-24.04 -- /home/user/venvs/parksoon-tf/bin/python -m unittest discover -s tests
```

자세한 변경·검증 기록과 한계는 `DEVELOPMENT_ROADMAP.md`에서 확인할 수 있습니다.

## 운영 파이프라인 파일 안내

아래는 `analysis_archive/`를 제외한 현재 운영·검증 파이프라인입니다.

### 애플리케이션과 판정

| 파일 | 설명 |
|---|---|
| `kiosk.py` | Streamlit 진입점입니다. 사용자 정보, 후각·배변·나선 검사, 결과 화면, 병원 안내를 연결합니다. |
| `ParkinsoonAI.py` | 결과 화면의 박인순 RAG 챗봇 UI와 대화 상태를 관리합니다. |
| `late_fusion.py` | 검사 품질·불확실도를 반영한 동적 Late Fusion 점수와 유효 검사 임계값 다수결 단계를 계산합니다. |
| `spiral_preprocessing.py` | 키오스크에서 그린 나선을 정렬하고 이미지 모델 입력 형태로 전처리합니다. |
| `fusion_config.json` | 애플리케이션이 읽는 통합 가중치·단계 임계값 설정입니다. |
| `character.png` | 키오스크와 챗봇에서 사용하는 박인순 캐릭터 이미지입니다. |
| `hospital_data.xlsx` | 결과에 따라 주변 의료기관을 안내할 때 사용하는 병원 정보입니다. |

### 운영 RAG

| 파일 | 설명 |
|---|---|
| `rag_chatbot.py` | 질문 분류, 근거 검색, 개인 결과 해석 경계, LLM 답변 생성과 출처 표시를 조정합니다. |
| `rag_retrieval.py` | 의미 검색과 키워드 검색을 결합한 기본 검색 로직입니다. |
| `rag_hierarchical.py` | 부모·자식 청크 문맥을 이용해 검색 결과를 구성합니다. |
| `rag_evidence_index.py` | 수치 질문 등 질문 유형에 맞는 근거 후보를 선택합니다. |
| `rag_evidence_gate.py` | 관련성과 근거 충족도를 확인해 답변에 사용할 청크를 제한합니다. |
| `rag_question_router.py` | 질문을 논문 근거, 생활관리 근거 또는 혼합 경로로 분류합니다. |
| `rag_cross_encoder.py` | 선택적으로 Cross-Encoder 재정렬을 적용해 근거 순위를 보정합니다. |
| `rag_structured_evidence.py` | 논문 표와 수치 문장을 구조화된 검색 근거로 변환합니다. |
| `rag_answer_safety.py` | 근거 없는 치료 주장, 개인 결과 누출, 답변 생성 흔적과 출처 표기를 검사·정리합니다. |
| `rag_senior_guardrails.py` | 약물 오복용, 응급 증상, 낙상, 불완전 검사 등 고령 사용자 안전 질문을 LLM보다 먼저 처리합니다. |

### 배포 모델과 검색 데이터

| 파일 | 설명 |
|---|---|
| `modeling/olf_con_model.pkl` | 후각 12문항과 배변 3문항을 사용하는 후각+배변 배포 모델입니다. |
| `modeling/drawing_domain_fold_1.keras` ~ `drawing_domain_fold_5.keras` | 피험자 단위 5-fold 나선 이미지 앙상블 모델입니다. |
| `modeling/drawing_kinematic_model.pkl` | 나선 그리기 속도·가속도·떨림 특성을 사용하는 운동학 배포 모델입니다. |
| `modeling/fusion_config.json` | 배포 모델 파일명, 검증 성능, 통합 설정과 단계 임계값의 운영 원본입니다. |
| `modeling/vector_db.pkl` | 논문 본문 중심의 기본 RAG 벡터 DB입니다. |
| `modeling/vector_child_db.pkl` | 세부 문장 검색을 위한 자식 청크 벡터 DB입니다. |
| `modeling/vector_evidence_db.pkl` | 표·수치 근거 검색을 위한 증거 벡터 DB입니다. |
| `modeling/*_metadata.json` | 모델별 검증 방법, 성능, 직렬화 환경과 파일 구성을 기록합니다. |
| `modeling/sklearn_model_migration_1_9.json` | scikit-learn 1.9.0 공통 직렬화 이관과 예측 동등성 결과입니다. |
| `modeling/customer_records.db` | 로컬 재검사 비교용 실행 DB입니다. 개인정보 보호를 위해 Git에서 제외됩니다. |

### 실행환경과 검증

| 파일 | 설명 |
|---|---|
| `run_kiosk_wsl.sh` | CUDA 라이브러리 경로를 설정하고 키오스크를 실행합니다. |
| `requirements.txt` | 검증된 WSL 운영환경의 직접 의존성 고정 버전입니다. |
| `requirements-kiosk-wsl.txt` | 동일한 키오스크 WSL 복구용 의존성 목록입니다. |
| `requirements-reranker-wsl.txt` | 선택적 RAG 재정렬 모델의 추가 의존성입니다. |
| `tests/test_late_fusion.py` | 임계값 로딩, 다수결 단계와 판정 보류 규칙을 검증합니다. |
| `tests/test_rag_safety.py` | 근거 채택, 결과 누출, 치료 주장과 고령 사용자 안전 답변을 검증합니다. |
| `RAG_EVALUATION_PROTOCOL.md` | RAG 개발·잠금 평가의 분리 원칙과 평가 절차입니다. |
| `DEVELOPMENT_ROADMAP.md` | 실제 변경, 실행 검증 결과, 제한점과 다음 우선순위를 누적 기록합니다. |
