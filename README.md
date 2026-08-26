# ParkIN Soon: 파킨슨병 위험군 조기 선별 서비스

후각·배변·나선 그리기 검사를 이용해 **파킨슨병 위험군을 조기에 선별하고 병원 상담을 안내하는 키오스크 서비스**입니다. 의료 진단이나 질병 발생 확률을 제공하지 않습니다.

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
