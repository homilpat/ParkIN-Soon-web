# 🧠 ParkIN Soon: ML/DL 기반 파킨슨 위험군 조기 선별 웹 서비스

비침습적 다중모달(Multi-modal) 데이터를 융합하여 파킨슨 위험군을 조기에 선별하고 보건소 추천 서비스를 제공하는 ML/DL 기반 웹 플랫폼입니다.

## 📌 핵심 아키텍처 및 문제 해결
**1. 데이터 희석을 방지하는 Late Fusion 아키텍처**
* **구조:** 설문 데이터(PPMI), 키네마틱 운동학 데이터, 나선형 그리기 데이터를 융합.
* **전략:** 데이터 간의 스케일과 특성이 크게 달라 병합 시 고유 특징이 손실되는 Early Fusion의 한계를 인지. 각 모달리티별로 독립적인 모델 학습 후 최종 예측값을 결합하는 **Late Fusion** 방식 채택.

**2. 머신비전 도메인 차이(Sim-to-Real Gap) 극복**
* **문제:** 나선형 그리기 딥러닝 파트에서, 실사용 환경의 조명 및 펜 색상 차이로 인해 인식률 저하.
* **해결 방안:** **OpenCV**를 활용하여 배경 이미지를 제거하고, 다양한 파란선 입력값을 검은선으로 변환시키는 도메인 보정 로직 구축.

**3. 사용자 편의를 위한 생성형 AI 도입**
* **특징:** OpenAI API를 활용한 RAG 기반 LLM 챗봇을 구현하여 질의응답 및 서비스 편의성 극대화.

## 🛠 Tech Stack
* **Language/Web:** Python, Streamlit
* **AI/ML/DL:** Scikit-learn, PyTorch, OpenCV, RAG LLM
* **Data:** PPMI(설문), Kinematic Data, Image Data
* 
# ParkIN Soon 키오스크 실행 안내

## OpenAI API 키 설정

AI 박인순 상담 기능을 사용하려면 프로젝트 폴더에 `.env` 파일을 만들고 아래처럼 입력하세요.

```env
OPENAI_API_KEY=여기에_본인_OpenAI_API_키를_입력
```

예시:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

주의:
- 변수 이름은 `OPENAI_API_KEY`로 입력해야 합니다.
- `.env` 파일은 깃허브나 공유 폴더에 올리지 마세요.
- API 키가 없어도 후각, 배변, 손그림 검사와 로컬 결과 계산은 동작합니다.
- API 키가 있으면 최종 결과 화면의 AI 박인순 Q&A 상담이 활성화됩니다.

## 실행

```bash
streamlit run kiosk.py
```
