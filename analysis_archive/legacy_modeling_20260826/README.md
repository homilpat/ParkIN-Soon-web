# 2026-08-26 분석 보관함

현재 ParkIN Soon 서비스와 RAG 실행 경로에서 import되지 않는 모델 실험 자료를 보관한다.
삭제하지 않고 이동했으므로 필요하면 원래 경로로 복원할 수 있다.

## 원위치에 유지한 사용 파일

- 서비스: `kiosk.py`, `ParkinsoonAI.py`, `drawing_pipeline.py`, `kinematic_pipeline.py`
- 판정: `late_fusion.py`, `fusion_config.json`, `modeling/fusion_config.json`
- 배포 모델: `modeling/olf_con_model.pkl`, `modeling/drawing_domain_fold_*.keras`
- RAG: `rag_chatbot.py`, `rag_*.py`, `evaluate_rag_*.py`, `evaluate_senior_rag.py`
- RAG 인덱스: `modeling/vector_db.pkl`, `modeling/vector_child_db.pkl`, `modeling/vector_evidence_db.pkl`

## 이동한 자료

- 과거 그림 모델 학습·비교 스크립트 7개
- 변환 전 노트북 2개
- 후보 모델과 메타데이터 `modeling/candidates/`
- PPMI 예측 배열 2개와 비교 그림 1개
- 발표·제출 참고 문서 3개

## 변경 파일 판단

Git에서 수정 상태인 파일은 모두 현재 서비스, 선별 판정, 배포 모델 또는 RAG가 직접 사용한다.
따라서 이동하지 않았다. 이 보관함에는 미사용 미추적 실험 자료만 이동했다.

`spiral_preprocessing.py`는 `kiosk.py`의 실행 의존성임을 재감사에서 확인해 프로젝트
루트로 복원했다.
