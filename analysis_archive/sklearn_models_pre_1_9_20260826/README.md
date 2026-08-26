# scikit-learn 1.9.0 이전 배포 모델 백업

2026-08-26 공통 런타임 직렬화 이전의 원본 모델과 후각 모델 메타데이터다.

- `olf_con_model.pkl`: scikit-learn 1.5.0 생성본, SHA-256 `4F45CD475C16F6940211E9635D06EE35A206FF5D22F19C6C29B9E9D55D567C06`
- `drawing_kinematic_model.pkl`: scikit-learn 1.7.2 생성본, SHA-256 `87FE5348634DCC804CA00339756CC818A24F4F4681C88FAB69B1539F28535BB2`
- `olf_con_model_metadata.json`: SHA-256 `81D60F46C6AFDE0B48BC831A545B9804CE915AE25F80DCBB556690870CFD3FC8`

운영 모델 복원이 꼭 필요할 때만 프로젝트 `modeling/`에 복사한다. 복원하면 원래 버전에 맞는 별도 환경에서 먼저 검증해야 한다.
