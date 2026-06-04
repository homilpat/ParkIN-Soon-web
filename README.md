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
