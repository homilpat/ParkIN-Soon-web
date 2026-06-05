import os
from pathlib import Path

import streamlit as st

from rag_chatbot import ParkinsonRAG


PLACEHOLDER_KEYS = {
    "",
    "sk-your-openai-api-key-here",
    "your-openai-api-key",
    "YOUR_OPENAI_API_KEY",
}


APP_DIR = Path(__file__).resolve().parent


def _character_img_html():
    path = APP_DIR / "character.png"
    if not path.exists():
        return '<div style="font-size:30px;">박</div>'
    import base64

    img_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return (
        f'<img src="data:image/png;base64,{img_b64}" '
        'style="width:64px;height:64px;object-fit:contain;'
        'border-radius:50%;background:#f8fafc;border:1px solid #dbeafe;'
        'box-shadow:0 2px 8px rgba(15,23,42,0.12);" />'
    )


def _get_api_key():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    try:
        api_key = api_key or st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        pass
    if api_key in PLACEHOLDER_KEYS:
        return ""
    return api_key


def _ensure_welcome_message():
    if "report_messages" not in st.session_state:
        st.session_state.report_messages = [
            {
                "role": "assistant",
                "content": (
                    "최종 레포트를 확인하셨나요?\n\n"
                    "궁금한 점을 적어주시면 검사 결과를 바탕으로 쉽게 설명드릴게요."
                ),
            }
        ]
    if "last_explained_step" not in st.session_state:
        st.session_state.last_explained_step = 0


def _auto_prompt_for_step(current_step, test_results):
    if current_step == 4:
        return (
            "사용자가 후각, 배변, 손그림 검사를 모두 완료했습니다. "
            "점수와 모델 확률을 종합해서 노인도 쉽게 이해할 수 있는 최종 설명을 작성해주세요. "
            "진단처럼 단정하지 말고, 보호자나 보건소 담당자에게 보여줄 수 있는 짧은 요약도 포함해주세요."
        )
    return ""


@st.cache_resource
def _get_rag_chatbot(api_key: str):
    return ParkinsonRAG(api_key=api_key)


def render_parkinson_chatbot(test_results: dict):
    current_step = test_results.get("step", 0)
    if current_step < 4:
        return
    if test_results.get("result_page") != 3:
        return

    character_html = _character_img_html()
    st.markdown(
        f"""
<div style="background:white;border:1.5px solid #dbeafe;border-radius:18px;
            padding:20px 24px;margin-top:28px;margin-bottom:14px;
            box-shadow:0 2px 12px rgba(15,23,42,0.06);">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <div style="flex:0 0 auto;">{character_html}</div>
        <div>
            <div style="font-size:24px;font-weight:800;color:#1A1A2E;">AI 박인순 상담</div>
            <div style="font-size:18px;color:#64748b;">검사 흐름에 맞춰 결과를 설명하고 질문에 답합니다.</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    api_key = _get_api_key()
    if not api_key:
        st.info(".env 파일의 OPENAI_API_KEY 값을 입력하면 AI 상담이 활성화됩니다.")
        return

    rag = _get_rag_chatbot(api_key)
    if not rag.db:
        st.warning("논문 벡터 DB가 없습니다. 아래 버튼으로 PDF 기반 DB를 먼저 생성해주세요.")
        if st.button("논문 벡터 DB 생성하기", use_container_width=True):
            with st.spinner("PDF를 읽고 로컬 벡터 DB를 생성하는 중입니다..."):
                count, msg = rag.build_db_from_pdfs()
                if count > 0:
                    st.success(f"DB 생성 완료: {count}개 문단")
                    rag.load_db()
                    st.rerun()
                else:
                    st.error(f"DB 생성 실패: {msg}")
        return

    _ensure_welcome_message()

    chat_container = st.container(height=360, border=True)
    with chat_container:
        for msg in st.session_state.report_messages:
            role_name = "사용자" if msg["role"] == "user" else "AI 박인순"
            avatar_val = "character.png" if msg["role"] == "assistant" else "👤"
            with st.chat_message(role_name, avatar=avatar_val):
                st.markdown(msg["content"])

    with st.form("parkinsoon_chat_form", clear_on_submit=True):
        user_query = st.text_input("질문", placeholder="결과에 대해 궁금한 점을 입력해주세요.", label_visibility="collapsed")
        submitted = st.form_submit_button("질문하기", use_container_width=True)

    if submitted and user_query.strip():
        st.session_state.report_messages.append({"role": "user", "content": user_query.strip()})
        with st.spinner("검사 결과를 살펴보고 답변을 작성하는 중입니다..."):
            answer = rag.generate_rag_answer(user_query.strip(), test_results, top_k=3)
        st.session_state.report_messages.append({"role": "assistant", "content": answer})
        st.rerun()
