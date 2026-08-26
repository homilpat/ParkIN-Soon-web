"""Route questions between academic evidence and official lifestyle guidance."""

import json
from typing import Any, Dict, List

LIFESTYLE_TERMS = (
    "운동", "식사", "음식", "생활", "관리", "예방", "물", "수분", "변비",
    "낙상", "넘어", "걷기", "스트레칭", "근력", "영양", "집에서", "주의사항",
    "어떻게 해야", "뭘 해야", "도움", "습관",
)
ACADEMIC_TERMS = (
    "논문", "연구", "근거", "수치", "AUC", "민감도", "특이도", "메타분석",
    "몇 배", "몇 년", "정확도", "발병", "전구기", "Grad-CAM", "모델",
)

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["academic", "lifestyle", "mixed"]},
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
    "additionalProperties": False,
}


def fallback_route(query: str) -> str:
    lifestyle = any(term.lower() in query.lower() for term in LIFESTYLE_TERMS)
    academic = any(term.lower() in query.lower() for term in ACADEMIC_TERMS)
    if lifestyle and academic:
        return "mixed"
    return "lifestyle" if lifestyle else "academic"


def route_question(client: Any, model: str, query: str) -> str:
    """Use the chat model as an intent router and fall back deterministically."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "질문을 세 범주로만 분류하세요. academic은 논문 사실·수치·기전·성능 질문, "
                    "lifestyle은 운동·식사·수분·낙상·일상 행동 질문, mixed는 두 근거가 모두 필요한 질문입니다."
                )},
                {"role": "user", "content": query + ("\n/no_think" if model == "parkinsoon-eval" else "")},
            ],
            temperature=0,
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "question_route", "strict": True, "schema": ROUTE_SCHEMA},
            },
        )
        route = json.loads(response.choices[0].message.content)["route"]
        return route if route in {"academic", "lifestyle", "mixed"} else fallback_route(query)
    except Exception:
        return fallback_route(query)


def lifestyle_evidence(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Select curated KDCA guidance without allowing the model to invent advice."""
    entries = [
        {
            "keywords": "운동 걷기 스트레칭 근력 균형 코어 수영 체조 요가 자전거",
            "section": "운동 관리",
            "text": "걷기, 스트레칭, 근력운동을 기본으로 하되 건강 상태에 맞게 무리하지 않고 꾸준히 한다. 넘어질 위험이 있으면 보조기구를 사용하거나 보호자와 함께한다.",
        },
        {
            "keywords": "식사 영양 음식 단백질 규칙 천천히",
            "section": "영양 관리",
            "text": "규칙적으로 식사하고 여러 영양소를 골고루 섭취한다. 식사는 충분한 시간을 두고 천천히 한다. 일반 선별 사용자에게 단백질 제한을 권하지 않는다.",
        },
        {
            "keywords": "변비 물 수분 섬유질 현미 통밀 채소 과일",
            "section": "변비 예방",
            "text": "변비 예방을 위해 수분을 충분히 섭취하고 현미, 통밀빵, 채소, 과일 등 섬유질이 풍부한 식품을 자주 먹는다.",
        },
        {
            "keywords": "낙상 넘어짐 넘어 휘청 균형 집안 환경 안전 보조기구",
            "section": "낙상 예방",
            "text": "보행이나 균형이 불안하면 넘어짐을 예방하도록 주변 환경을 정리하고, 운동할 때 보조기구나 보호자의 도움을 고려한다.",
        },
        {
            "keywords": "주의 신호 재검사 상담 전문의 검사 결과",
            "section": "선별 결과 다음 행동",
            "text": "선별 결과는 진단이 아니다. 불편한 증상이 지속되거나 여러 주의 신호가 함께 보이면 결과를 의료진에게 보여주고 상담한다.",
        },
    ]
    normalized_query = query.lower().replace(" ", "")

    def relevance(item):
        keywords = item["keywords"].lower().split()
        return sum(keyword.replace(" ", "") in normalized_query for keyword in keywords)

    scored = [(relevance(item), item) for item in entries]
    ranked = [item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True)
              if score > 0][:top_k]
    if not ranked:
        ranked = [entries[-1]]
    for item in ranked:
        item.update({
            "source": "질병관리청 국가건강정보포털 - 파킨슨병 관리",
            "url": "https://health.kdca.go.kr/healthinfo/biz/health/ntcnInfo/healthSourc/thtimtCntnts/thtimtCntntsView.do?thtimt_cntnts_sn=125",
            "evidence_type": "official_lifestyle",
        })
    return ranked
