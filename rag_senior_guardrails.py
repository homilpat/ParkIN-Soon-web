"""Deterministic safety responses for high-risk or context-free senior questions."""

import re
from typing import Optional


def senior_guard_response(query: str) -> Optional[str]:
    normalized = re.sub(r"\s+", "", query.lower())

    sudden_confusion = "갑자기" in normalized and any(
        term in normalized for term in ("말이이상", "말이어눌", "어디인지모르", "정신이없", "의식")
    )
    sudden_weakness = "갑자기" in normalized and any(
        term in normalized for term in ("한쪽", "팔에힘이없", "다리에힘이없", "마비")
    )
    if sudden_confusion or sudden_weakness:
        return (
            "지금은 파킨슨 선별 검사를 할 때가 아닙니다. 즉시 119에 연락하세요. "
            "혼자 이동하지 말고 가까운 사람에게 도움을 요청하세요."
        )

    wrong_medicine_time = any(term in normalized for term in ("약을잘못", "저녁약을아침", "아침약을저녁"))
    possible_double_dose = any(term in normalized for term in ("두번먹", "두번드", "중복복용"))
    if wrong_medicine_time or possible_double_dose:
        return (
            "다음 약을 임의로 드시거나 건너뛰지 마세요. 지금 약 봉투를 준비해 약사나 처방 의료진에게 바로 확인하세요. "
            "심하게 졸리거나 숨쉬기 어렵고 의식이 흐려지면 즉시 119에 연락하세요."
        )

    took_medicine = any(term in normalized for term in ("먹었는지", "복용했는지", "먹은지"))
    uncertain_memory = any(term in normalized for term in ("모르", "기억이안", "헷갈"))
    medication_uncertain = took_medicine and uncertain_memory and any(
        term in normalized for term in ("한번더", "또먹", "추가로")
    )
    if medication_uncertain:
        return (
            "지금은 약을 한 번 더 드시지 마세요. 약 봉투나 복약 기록을 확인하고, "
            "확실하지 않으면 약사나 처방 의료진에게 바로 문의하세요."
        )

    stopped_test = any(term in normalized for term in ("검사하다", "검사중", "중간에")) and any(
        term in normalized for term in ("껐", "중단", "그만", "못했", "힘들")
    )
    if stopped_test:
        return (
            "중간에 멈춘 검사는 정보가 부족해 결과를 그대로 쓰기 어렵습니다. "
            "먼저 편하게 쉬세요. 괜찮아진 뒤 처음부터 다시 검사하고, 계속 힘들면 보호자나 담당자에게 도움을 요청하세요."
        )

    partial_smell = any(term in normalized for term in ("냄새", "후각")) and any(
        term in normalized for term in ("몇개만", "일부만", "다못", "중간에")
    )
    if partial_smell:
        return (
            "후각 검사를 일부만 했다면 정보가 부족해 그 점수를 믿기 어렵습니다. "
            "쉬었다가 안내에 따라 처음부터 다시 검사하세요."
        )

    hearing_short = any(term in normalized for term in ("귀가잘안들", "안들리", "청력")) and any(
        term in normalized for term in ("짧게", "다시설명", "말해")
    )
    if hearing_short:
        return (
            "검사 결과는 진단이 아닙니다. 화면을 보호자나 담당자에게 보여주고, "
            "가장 중요한 다음 행동을 큰 소리로 천천히 설명해 달라고 요청하세요."
        )

    one_action = any(term in normalized for term in ("한가지만", "제일중요", "뭐부터"))
    if one_action:
        return (
            "지금 할 일은 결과 화면을 보호자나 의료진에게 보여주고 상담하는 것입니다. "
            "이 결과는 진단이 아니라 위험군 선별을 돕는 참고 자료입니다."
        )

    sudden_neuro = (
        "갑자기" in normalized
        and any(term in normalized for term in ("한쪽", "팔에힘이없", "다리에힘이없", "마비"))
        and any(term in normalized for term in ("말이어눌", "의식", "얼굴", "힘이없", "마비"))
    )
    chest_breathing = (
        any(term in normalized for term in ("가슴이아", "가슴통증"))
        and any(term in normalized for term in ("숨쉬기", "숨이차", "호흡"))
    )
    head_injury = (
        any(term in normalized for term in ("머리를부딪", "머리다쳤"))
        and any(term in normalized for term in ("졸려", "토해", "의식", "심한두통"))
    )
    if sudden_neuro or chest_breathing or head_injury:
        return (
            "지금은 검사를 기다릴 상황이 아닙니다. 즉시 119에 연락하거나 응급실로 가세요. "
            "혼자 이동하지 말고 가까운 사람에게 도움을 요청하세요."
        )

    fall_risk = any(term in normalized for term in ("넘어질것", "넘어질거", "자꾸넘어", "휘청"))
    if fall_risk:
        return (
            "지금은 혼자 무리해서 걷지 마세요. 앉아서 안전을 확보하고 가까운 사람에게 도움을 요청하세요. "
            "다쳤거나 갑자기 힘이 빠졌다면 119 또는 의료기관에 바로 연락하세요."
        )

    low_result = any(term in normalized for term in ("낮음", "낮게")) and any(
        term in normalized for term in ("병원", "안가", "괜찮", "안심")
    )
    if low_result:
        return (
            "낮음은 이번 선별 검사에서 주의 신호가 적었다는 뜻이지, 병이 없다는 확정은 아닙니다. "
            "불편한 증상이 계속되면 결과와 관계없이 신경과 의료진과 상담하세요."
        )

    high_result = any(term in normalized for term in ("주의", "높음", "높게")) and any(
        term in normalized for term in ("파킨슨", "확정", "병인가", "걸린")
    )
    if high_result:
        return (
            "아니요. 주의 단계나 높은 점수만으로 파킨슨병을 확정할 수 없습니다. "
            "위험군 선별 보조 결과를 가지고 신경과 의료진과 상담하세요."
        )

    caregiver_result = any(term in normalized for term in ("엄마", "어머니", "아빠", "아버지")) and any(
        term in normalized for term in ("결과", "주의", "점수")
    ) and any(term in normalized for term in ("뭘해", "어떻게", "도와"))
    if caregiver_result:
        return (
            "이 결과만으로 파킨슨병을 진단할 수는 없습니다. 가족이 할 일은 결과 화면을 함께 확인하고, "
            "불편한 증상이 계속되면 결과를 가지고 신경과 의료진과 상담하도록 돕는 것입니다."
        )

    score_probability = any(term in normalized for term in ("종합점수", "점수")) and any(
        term in normalized for term in ("확률", "%", "퍼센트")
    )
    if score_probability:
        return (
            "아니요. 이 수치는 파킨슨병에 걸렸을 확률이 아닙니다. "
            "세 검사 신호를 합친 위험군 선별 보조 점수이며, 진단은 의료진의 별도 평가가 필요합니다."
        )

    constipation_pd = "변비" in normalized and "파킨슨" in normalized
    if constipation_pd:
        return (
            "변비만으로 파킨슨병이라고 진단할 수는 없습니다. 원인은 여러 가지일 수 있습니다. "
            "변비가 오래 지속되거나 다른 불편이 함께 있으면 의료진과 상담하세요."
        )

    smell_after_cold = any(term in normalized for term in ("감기", "코로나")) and any(
        term in normalized for term in ("냄새", "후각")
    )
    if smell_after_cold:
        return (
            "감기 뒤 후각 저하만으로 파킨슨병 때문이라고 단정할 수 없습니다. "
            "냄새를 못 맡는 증상이 계속되면 이비인후과나 의료진과 상담하세요."
        )

    why_smell_test = any(term in normalized for term in ("냄새검사", "후각검사")) and any(
        term in normalized for term in ("왜", "이유", "들어가")
    )
    if why_smell_test:
        return (
            "후각 검사는 냄새를 구별하는 변화를 여러 선별 신호 중 하나로 확인하기 위해 포함됩니다. "
            "후각 결과 하나만으로 파킨슨병을 진단할 수는 없습니다."
        )

    exercise_question = "운동" in normalized and any(
        term in normalized for term in ("뭐", "무슨", "어떻게", "도움")
    )
    if exercise_question:
        return (
            "집에서는 안전한 범위에서 걷기나 가벼운 스트레칭부터 시작하세요. "
            "넘어질 위험이 있으면 혼자 하지 말고, 통증이나 어지럼이 생기면 멈춘 뒤 의료진과 상담하세요."
        )

    context_words = ("아까", "그거", "그걸", "그게", "그점수", "그결과")
    vague_risk = any(term in normalized for term in context_words) and any(
        term in normalized for term in ("무슨뜻", "얼마나위험", "어떻게", "뭐야", "다시", "해야")
    )
    if vague_risk:
        return (
            "어떤 검사나 결과를 말씀하시는지 먼저 확인해야 합니다. "
            "화면에 나온 검사 이름을 알려주세요."
        )

    drawing_problem = any(term in normalized for term in ("그림", "선", "나선")) and any(
        term in normalized for term in ("삐뚤", "못그", "실패", "끊겼", "다시해야")
    )
    if drawing_problem:
        return (
            "그림이 제대로 그려지지 않았다면 다시 검사하는 것이 좋습니다. "
            "편하게 앉아 안내선을 따라 한 번 더 그려주세요. 반복해도 어렵다면 보호자나 담당자에게 도움을 요청하세요."
        )

    supplement_prevention = any(
        term in normalized for term in ("영양제", "건강기능식품")
    ) and any(term in normalized for term in ("막을", "예방"))
    if supplement_prevention:
        return (
            "특정 영양제가 파킨슨병을 예방한다고 단정할 수는 없습니다. "
            "영양제를 시작하기 전에는 복용 중인 약과 함께 신경과 의료진에게 확인하세요."
        )

    unsupported_treatment = any(
        term in normalized for term in ("침맞", "한약", "민간요법")
    ) and any(term in normalized for term in ("완전히", "완치", "낫", "막을", "예방"))
    if unsupported_treatment:
        return (
            "침이나 한약, 민간요법으로 떨림이나 파킨슨병이 완전히 낫는다고 단정할 수는 없습니다. "
            "보조요법을 시작하기 전에는 신경과 의료진과 상의하세요."
        )
    return None
