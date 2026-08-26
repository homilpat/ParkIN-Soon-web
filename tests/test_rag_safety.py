"""Offline unit tests for RAG evidence and treatment-claim safety."""

import unittest

import numpy as np

from rag_answer_safety import clean_user_answer, personal_result_leak, unsupported_treatment_claim
from rag_evidence_gate import evidence_decision
from rag_hierarchical import hierarchical_search
from rag_senior_guardrails import senior_guard_response


class EvidenceGateTests(unittest.TestCase):
    def test_accepts_matching_high_score_evidence(self):
        chunk = {"text": "후각 저하는 운동 증상 전에 나타날 수 있다.",
                 "similarity": 0.61, "retrieval_score": 0.71}
        self.assertTrue(evidence_decision("후각 저하는 언제 나타나요?", chunk)["accepted"])

    def test_rejects_entity_mismatch(self):
        chunk = {"text": "변비와 장 운동에 관한 연구", "similarity": 0.81,
                 "retrieval_score": 0.82}
        self.assertFalse(evidence_decision("후각 저하는 언제 나타나요?", chunk)["accepted"])

    def test_rejects_numeric_mismatch(self):
        chunk = {"text": "추적 기간은 5년이었다.", "similarity": 0.81,
                 "retrieval_score": 0.82}
        self.assertFalse(evidence_decision("10년 추적 결과가 뭐예요?", chunk)["accepted"])

    def test_smell_wording_is_not_personal_pronoun(self):
        query = "요즘 냄새가 잘 안 나는데 파킨슨병인가요?"
        self.assertIsNone(__import__("re").search(
            r"(?:^|\s)(?:나는|제가|나의|저의)(?=\s|$)", query
        ))

    def test_hierarchical_search_returns_selected_metadata(self):
        db = {
            "metadata": [
                {"text": "후각 저하 연구", "source": "paper.pdf", "page": 1},
                {"text": "변비 연구", "source": "paper.pdf", "page": 2},
            ],
            "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        }
        result = hierarchical_search(db, np.asarray([1.0, 0.0]), "후각", top_k=1)
        self.assertEqual(result[0]["text"], "후각 저하 연구")


class TreatmentClaimTests(unittest.TestCase):
    def test_cleans_thinking_and_web_page_citation(self):
        answer = clean_user_answer(
            "<think>숨은 추론</think>안내입니다. (출처: 질병관리청, p. 웹페이지)"
        )
        self.assertNotIn("숨은 추론", answer)
        self.assertEqual(answer, "안내입니다. (출처: 질병관리청)")

    def test_flags_affirmative_claim(self):
        self.assertTrue(unsupported_treatment_claim("이 영양제는 파킨슨병을 예방합니다."))

    def test_allows_safe_boundary(self):
        answer = "영양제로 파킨슨병을 예방한다고 단정할 수 없습니다. 의료진과 상담하세요."
        self.assertFalse(unsupported_treatment_claim(answer))

    def test_detects_personal_result_leak(self):
        self.assertTrue(personal_result_leak("검사에서 후각 점수는 정상 범위에 있어요."))

    def test_allows_general_screening_explanation(self):
        self.assertFalse(personal_result_leak("이 검사는 병을 확정하는 검사가 아닙니다."))

    def test_guardrail_refuses_supplement_prevention(self):
        answer = senior_guard_response("무슨 영양제를 먹으면 파킨슨을 막을 수 있어요?")
        self.assertIsNotNone(answer)
        self.assertIn("예방한다고 단정할 수", answer)
        self.assertFalse(unsupported_treatment_claim(answer))

    def test_guardrail_refuses_acupuncture_cure(self):
        answer = senior_guard_response("침 맞으면 떨림이 완전히 낫나요?")
        self.assertIsNotNone(answer)
        self.assertFalse(unsupported_treatment_claim(answer))

    def test_guardrail_blocks_uncertain_double_dose(self):
        answer = senior_guard_response("아침 약을 먹었는지 모르겠는데 한 번 더 먹을까요?")
        self.assertIn("한 번 더 드시지 마세요", answer)
        self.assertIn("약사나 처방 의료진", answer)
        variant = senior_guard_response("약을 먹었는지 기억이 안 나요. 또 먹어도 돼요?")
        self.assertIn("한 번 더 드시지 마세요", variant)

    def test_guardrail_requires_retest_after_stopping(self):
        answer = senior_guard_response("검사하다 힘들어서 중간에 껐어요. 이 결과 써도 돼요?")
        self.assertIn("정보가 부족", answer)
        self.assertIn("쉬세요", answer)
        self.assertIn("다시 검사", answer)

    def test_guardrail_keeps_hearing_answer_short(self):
        answer = senior_guard_response("귀가 잘 안 들려요. 설명을 다시 짧게 해줘요.")
        self.assertLessEqual(len(answer), 100)
        self.assertIn("보호자나 담당자", answer)
        variant = senior_guard_response("잘 안 들리니 중요한 것만 짧게 다시 말해 주세요.")
        self.assertIn("보호자나 담당자", variant)

    def test_guardrail_handles_wrong_time_and_double_dose(self):
        wrong_time = senior_guard_response("저녁 약을 아침에 잘못 먹었어요. 다음 약은요?")
        self.assertIn("임의로", wrong_time)
        self.assertIn("약사나 처방 의료진", wrong_time)
        double = senior_guard_response("아버지가 약을 두 번 드신 것 같아요.")
        self.assertIn("119", double)

    def test_guardrail_handles_sudden_confusion(self):
        answer = senior_guard_response("갑자기 말이 이상하고 어디인지 모르겠어요.")
        self.assertIn("즉시 119", answer)

    def test_guardrail_handles_fall_risk(self):
        answer = senior_guard_response("혼자 있는데 자꾸 넘어질 것 같아요.")
        self.assertIn("혼자 무리해서 걷지", answer)
        self.assertIn("도움을 요청", answer)

    def test_guardrail_explains_screening_levels(self):
        low = senior_guard_response("결과가 낮음이면 병원에 안 가도 되나요?")
        self.assertIn("확정은 아닙니다", low)
        high = senior_guard_response("주의 단계면 파킨슨병 확정인가요?")
        self.assertIn("확정할 수 없습니다", high)
        self.assertIn("위험군 선별 보조", high)

    def test_guardrail_clarifies_vague_retest(self):
        answer = senior_guard_response("그걸 다시 해야 하는 건가요?")
        self.assertIn("어떤 검사", answer)

    def test_guardrail_retests_partial_smell_test(self):
        answer = senior_guard_response("냄새를 몇 개만 맡았는데 점수를 믿어도 돼요?")
        self.assertIn("정보가 부족", answer)
        self.assertIn("다시 검사", answer)

    def test_guardrail_handles_common_result_questions(self):
        caregiver = senior_guard_response("엄마 결과가 주의인데 가족이 뭘 해드리면 돼요?")
        self.assertIn("진단할 수는 없습니다", caregiver)
        self.assertIn("결과를 가지고", caregiver)
        probability = senior_guard_response("종합 점수 79%면 파킨슨 확률이 79%예요?")
        self.assertIn("확률이 아닙니다", probability)
        self.assertIn("위험군 선별 보조 점수", probability)

    def test_guardrail_handles_constipation_and_post_cold_smell(self):
        constipation = senior_guard_response("변비가 오래됐는데 이것만으로 파킨슨인가요?")
        self.assertIn("진단할 수는 없습니다", constipation)
        smell = senior_guard_response("감기 뒤로 냄새가 안 나요. 파킨슨 때문인가요?")
        self.assertIn("단정할 수 없습니다", smell)
        self.assertIn("이비인후과나 의료진", smell)

    def test_guardrail_explains_smell_test_without_diagnosis(self):
        answer = senior_guard_response("냄새 검사는 왜 파킨슨 선별에 들어가요?")
        self.assertIn("여러 선별 신호 중 하나", answer)
        self.assertIn("진단할 수는 없습니다", answer)

    def test_guardrail_keeps_exercise_guidance_short_and_safe(self):
        answer = senior_guard_response("집에서는 무슨 운동을 하면 도움이 돼요?")
        self.assertLessEqual(len(answer), 120)
        self.assertIn("혼자 하지 말고", answer)
        self.assertIn("의료진과 상담", answer)


if __name__ == "__main__":
    unittest.main()
