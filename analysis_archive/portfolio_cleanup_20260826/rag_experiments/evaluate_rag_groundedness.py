"""Generate RAG answers and grade every factual claim against gold evidence."""

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_chatbot import ParkinsonRAG

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "rag_grounding_cases.json"
EXTRA_CASES_PATH = ROOT / "rag_grounding_cases_extra.json"
BLIND_CASES_PATH = ROOT / "rag_blind_grounding_cases.json"
BLIND_V2_CASES_PATH = ROOT / "rag_blind_grounding_cases_v2.json"
OUTPUT = ROOT / "modeling/rag_groundedness_evaluation.json"
BLIND_OUTPUT = ROOT / "modeling/rag_blind_groundedness_evaluation.json"
BLIND_V2_OUTPUT = ROOT / "modeling/rag_blind_v2_groundedness_evaluation.json"
GENERATOR_MODEL = "parkinsoon-eval"
JUDGE_MODEL = "parkinsoon-judge"

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "label": {"type": "string", "enum": ["supported", "unsupported", "not_factual"]},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "label", "reason"],
                "additionalProperties": False,
            },
        },
        "expected_claims_covered": {"type": "array", "items": {"type": "boolean"}},
        "forbidden_claim_present": {"type": "boolean"},
        "citation_source_correct": {"type": "boolean"},
        "citation_page_correct": {"type": "boolean"},
        "diagnostic_overclaim": {"type": "boolean"},
    },
    "required": ["claims", "expected_claims_covered", "forbidden_claim_present",
                 "citation_source_correct", "citation_page_correct", "diagnostic_overclaim"],
    "additionalProperties": False,
}


def neutral_results():
    return {
        "olf_results": [0] * 12, "scopa_results": [0, 0, 0],
        "p_olf": .30, "p_img": .30, "p_kin": .30, "fusion_score": .30,
        "fusion_weights": {"olf": .38, "img": .35, "kin": .27},
        "modality_quality": {"olf": 1.0, "img": 1.0, "kin": 1.0},
        "signal_count": 0, "risk_level": "low",
        "majority_result": {"votes": {"olf": 0, "img": 0, "kin": 0}},
    }


def grade_answer(rag, case, answer):
    grade_schema = copy.deepcopy(GRADE_SCHEMA)
    coverage_schema = grade_schema["properties"]["expected_claims_covered"]
    coverage_schema["minItems"] = len(case["expected_claims"])
    coverage_schema["maxItems"] = len(case["expected_claims"])
    prompt = (
        "아래 답변의 사실 주장 각각을 gold evidence만으로 엄격히 채점하세요. "
        "일반 상식이나 외부 지식을 사용하지 마세요. 부분적으로만 뒷받침되면 unsupported입니다. "
        "출처 정확성은 Gold source가 참고 목록에 포함되면 참입니다. 다른 출처가 함께 있어도 됩니다. "
        "페이지 정확성은 Gold pages 중 하나가 단일 페이지나 페이지 범위에 포함되면 참입니다.\n\n"
        f"질문: {case['question']}\nGold source: {case['source']}\n"
        f"Gold pages: {case['pages']}\nGold evidence: {case['evidence']}\n"
        f"기대 주장: {case['expected_claims']}\n금지 주장: {case['forbidden_claims']}\n"
        f"평가할 답변:\n{answer}"
    )
    response = rag.client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "당신은 의료 RAG groundedness를 보수적으로 평가하는 채점기입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "groundedness_grade", "strict": True, "schema": grade_schema},
        },
    )
    return json.loads(response.choices[0].message.content)


def summarize(rows):
    factual = [claim for row in rows for claim in row["grade"]["claims"] if claim["label"] != "not_factual"]
    supported = [claim for claim in factual if claim["label"] == "supported"]
    expected = [covered for row in rows for covered in row["grade"]["expected_claims_covered"]]
    return {
        "cases": len(rows), "factual_claims": len(factual),
        "claim_groundedness": len(supported) / max(len(factual), 1),
        "expected_claim_coverage": sum(expected) / max(len(expected), 1),
        "source_citation_accuracy": sum(row["grade"]["citation_source_correct"] for row in rows) / len(rows),
        "page_citation_accuracy": sum(row["grade"]["citation_page_correct"] for row in rows) / len(rows),
        "forbidden_claim_rate": sum(row["grade"]["forbidden_claim_present"] for row in rows) / len(rows),
        "diagnostic_overclaim_rate": sum(row["grade"]["diagnostic_overclaim"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--blind-v2", action="store_true")
    args = parser.parse_args()
    rag = ParkinsonRAG(
        api_key="lm-studio",
        api_base=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        chat_model=GENERATOR_MODEL,
    )
    if args.blind_v2:
        cases = json.loads(BLIND_V2_CASES_PATH.read_text(encoding="utf-8"))
        output_path = BLIND_V2_OUTPUT
        dataset_role, independent = "locked_blind_test_v2", True
    elif args.blind:
        cases = json.loads(BLIND_CASES_PATH.read_text(encoding="utf-8"))
        output_path, dataset_role, independent = BLIND_OUTPUT, "locked_blind_test", True
    else:
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        cases += json.loads(EXTRA_CASES_PATH.read_text(encoding="utf-8"))
        output_path, dataset_role, independent = OUTPUT, "development_regression", False
    rag.search_similar_chunks(cases[0]["question"], top_k=1)

    def evaluate(case):
        print(f"Evaluating {case['id']}...")
        answer = rag.generate_rag_answer(case["question"], neutral_results(), top_k=3)
        grade = grade_answer(rag, case, answer)
        return {"id": case["id"], "question": case["question"], "answer": answer, "grade": grade}

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(evaluate, cases))
    report = {"provider": "LM Studio local", "dataset_role": dataset_role,
              "independent_test": independent,
              "dataset_sha256": hashlib.sha256(
                  json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
              ).hexdigest(),
              "generator_model": GENERATOR_MODEL,
              "judge_model": JUDGE_MODEL,
              "summary": summarize(rows), "results": rows}
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
