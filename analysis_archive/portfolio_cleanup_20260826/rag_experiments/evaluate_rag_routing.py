"""Regression test for academic, lifestyle, and mixed RAG routing."""

import json
from pathlib import Path

from openai import OpenAI

from rag_question_router import route_question

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "modeling/rag_routing_evaluation.json"
MODEL = "parkinsoon-eval"

CASES = [
    ("후각 저하는 운동 증상보다 몇 년 먼저 나타나나요?", "academic"),
    ("변비 메타분석의 통합 오즈비는 얼마인가요?", "academic"),
    ("나선 모델의 민감도와 특이도는 얼마인가요?", "academic"),
    ("집에서 어떤 운동을 하면 좋나요?", "lifestyle"),
    ("변비가 있는데 물과 음식은 어떻게 먹어야 하나요?", "lifestyle"),
    ("넘어지지 않으려면 생활에서 무엇을 조심해야 하나요?", "lifestyle"),
    ("변비가 위험과 관련 있나요, 생활에서는 뭘 해야 하나요?", "mixed"),
    ("후각 연구 근거와 일상 관리법을 함께 알려주세요.", "mixed"),
    ("나선 검사 신호의 연구 의미와 도움이 되는 운동을 함께 알려주세요.", "mixed"),
]


def main():
    client = OpenAI(api_key="lm-studio", base_url="http://127.0.0.1:1234/v1")
    rows = []
    for question, expected in CASES:
        predicted = route_question(client, MODEL, question)
        rows.append({
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
        })
    summary = {
        "cases": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "errors": sum(not row["correct"] for row in rows),
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
