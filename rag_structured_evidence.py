"""Extract table rows and contextual numeric evidence from academic PDFs."""

import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|p\s*[<=>]\s*\.?\d+", re.IGNORECASE)
STATISTIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*[-–~]\s*\d+(?:\.\d+)?"
    r"|p\s*[<=>]\s*\.?\d+|\b(?:n|or|rr|hr|ci|auc)\s*[=:<>]?\s*\d"
    r"|\d+(?:\.\d+)?\s*(?:fold|times?|years?|months?|items?))",
    re.IGNORECASE,
)


def _clean_cell(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def extract_table_chunks(pdf_path: Path) -> List[Dict[str, Any]]:
    """Return numeric table rows with headers and exact PDF page metadata."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for table_index, table in enumerate(page.extract_tables() or [], 1):
                rows = [[_clean_cell(cell) for cell in row] for row in table if row]
                rows = [row for row in rows if any(row)]
                if len(rows) < 2 or max(map(len, rows), default=0) < 2:
                    continue
                header = rows[0]
                for row_index, row in enumerate(rows[1:], 1):
                    row_text = " | ".join(cell for cell in row if cell)
                    if len(row_text) < 20 or not NUMBER_RE.search(row_text):
                        continue
                    pairs = []
                    for column, value in enumerate(row):
                        if not value:
                            continue
                        label = header[column] if column < len(header) and header[column] else f"column {column + 1}"
                        pairs.append(f"{label}: {value}")
                    text = " | ".join(pairs)
                    chunks.append({
                        "text": f"[Evidence type: Table] {text}",
                        "page": page_number,
                        "page_end": page_number,
                        "section": "Table",
                        "evidence_type": "table",
                        "table_index": table_index,
                        "row_index": row_index,
                    })
    return chunks


def extract_numeric_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve sentences containing study numbers as dedicated evidence."""
    output = []
    seen = set()
    for chunk in chunks:
        body = re.sub(r"^\[Section: [^]]+\]\s*", "", chunk["text"])
        sentences = re.split(r"(?<=[.!?。！？])\s+", body)
        for index, sentence in enumerate(sentences):
            if len(sentence) < 25 or not STATISTIC_RE.search(sentence):
                continue
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            context = " ".join(sentences[start:end]).strip()
            key = (chunk["page"], context)
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "text": f"[Evidence type: Numeric] {context}",
                "page": chunk["page"],
                "page_end": chunk.get("page_end", chunk["page"]),
                "section": chunk.get("section", "Body"),
                "evidence_type": "numeric",
            })
    return output
