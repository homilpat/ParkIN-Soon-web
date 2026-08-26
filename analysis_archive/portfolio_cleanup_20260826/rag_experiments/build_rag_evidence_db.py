"""Build the dedicated table, numeric, and caption evidence index."""

import pickle

from rag_chatbot import CHILD_DB_PATH, MODEL_DIR
from rag_evidence_index import build_evidence_db

OUTPUT = MODEL_DIR / "vector_evidence_db.pkl"


def main():
    with CHILD_DB_PATH.open("rb") as handle:
        child_db = pickle.load(handle)
    print({**build_evidence_db(child_db, OUTPUT), "output": str(OUTPUT)})


if __name__ == "__main__":
    main()
