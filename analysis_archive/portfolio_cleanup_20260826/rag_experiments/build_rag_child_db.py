"""Build sentence-level child embeddings from the active parent vector DB."""

import pickle

from rag_chatbot import DB_PATH, MODEL_DIR, get_local_embedder
from rag_hierarchical import save_child_db

OUTPUT = MODEL_DIR / "vector_child_db.pkl"


def main():
    with DB_PATH.open("rb") as handle:
        parent_db = pickle.load(handle)
    count = save_child_db(parent_db, get_local_embedder(), OUTPUT)
    print({"child_chunks": count, "output": str(OUTPUT)})


if __name__ == "__main__":
    main()
