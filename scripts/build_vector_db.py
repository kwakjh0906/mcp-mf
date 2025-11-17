from pathlib import Path
import json
import requests
import chromadb
from chromadb.config import Settings

BASE_DIR = Path(__file__).resolve().parent.parent
JSONL_PATH = BASE_DIR / "data" / "manufacturing_terms.jsonl"
CHROMA_DIR = BASE_DIR / "vector_db" / "chroma"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "qwen2.5:3b-instruct"


def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def build_chroma():
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(CHROMA_DIR),
        )
    )

    collection = client.get_or_create_collection(
        name="manufacturing_terms",
        metadata={"hnsw:space": "cosine"},
    )

    docs = []
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"총 {len(docs)}개 용어 임베딩 시작…")

    for i, doc in enumerate(docs):
        vector = embed(doc["text"])
        collection.add(
            ids=[f"term_{i}"],
            embeddings=[vector],
            documents=[doc["text"]],
            metadatas=[
                {
                    "term": doc["term"],
                    "category": doc["category"],
                }
            ],
        )
        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{len(docs)}")

    client.persist()
    print(f"Chroma 벡터DB 저장 완료: {CHROMA_DIR}")


if __name__ == "__main__":
    build_chroma()
