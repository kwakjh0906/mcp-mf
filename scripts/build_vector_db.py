from pathlib import Path
import json
import requests
from chromadb import PersistentClient

BASE_DIR = Path(__file__).resolve().parent.parent
JSONL_PATH = BASE_DIR / "data" / "manufacturing_terms.jsonl"
CHROMA_DIR = BASE_DIR / "vector_db" / "chroma"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def build_chroma():
    print("Chroma PersistentClient 생성 중…")

    client = PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name="manufacturing_terms",
        metadata={"hnsw:space": "cosine"}
    )

    print("JSONL 로드…")
    docs = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"총 {len(docs)}개 문서 임베딩 시작…")

    for idx, item in enumerate(docs):
        emb = embed(item["text"])
        
        # 임베딩 차원 출력
        if idx == 0:
            print(f"첫 임베딩 차원: {len(emb)}")

        collection.add(
            ids=[f"term_{idx}"],
            embeddings=[emb],
            documents=[item["text"]],
            metadatas=[{
                "term": item["term"],
                "category": item["category"]
            }]
        )

        if (idx + 1) % 10 == 0:
            print(f"진행률: {idx + 1}/{len(docs)}")

    print("임베딩 완료!")
    print(f"저장 위치: {CHROMA_DIR}")


if __name__ == "__main__":
    build_chroma()
