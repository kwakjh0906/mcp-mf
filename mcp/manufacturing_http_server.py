# manufacturing_http_server.py
from fastapi import FastAPI
from chromadb import PersistentClient
from fastapi.responses import JSONResponse
import requests
import traceback

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "vector_db/chroma"


# -----------------------------
# Ollama Embeddings API
# -----------------------------
def embed(text: str):
    """Ollama embedding API for nomic-embed-text"""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBED_MODEL,
                "prompt": text   # ← ★ 반드시 prompt!
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        # "embedding" 필드가 정상적으로 있는 경우
        if "embedding" in data:
            return data["embedding"]

        # 비정상 응답을 대비한 로그
        print("[Embedding Warning] 예상치 못한 응답 구조:", data)
        return None

    except Exception as e:
        print("[Embedding 오류]", e)
        traceback.print_exc()
        return None

# -----------------------------
# ChromaDB 로드
# -----------------------------
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="manufacturing_terms",
    metadata={"hnsw:space": "cosine"}
)

# -----------------------------
# Search API
# -----------------------------
@app.get("/search_terms")
def search_terms(query: str, top_k: int = 5):
    embedding = embed(query)

    if embedding is None:
        return JSONResponse(
            {"error": "Embedding 생성 실패"},
            status_code=500
        )

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )

    docs = []
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        docs.append({
            "term": meta["term"],
            "category": meta["category"],
            "text": doc
        })

    # FastAPI가 UTF-8 JSON으로 반환
    return JSONResponse(content=docs)
