from fastapi import FastAPI
from chromadb import PersistentClient
import requests

app = FastAPI()

OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "/data/chroma"

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("manufacturing_terms")

def embed(text: str):
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

@app.get("/search")
def search_terms(q: str, top_k: int = 5):
    emb = embed(q)
    result = collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )
    docs = []
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        docs.append({
            "term": meta["term"],
            "category": meta["category"],
            "text": doc
        })
    return docs
