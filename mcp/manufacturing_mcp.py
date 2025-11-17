# mcp/manufacturing_http_server.py
from fastapi import FastAPI
from chromadb import PersistentClient

app = FastAPI()

client = PersistentClient(path="vector_db/chroma")
collection = client.get_or_create_collection("manufacturing_terms")

@app.get("/search_terms")
def search_terms(query: str, top_k: int = 5):
    result = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    return [
        {
            "term": m["term"],
            "category": m["category"],
            "text": d
        }
        for d, m in zip(docs, metas)
    ]
