from fastmcp import FastMCP, Tool
from pathlib import Path
import chromadb
from chromadb.config import Settings

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "vector_db" / "chroma"

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(CHROMA_DIR),
    )
)
collection = client.get_collection("manufacturing_terms")

mcp = FastMCP(
    name="manufacturing-rag",
    instructions=(
        "제조업 장비/공정 용어 RAG MCP입니다. "
        "도구는 용어/공정 정의를 찾을 때 사용하세요."
    ),
)


@Tool
def search_terms(query: str, top_k: int = 5) -> list[dict]:
    """
    제조 용어/공정에 대한 설명을 검색합니다.
    query: 사용자의 검색어 또는 질문
    """
    result = collection.query(query_texts=[query], n_results=top_k)

    docs = result["documents"][0]
    metas = result["metadatas"][0]

    out = []
    for doc, meta in zip(docs, metas):
        out.append(
            {
                "term": meta.get("term"),
                "category": meta.get("category"),
                "text": doc,
            }
        )
    return out


@Tool
def get_term(term: str) -> list[dict]:
    """
    특정 용어명을 기준으로 검색합니다.
    """
    result = collection.query(
        query_texts=[term],
        n_results=3,
        where={"term": term},
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    return [
        {
            "term": m.get("term"),
            "category": m.get("category"),
            "text": d,
        }
        for d, m in zip(docs, metas)
    ]


if __name__ == "__main__":
    mcp.run()
