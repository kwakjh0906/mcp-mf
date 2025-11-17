# agents/manufacturing_agent.py
import httpx

RAG_SERVER_URL = "http://localhost:8000"

async def call_rag_search(query: str):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{RAG_SERVER_URL}/search_terms",
                params={"query": query, "top_k": 5},
                timeout=10
            )
            resp.raise_for_status()

            # ★ 여기서 JSON 파싱 실패하면 RAG 서버로부터 받은 실제 내용을 출력
            try:
                return resp.json()
            except Exception as e:
                return {"error": f"JSON 파싱 오류: {str(e)}", "raw": resp.text}

    except httpx.HTTPError as e:
        return {"error": f"HTTP 오류: {str(e)}"}

    except Exception as e:
        return {"error": f"RAG 서버 호출 오류: {str(e)}"}


async def manufacturing_agent(question: str):
    print("[DEBUG] manufacturing_agent 질문:", question)
    docs = await call_rag_search(question)
    print("[DEBUG] manufacturing_agent 결과 docs:", docs)
    return docs
