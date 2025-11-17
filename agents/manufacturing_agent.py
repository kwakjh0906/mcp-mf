import asyncio
import json
import requests
from mcp.client.stdio import StdioClient


OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b-instruct"


async def call_mcp_search(query: str):
    client = await StdioClient.create(
        command=["python", "mcp/manufacturing_mcp.py"]
    )

    tools = await client.list_tools()
    # 여기서는 search_terms만 사용
    result = await client.call_tool(
        name="search_terms",
        arguments={"query": query, "top_k": 5},
    )
    await client.close()
    return result


def call_ollama(chat_prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "너는 제조업 장비/공정 전문가야. "
                        "주어진 context 안에서만 대답하고, "
                        "모르면 모른다고 말해."
                    ),
                },
                {"role": "user", "content": chat_prompt},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


async def answer_with_rag(question: str) -> str:
    # 1) MCP RAG 검색
    tool_result = await call_mcp_search(question)

    # fastmcp ToolResult 객체 → JSON 파싱
    if isinstance(tool_result, str):
        docs = json.loads(tool_result)
    else:
        docs = tool_result

    context_blocks = []
    for d in docs:
        context_blocks.append(f"[{d['term']}]\n{d['text']}")

    context_text = "\n\n".join(context_blocks)

    # 2) Ollama에게 최종 답변 요청
    prompt = (
        f"아래는 제조 용어 사전에서 검색된 내용이야.\n\n"
        f"{context_text}\n\n"
        f"위 내용을 참고해서 다음 질문에 한국어로 자세히 답변해줘:\n\n"
        f"{question}"
    )

    answer = call_ollama(prompt)
    return answer


if __name__ == "__main__":
    # 간단 CLI 테스트
    q = input("질문: ")
    out = asyncio.run(answer_with_rag(q))
    print("\n=== 답변 ===")
    print(out)
