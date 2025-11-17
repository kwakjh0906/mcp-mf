# agents/supervisor.py
import requests
import asyncio
from agents.manufacturing_agent import manufacturing_agent

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:3b-instruct"


def call_ollama(prompt: str) -> str:
    """Ollama Chat API 호출"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system",
                     "content": "너는 제조 전문가 Supervisor다. 제공된 문서 내에서만 답해라."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=30,
        )

        resp.raise_for_status()
        data = resp.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        if "error" in data:
            return f"[LLM 오류] {data['error']}"

        return "[LLM 응답 오류: content 누락]"

    except Exception as e:
        return f"[LLM 호출 실패] {str(e)}"


async def route_message(question: str) -> str:
    print("[DEBUG] supervisor 입력:", question)

    # 1) manufacturing agent 검색
    docs = await manufacturing_agent(question)
    print("[DEBUG] supervisor 결과:", docs)

    # RAG 실패 처리
    if isinstance(docs, dict) and "error" in docs:
        return docs["error"]

    # 2) context 생성
    context = "\n".join(
        f"[{d['term']}]\n{d['text']}"
        for d in docs
    )

    # 3) 최종 prompt
    final_prompt = f"""아래는 제조 용어 관련 참고 문서들이다.

{context}

위 문서를 참고하여 다음 질문에 가장 정확하게 답하라:
'{question}'
"""
    answer = await asyncio.to_thread(call_ollama, final_prompt)
    return answer
