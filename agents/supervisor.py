import asyncio
from agents.manufacturing_agent import answer_with_rag


async def route_message(question: str) -> str:
    # 나중에: if "날씨" in question: weather_agent() ...
    # 지금은 제조 한 종류만.
    return await answer_with_rag(question)


if __name__ == "__main__":
    while True:
        q = input("\n[사용자] ")
        if q.lower() in ("exit", "quit"):
            break
        ans = asyncio.run(route_message(q))
        print("\n[Supervisor→Agent 답변]")
        print(ans)
