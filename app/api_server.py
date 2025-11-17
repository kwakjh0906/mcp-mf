from fastapi import FastAPI
from agents.supervisor import route_message

app = FastAPI()

@app.get("/chat")
async def chat(q: str):
    answer = await route_message(q)
    return {"answer": answer}
