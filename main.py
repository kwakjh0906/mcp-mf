import asyncio
from agents.supervisor import route_message

def main():
    print("=== 제조 AI 챗봇 ===")
    print("종료하려면 'exit' 입력\n")

    while True:
        user_input = input("질문 : ")

        if user_input.lower().strip() in ["exit", "quit"]:
            print("종료합니다.")
            break

        try:
            answer = asyncio.run(route_message(user_input))
            print("\n제조 AI:\n" + answer + "\n")
        except Exception as e:
            print(f"\n[오류 발생] {e}\n")

if __name__ == "__main__":
    main()
