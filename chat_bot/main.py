import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Introduce yourself in your first response. "
            "Keep your answers short and concise."
        ),
    }
]

LLM = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url='https://api.groq.com/openai/v1'
)


def bot(message):
    messages.append({"role": "user", "content": message})

    response = LLM.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )

    response_content = response.choices[0].message.content

    messages.append({"role": "assistant", "content": response_content})

    return response_content


def chat():
    while True:
        user_input = input(
            "Ask something (type 'q' to terminate): "
        ).strip()

        if user_input.lower() == "q":
            print("Good Luck")
            break

        bot_answer = bot(user_input)
        print(bot_answer)


def main():
    chat()
    print(messages)


if __name__ == "__main__":
    main()
