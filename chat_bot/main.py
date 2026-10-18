import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url='https://api.groq.com/openai/v1'
)


def bot(message):
    response = LLM.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=100,
    )

    return response.choices[0].message.content


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


if __name__ == "__main__":
    main()
