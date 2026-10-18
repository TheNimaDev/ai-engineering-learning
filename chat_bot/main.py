import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ChatBot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url='https://api.groq.com/openai/v1'
        )

        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Introduce yourself in your first response. "
                    "Keep your answers short and concise."
                ),
            }
        ]

        self.total_tokens = 0

    def bot(self, message):
        self.messages.append({"role": "user", "content": message})

        response = self.llm.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=self.messages,
            temperature=0.7,
            max_tokens=100,
        )

        self.total_tokens += response.usage.total_tokens

        response_content = response.choices[0].message.content
        self.messages.append(
            {"role": "assistant", "content": response_content})

        return response_content

    def chat(self):
        while True:
            user_input = input(
                "Ask something (type 'q' to terminate): "
            ).strip()

            if user_input.lower() == "q":
                print("Good Luck")
                break

            bot_answer = self.bot(user_input)
            print(bot_answer)

    def report(self):
        for message in self.messages:
            print(message)
        print(f"Consume tokens: {self.total_tokens}")


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == "__main__":
    main()
