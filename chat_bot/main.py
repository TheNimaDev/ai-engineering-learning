import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from transformers import AutoTokenizer

load_dotenv()

INPUT_PRICE_PER_MILLION = 0.05
OUTPUT_PRICE_PER_MILLION = 0.08
TOKEN_LIMIT = 10_000
INPUT_TOKEN_LIMIT = 10
SUMMARY_INTERVAL = 5


class ChatBot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        self.system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Introduce yourself in your first response. "
                "Keep your answers short and concise."
            ),
        }

        self.messages = []
        self.summary = None

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.iteration = 0

    def track_usage(self, usage):
        if usage is None:
            return

        self.total_tokens += usage.total_tokens
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens

    def summarize_messages(self):
        if self.iteration % SUMMARY_INTERVAL != 0:
            return

        messages_to_keep = SUMMARY_INTERVAL

        if len(self.messages) <= messages_to_keep:
            return

        split_index = len(self.messages) - messages_to_keep

        split_index -= split_index % 2

        if split_index <= 0:
            return

        early_messages = self.messages[:split_index]
        recent_messages = self.messages[split_index:]

        summary_input = []

        if self.summary:
            summary_input.append({
                "role": "user",
                "content": f"Previous conversation summary:\n{self.summary}",
            })

        summary_input.extend(early_messages)

        response = self.llm.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the conversation concisely. "
                        "Preserve important facts, user preferences, "
                        "decisions, and context needed for future replies. "
                        "Incorporate the previous summary if provided."
                    ),
                },
                *summary_input,
            ],
        )

        self.track_usage(response.usage)
        self.summary = response.choices[0].message.content

        self.messages = recent_messages

        print("\n[Conversation summary updated]")

    def bot(self, message):
        self.messages.append({"role": "user", "content": message})

        request_messages = [self.system_message]

        if self.summary:
            request_messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self.summary}",
            })

        request_messages.extend(self.messages)

        try:
            response = self.llm.chat.completions.create(
                model=os.getenv("GROQ_MODEL"),
                messages=request_messages,
                temperature=0.7,
                max_tokens=100,
                stream=True,
                stream_options={"include_usage": True},
            )

            response_text = ""

            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content

                    if delta:
                        print(delta, end="", flush=True)
                        response_text += delta
                        time.sleep(0.1)

                if chunk.usage:
                    self.track_usage(chunk.usage)

            print()

            self.messages.append(
                {"role": "assistant", "content": response_text})

            self.iteration += 1
            self.summarize_messages()

            return response_text

        except Exception:
            if (
                self.messages
                and self.messages[-1]["role"] == "user"
                and self.messages[-1]["content"] == message
            ):
                self.messages.pop()

            raise

    def chat(self):
        model_name = os.getenv("GROQ_MODEL")

        if not model_name:
            raise ValueError("GROQ_MODEL is missing from the .env file.")

        encoding = AutoTokenizer.from_pretrained(model_name)

        while True:
            user_input = input(
                "Ask something (type 'q' to terminate): "
            ).strip()

            if user_input.lower() == "q":
                print("Good Luck")
                break

            if not user_input:
                continue

            token_count = len(encoding.encode(user_input))

            if token_count > INPUT_TOKEN_LIMIT:
                print("Your message is too long. Please try a shorter one.")
                continue

            if self.total_tokens >= TOKEN_LIMIT:
                print("Unfortunately, you exceeded the token limit for this session.")
                break

            self.bot(user_input)

    def report(self):
        print("\n--- Conversation ---")
        print(self.system_message)

        if self.summary:
            print({
                "role": "system",
                "content": f"Earlier conversation summary: {self.summary}",
            })

        for message in self.messages:
            print(message)

        cost = (
            self.input_tokens * INPUT_PRICE_PER_MILLION
            + self.output_tokens * OUTPUT_PRICE_PER_MILLION
        ) / 1_000_000

        print(f"Consumed tokens: {self.total_tokens}")
        print(f"Estimated cost: ${cost:.6f}")


def main():
    bot = ChatBot()

    try:
        bot.chat()
    finally:
        bot.report()


if __name__ == "__main__":
    main()
