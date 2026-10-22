import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


class Agent:
    def __init__(self):
        self.llm_model_name = os.getenv("GROQ_MODEL")

        if not self.llm_model_name:
            raise ValueError(
                "GROQ_MODEL is missing from the .env file."
            )

        self.llm_model = ChatGroq(model=self.llm_model_name, temperature=0.5)

        self.messages = [
            SystemMessage('Give short answers.')
        ]

    def llm(self, query):
        self.messages.append(HumanMessage(query))

        result = self.llm_model.invoke(self.messages)

        self.messages.append(result)

        return result.content

    def chat(self):
        while True:
            query = input(
                "Enter some text (type q to terminate): "
            ).strip()

            if query.lower() == "q":
                print("Good Luck")
                break

            if not query:
                print("Please enter some text.")
                continue
            try:
                result = self.llm(query)
                print(result)
            except Exception as error:
                print(f"Failed to get response from the LLM: {error}")
                continue


def main():
    agent = Agent()

    try:
        agent.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
