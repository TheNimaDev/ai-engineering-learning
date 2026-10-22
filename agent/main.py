import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()


class Agent:
    def __init__(self):
        self.llm_model_name = os.getenv("GROQ_MODEL")

        if not self.llm_model_name:
            raise ValueError(
                "GROQ_MODEL is missing from the .env file."
            )

        self.llm_model = ChatGroq(model=self.llm_model_name, temperature=0.5)

        self.messages = []

        self.agent = create_agent(
            model=self.llm_model,
            system_prompt=(
                "You are a helpful conversational assistant. "
                "Answer the given questions. "
                "Use the provided tools whenever needed. "
                "Provide concise and accurate answers."
            )
        )

    def llm(self, query: str) -> str:
        self.messages.append(HumanMessage(content=query))

        result = self.agent.invoke({"messages": self.messages})

        self.messages = result["messages"]

        return self.messages[-1].content

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
