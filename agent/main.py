import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()


class Agent:
    def __init__(self):
        self.llm_model_name = os.getenv("GROQ_MODEL")

        if not self.llm_model_name:
            raise ValueError(
                "GROQ_MODEL is missing from the .env file."
            )

        self.llm_model = ChatGroq(model=self.llm_model_name, temperature=0.5)

        self.search_tool = TavilySearch(max_results=3)

        self.messages = []

        self.agent = create_agent(
            model=self.llm_model,
            tools=[self.get_weather, self.search_tool],
            system_prompt=(
                "You are a helpful conversational assistant. "
                "Answer the given questions. "
                "Use the provided tools whenever needed. "
                "If necessary, use the web search tool before responding. "
                "Provide concise and accurate answers."
            )
        )

    def get_weather(self, city: str):
        """Get the current weather for a given city."""
        return {
            "city": city,
            "weather": "sunny",
            "temperature": 22,
            "unit": "C",
        }

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
