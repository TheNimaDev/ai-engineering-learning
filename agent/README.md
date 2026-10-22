# Conversational AI Agent

A simple command-line conversational AI agent built with LangChain and Groq, with tools for web search and weather information.

## About

This project demonstrates how to build a tool-using conversational AI agent with Python and LangChain. The agent uses a Groq-hosted language model to understand user queries, maintain conversation history, and call tools when needed.

It integrates Tavily Search for web searches and a custom weather tool to demonstrate tool calling.

## Features

- Interactive command-line conversation
- Maintain conversation history across messages
- Use a Groq-hosted language model
- Search the web using Tavily
- Call a custom weather tool
- Load model configuration from environment variables
- Handle empty input and keyboard interruption

## Tech Stack

- Python
- LangChain
- LangChain Groq
- LangChain Tavily
- python-dotenv

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- A Groq API key
- A Tavily API key

### Installation

Clone the repository:

```bash
git clone https://github.com/TheNimaDev/ai-engineering-learning.git
cd ai-engineering-learning/agent
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model_name
TAVILY_API_KEY=your_tavily_api_key
```

- `GROQ_API_KEY`: API key for accessing the Groq language model.
- `GROQ_MODEL`: The model identifier used by `ChatGroq`.
- `TAVILY_API_KEY`: API key for web search.

Keep your API keys private and never commit your `.env` file.

### Run the Project

```bash
python main.py
```

Enter a message to start a conversation with the agent.

Type `q` to exit.

## Available Tools

### Web Search

Uses Tavily Search to retrieve web results for queries that require external information. The tool is configured to return up to three results per search.

### Weather

A custom tool that accepts a city name and returns weather information.

**Note:** The weather tool currently returns hardcoded sample data. It does not retrieve real-time weather information.

## How It Works

1. The user enters a message in the command line.
2. The message is added to the conversation history.
3. The LangChain agent processes the conversation using the configured Groq model.
4. When appropriate, the agent calls an available tool, such as Tavily Search or the custom weather function.
5. The agent uses the tool output to generate a response.
6. The conversation history is updated for subsequent interactions.

## Learning Goals

- Understand AI agents and tool calling
- Build agents using LangChain
- Integrate a hosted LLM through Groq
- Perform web searches using Tavily
- Maintain conversational history
- Manage API keys and configuration with environment variables
- Build interactive command-line AI applications