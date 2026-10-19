# Chat Bot

A simple command-line chatbot built with Python and the Groq API, as part of my journey toward AI Engineering.

## Features

- Stream AI responses
- Maintain conversation history
- Summarize older messages
- Track token usage and estimate API costs
- Limit input length

## Tech Stack

- Python
- Groq API
- OpenAI Python SDK
- Hugging Face Transformers
- python-dotenv

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/TheNimaDev/ai-engineering-learning.git
cd ai-engineering-learning/chat_bot
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file using the following variables:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_model_id
HF_TOKEN=your_huggingface_token
```

Get your [Groq API key](https://console.groq.com/keys) and [Hugging Face token](https://huggingface.co/settings/tokens).

**5. Run the chatbot**

```bash
python main.py
```

Type `q` to exit.

## Learning Goals

Practice Python, LLM APIs, streaming, tokenization, and conversation summarization.