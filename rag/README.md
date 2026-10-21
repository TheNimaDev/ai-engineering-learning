# Retrieval-Augmented Generation (RAG)

A command-line application that retrieves relevant information from a PDF and uses a large language model (LLM) to generate answers grounded in the retrieved content.

## About

This project demonstrates Retrieval-Augmented Generation (RAG) with Python, Hugging Face embeddings, Chroma, and the Groq API.

The application extracts text from a PDF, splits it into smaller chunks, stores their embeddings in a vector database, and retrieves relevant chunks when a user asks a question. The retrieved content is then provided to an LLM to generate a source-grounded answer.

## Features

- Extract text from PDF pages
- Split documents into smaller text chunks
- Generate embeddings using a pretrained Hugging Face model
- Store and retrieve document chunks using Chroma
- Search for relevant content using semantic similarity
- Generate answers with a Groq-hosted LLM
- Include source and page references in retrieved context
- Load model configuration and API credentials from environment variables
- Handle empty input and keyboard interruption

## Tech Stack

- Python
- LangChain
- Hugging Face Sentence Transformers
- Chroma
- Groq API
- OpenAI Python SDK
- pypdf
- python-dotenv

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- A Groq API key
- A Hugging Face embedding model
- A PDF knowledge source (`symptoms.pdf`)

### Installation

Clone the repository:

```bash
git clone https://github.com/TheNimaDev/ai-engineering-learning.git
cd ai-engineering-learning/rag
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Place `symptoms.pdf` in the project directory, alongside `main.py`.

### Configuration

Create a `.env` file in the project directory:

```env
# Get your API key from https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key

# Model ID from https://console.groq.com/docs/models
GROQ_MODEL=your_groq_model_id

# Model ID from https://huggingface.co/models
HF_MODEL=your_huggingface_embedding_model

# Get your API key from https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token
```

The application reads `GROQ_API_KEY`, `GROQ_MODEL`, and `HF_MODEL` from the environment. `HF_TOKEN` is included for Hugging Face configuration, but the current code does not directly read it.

Do not commit your `.env` file or expose API keys.

### Run the Project

Start the application:

```bash
python main.py
```

Enter a question about information covered by the PDF. The application retrieves relevant document chunks and sends them to the LLM as context.

Type `q` to exit.

## How It Works

1. **PDF extraction:** `pypdf` extracts text from each page of `symptoms.pdf`.
2. **Text splitting:** LangChain's `RecursiveCharacterTextSplitter` divides page content into smaller chunks with overlap.
3. **Embedding generation:** A Hugging Face embedding model converts the chunks into vectors. Embeddings are normalized.
4. **Vector storage:** Chroma stores the chunks and embeddings in the local `./chroma` directory, using cosine distance for similarity search.
5. **Retrieval:** When a user submits a question, Chroma retrieves the top matching chunks and their cosine distances.
6. **Answer generation:** The retrieved chunks are passed to a Groq-hosted LLM. The system prompt instructs the model to answer using only the provided sources and cite them using source labels.

Cosine distance indicates how far apart vectors are under the configured distance metric. A lower distance generally indicates a closer match, but retrieved content should still be checked for relevance.

## Learning Goals

- Understand the core RAG workflow
- Extract and preprocess text from PDF documents
- Work with text chunking and overlap
- Generate embeddings using a pretrained model
- Store and search vectors with Chroma
- Connect a retrieval pipeline to an LLM
- Ground LLM responses in retrieved source material
- Manage configuration using environment variables
- Build an interactive command-line application
