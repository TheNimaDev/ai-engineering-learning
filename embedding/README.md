# Semantic Search with Sentence Embeddings

A simple command-line application that retrieves the most semantically relevant documents for a user-provided query using pretrained sentence embeddings.

## About

This project demonstrates semantic search with Python and Sentence Transformers. Instead of matching documents only by exact keywords, the application converts the query and documents into vector embeddings and ranks documents by semantic similarity.

## Features

- Search documents using semantic similarity
- Generate embeddings with a pretrained Sentence Transformers model
- Retrieve and display the top-k most relevant documents
- Display similarity scores for search results
- Load model configuration from environment variables
- Handle empty input and keyboard interruption

## Tech Stack

- Python
- Sentence Transformers
- PyTorch
- python-dotenv

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/TheNimaDev/ai-engineering-learning.git
cd ai-engineering-learning/embedding
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
HF_TOKEN=your_huggingface_token
HF_MODEL=your_sentence_transformer_model
```

Set `HF_MODEL` to the Hugging Face model ID you want to use. The application loads this model to encode the query and documents.

### Run the Project

```bash
python main.py
```

Enter a query to find relevant documents.

The application displays up to three matching documents with their similarity scores. Type `q` to exit.

## How It Works

The application loads a pretrained Sentence Transformers model and converts the stored documents into vector embeddings.

When a user enters a query, the application encodes it into an embedding and uses `util.semantic_search` to find the most similar document embeddings. Results are ranked by cosine similarity, and the top-k matches are returned.

Similarity scores indicate how closely the embeddings match. They are not guaranteed probabilities that a result is correct.

## Learning Goals

- Understand text embeddings and their role in semantic search
- Convert text into vector representations using Sentence Transformers
- Compare embeddings using cosine similarity
- Retrieve and rank relevant documents
- Use pretrained models in a Python application
- Load configuration using environment variables
- Build an interactive command-line application
