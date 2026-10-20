# Sentiment Analysis

A simple command-line sentiment analysis tool built with Python and Hugging Face Transformers.

## About

This project analyzes the sentiment of user-provided text using a pretrained model. It predicts whether the text is positive or negative and displays the prediction confidence.

## Features

- [x] Analyze text sentiment
- [x] Display predicted label and confidence score
- [x] Load the model ID from environment variables
- [x] Interactive CLI with exit command
- [x] Handle empty input and keyboard interruption

## Tech Stack

- Python
- Hugging Face Transformers
- PyTorch
- python-dotenv

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### Installation

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

Create a `.env` file and configure the model:

```dotenv
HF_TOKEN=your_huggingface_token
HF_MODEL=distilbert/distilbert-base-uncased-finetuned-sst-2-english
```

### Run the Project

```bash
python main.py
```

Enter text to analyze its sentiment. Type `q` to exit.

## Learning Goals

- Use pretrained models with Hugging Face Transformers
- Understand sentiment classification and confidence scores
- Load configuration from environment variables
- Build an interactive Python CLI
