# Zero-Shot Text Classification

A simple command-line application that classifies text into predefined categories using a pretrained Hugging Face model.

## About

This project demonstrates zero-shot text classification with Python and Hugging Face Transformers. Unlike traditional classification models that are trained for specific categories, zero-shot classification allows users to provide candidate labels at inference time without training a new model.

## Features

- Classify text into predefined categories
- Use a pretrained Hugging Face model
- Display predicted labels and classification scores
- Load model configuration from environment variables
- Handle empty input and keyboard interruption

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

Clone the repository:

```bash
git clone https://github.com/TheNimaDev/ai-engineering-learning.git
cd ai-engineering-learning/zero_shot_classification
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
HF_MODEL=typeform/distilbert-base-uncased-mnli
```

The model is configured through the `HF_MODEL` environment variable.

### Run the Project

```bash
python main.py
```

Enter text to classify it into one of the supported categories:

- Technology
- Sports
- Politics

Type `q` to exit.

## How It Works

The application uses the `zero-shot-classification` pipeline from Hugging Face Transformers. It receives input text and candidate labels, then returns the labels ranked by their scores.

The scores represent the model's relative assessment of how well each label matches the input. They should not be interpreted as guaranteed probabilities of correctness.

## Learning Goals

- Understand zero-shot classification
- Work with pretrained transformer models
- Use Hugging Face pipelines
- Load configuration using environment variables
- Build interactive command-line applications

