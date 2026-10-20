import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()


class Analyzer:
    def __init__(self):
        model = os.getenv("HF_MODEL")

        if not model:
            raise ValueError("HF_MODEL is missing from the .env file.")

        self.classifier = pipeline(
            "sentiment-analysis",
            model=model,
        )

    def analyze_sentiment(self, text):
        result = self.classifier(text)[0]
        return result

    def chat(self):
        while True:
            text = input(
                "Enter some text (type q to terminate): "
            ).strip()

            if text.lower() == "q":
                print("Good Luck")
                break

            if not text:
                print("Please enter some text.")
                continue

            sentiment = self.analyze_sentiment(text)

            print(f"Sentiment: {sentiment['label']}")
            print(f"Confidence: {sentiment['score']:.4f}")


def main():
    analyzer = Analyzer()

    try:
        analyzer.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
