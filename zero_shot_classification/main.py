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
            "zero-shot-classification",
            model=model,
        )

        self.labels = ["technology", "sports", "politics"]

    def classify_text(self, text):
        return self.classifier(
            text,
            candidate_labels=self.labels,
        )

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

            result = self.classify_text(text)

            print(f"Labels: {result['labels']}")
            print(f"Scores: {[round(score, 4) for score in result['scores']]}")


def main():
    analyzer = Analyzer()
    try:
        analyzer.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
