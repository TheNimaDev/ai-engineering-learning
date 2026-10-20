import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()


class Embedding:
    def __init__(self):
        model_name = os.getenv("HF_MODEL")

        if not model_name:
            raise ValueError("HF_MODEL is missing from the .env file.")

        self.model = SentenceTransformer(model_name)

        self.documents = [
            "A car needs a healthy battery to start.",
            "Football is a popular sport.",
            "Cold weather can make car engines harder to start.",
        ]

        self.documents_embeddings = self.embed(self.documents)

    def embed(self, text):
        return self.model.encode(text)

    def semantic_search(self, query, top_k=3):
        query_embedding = self.embed(query)

        search_results = util.semantic_search(
            query_embedding,
            self.documents_embeddings,
            top_k=top_k,
        )[0]

        return [
            {
                "document": self.documents[result["corpus_id"]],
                "score": result["score"],
            }
            for result in search_results
        ]

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

            results = self.semantic_search(query)

            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. {result['document']} "
                    f"(score: {result['score']:.4f})"
                )


def main():
    embedding = Embedding()

    try:
        embedding.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
