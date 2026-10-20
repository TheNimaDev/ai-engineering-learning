import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import util
import torch

load_dotenv()

FILE_NAME = "symptoms.pdf"


class Rag:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL"),
            encode_kwargs={'normalize_embeddings': True},
        )

        self.chunks = self.splitter(self.read_pdf_file())
        self.vectors = self.embed_chunks(self.chunks)

    def read_pdf_file(self):
        reader = PdfReader(FILE_NAME)

        docs = []

        for index, page in enumerate(reader.pages):
            page_text = page.extract_text()

            if not page_text:
                continue

            doc = Document(
                page_content=page_text,
                metadata={"source": FILE_NAME, "page": index + 1},
            )

            docs.append(doc)

        return docs

    def splitter(self, docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=110,
            chunk_overlap=10
        )

        chunks = text_splitter.split_documents(docs)
        return chunks

    def embed_chunks(self, chunks):
        chunk_texts = [chunk.page_content for chunk in chunks]

        vectors = self.model.embed_documents(chunk_texts)

        return vectors

    def semantic_search(self, query, top_k=3):
        query_embedding = self.model.embed_query(query)

        search_results = util.semantic_search(
            torch.tensor([query_embedding]),
            torch.tensor(self.vectors),
            top_k=top_k,
        )[0]

        return [
            {
                "chunk": self.chunks[result["corpus_id"]],
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

            for result in results:
                print(result["score"])
                print(result["chunk"].page_content)


def main():
    rag = Rag()

    try:
        rag.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
