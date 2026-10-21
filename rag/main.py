import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib

load_dotenv()

FILE_NAME = "symptoms.pdf"


class Rag:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL"),
            encode_kwargs={'normalize_embeddings': True},
        )

        self.vectorstore = Chroma(
            collection_name='rag_project_database',
            embedding_function=self.model,
            persist_directory='./chroma',
            collection_metadata={'hnsw:space': 'cosine'}
        )

        self.chunks = self.splitter(self.read_pdf_file())
        self.ids = self.id_generator()
        self.vectorstore.add_documents(self.chunks, ids=self.ids)

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

    def id_generator(self):
        return [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in self.chunks]

    def splitter(self, docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=110,
            chunk_overlap=10
        )

        chunks = text_splitter.split_documents(docs)
        return chunks

    def similarity_search(self, query, k=3):
        result_chunks = self.vectorstore.similarity_search_with_score(
            query, k=k)

        result = []

        for index, (doc, distance) in enumerate(result_chunks, 1):
            result.append(
                [f"Source: {index} | Page: {doc.metadata.get('page', '-1')} | Cosine distance: {distance:.4f} => {doc.page_content}"]
            )

        return result

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

            result = self.similarity_search(query)


def main():
    rag = Rag()

    try:
        rag.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
