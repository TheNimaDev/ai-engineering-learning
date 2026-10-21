import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import hashlib

load_dotenv()

FILE_NAME = "symptoms.pdf"


class Rag:
    def __init__(self):
        self.embedding_model_name = os.getenv("HF_MODEL")
        self.llm_model_name = os.getenv("GROQ_MODEL")
        self.llm_api_key = os.getenv("GROQ_API_KEY")

        if not self.llm_api_key or not self.embedding_model_name or not self.llm_model_name:
            raise ValueError(
                "HF_MODEL or GROQ_API_KEY or GROQ_MODEL is missing from the .env file."
            )

        self.llm_model = OpenAI(
            api_key=self.llm_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={'normalize_embeddings': True},
        )

        self.vectorstore = Chroma(
            collection_name='rag_project_database',
            embedding_function=self.embedding_model,
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
            print(f"Source {index} | Distance: {distance:.4f} | Page: {doc.metadata['page']}")
            result.append(
                f"[Source {index} | Page {doc.metadata.get('page', '-1')}]\n"
                f"{doc.page_content}"
            )

        return result

    def llm(self, query, context):
        system_prompt = f'''You are a medical information assistant. Answer the user's question using ONLY
        the provided source documents below.
    
        Rules you must follow:
        1. Base your answer ONLY on the provided sources — do not use outside knowledge
        2. If the sources don't contain enough information to answer, say exactly:
        "I don't have enough information to answer this question."
        3. Cite sources using [Source N] notation whenever you use information from them
        4. If sources contradict each other, mention the contradiction
        5. Be concise — answer in 3-5 sentences unless the question requires more detail
        6. Never speculate or make assumptions beyond what the sources say
    
        Sources:
        {context}'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query},
        ]

        response = self.llm_model.chat.completions.create(
            model=self.llm_model_name,
            temperature=0.5,
            messages=messages
        )

        return response.choices[0].message.content

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

            search_result = self.similarity_search(query)

            context = '\n\n_____\n\n'.join(search_result)

            try:
                result = self.llm(query, context)
            except Exception as error:
                print(f"Failed to get response from the LLM: {error}")
                continue

            print(f"\n{result}\n")


def main():
    rag = Rag()

    try:
        rag.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
