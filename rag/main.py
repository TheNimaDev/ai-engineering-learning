from pypdf import PdfReader
from langchain_core.documents import Document

FILE_NAME = ""


class Rag:
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