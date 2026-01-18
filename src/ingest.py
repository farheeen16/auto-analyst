from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

print("INGEST STARTED")

def ingest():
    docs = []
    for file in os.listdir("data/pdfs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data/pdfs", file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

    Chroma.from_documents(
        chunks,
        embed,
        persist_directory="chroma_db"
    )

    print("INGEST FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    ingest()
