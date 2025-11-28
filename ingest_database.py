import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "C:/Users/Ritam Choudhury/rag-chatbot/chroma_db"
DATA_PATH = "C:/Users/Ritam Choudhury/rag-chatbot/data/books"


def main():
    generate_datastore()


def generate_datastore():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_text(documents)

    print("Saving to Chroma DB...")
    save_to_chroma(chunks)

    print("Ingestion complete.")


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear old DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_PATH,
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
