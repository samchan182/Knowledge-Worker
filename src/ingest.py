import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import (
    OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    DB_NAME, KNOWLEDGE_BASE_DIR
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def add_metadata(doc, doc_type):
    """Tag each document with its category."""
    doc.metadata["doc_type"] = doc_type
    return doc


def load_documents():
    """Read all .md files from knowledge-base directory."""
    text_loader_kwargs = {"encoding": "utf-8"}

    md_files = glob.glob(f"{KNOWLEDGE_BASE_DIR}/**/*.md", recursive=True)

    documents = []
    for file_path in md_files:
        loader = TextLoader(file_path, **text_loader_kwargs)
        file_docs = loader.load()
        doc_type = os.path.splitext(os.path.basename(file_path))[0]
        documents.extend([add_metadata(doc, doc_type) for doc in file_docs])

    return documents


def chunk_documents(documents):
    """Split documents into overlapping chunks."""
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and persist to ChromaDB."""
    embeddings = OpenAIEmbeddings()

    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )
    return vectorstore


def run_ingestion():
    """Full ingestion pipeline."""
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s)")
    print(f"Document types: {set(doc.metadata['doc_type'] for doc in documents)}")

    print("Chunking...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Building vector store...")
    vectorstore = build_vectorstore(chunks)
    count = vectorstore._collection.count()
    print(f"Vector store ready: {count} vectors stored")

    return vectorstore


if __name__ == "__main__":
    run_ingestion()
