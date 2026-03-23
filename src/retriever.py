import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import DB_NAME, TOP_K, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def get_retriever():
    """Load persisted vector store and return a retriever."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )
    return retriever, vectorstore
