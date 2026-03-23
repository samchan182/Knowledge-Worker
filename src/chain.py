import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

SYSTEM_PROMPT = """You are an internal assistant for Dragon Palace restaurant.
Answer questions about recipes, allergens, and kitchen operations using ONLY the context provided below.
If the answer is not in the context, say you don't have that information.
Be precise and conservative about allergen information — this is safety-critical.

Context:
{context}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_chain():
    """Build the RAG chain using modern LCEL."""
    retriever, _ = get_retriever()

    llm = ChatOpenAI(
        temperature=LLM_TEMPERATURE,
        model_name=LLM_MODEL,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, question, chat_history):
    """Run one turn and return (answer, updated_history)."""
    answer = chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history
