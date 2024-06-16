import os

from operator import itemgetter
from typing import List, Tuple
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

app = FastAPI(
    title="Langs",
    version="0.0.0",
    description="Server for Lang-* components combined.",
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

def vector_store(path):
    raw_documents = TextLoader(path).load()
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    store = FAISS.from_documents(documents, embeddings_model)

    return store


# vector store of chat history.
chat_db_path = "./app/data/sample_text_kor.txt"
vectorstore = vector_store(chat_db_path)
retriever = vectorstore.as_retriever(search_type="mmr")

# template
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)


# Add the chain you want to add
add_routes(app, chain, enable_feedback_endpoint=True)

# test
# print("TEST")
# print(chain.invoke({"input": {"question": "what do you know about harrison", "chat_history": []}}))
# print(conversational_qa_chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# curl -X POST http://localhost:8000/invoke -H "Content-Type: application/json" -d '{"input": {"question": "why did Sue and Terry exciting", "chat_history": []}}'
# curl -X POST http://localhost:8000/invoke -H "Content-Type: application/json" -d '{"input": {"question": "수현이 무슨 옷을 하려고 했지", "chat_history": []}}'