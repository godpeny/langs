from operator import itemgetter
from typing import Annotated

from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from langserve.pydantic_v1 import BaseModel

from app.llms import llm, emb
from app.embed import embed, save_embed, load_embed

# template
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.
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


class ChatHistory(BaseModel):
    """Chat history with the bot."""
    question: str


@tool
def emb_finder(
        message: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    try:
        # message : [2023-08-23.txt]:::(what happen last year?)
        # Step 1 : Separating message with date and question
        date_str, question = message.split(":::")
        # Step 2: Remove the brackets around the dates
        date_str = date_str.strip("[]")
        # Step 3: Convert the string of dates to a list
        date_list = date_str.split(", ")
        question = question.strip("()")

        for date in date_list:
            # vector store of chat history.
            date = date.strip("''")
            print("!! date !!", date)
            loaded_vectorstore = load_embed(date)
            retriever = loaded_vectorstore.as_retriever(search_type="mmr")

            _inputs = RunnableMap(
                standalone_question=RunnablePassthrough.assign()
                                    | CONDENSE_QUESTION_PROMPT
                                    | llm
                                    | StrOutputParser(),
            )

            _context = {
                "context": itemgetter("standalone_question") | retriever | _combine_documents,
                "question": lambda x: x["standalone_question"],
            }

            conversational_qa_chain = (
                    _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()
            )

            chain = conversational_qa_chain.with_types(input_type=ChatHistory)
            return chain.invoke({f"question": {question}})

    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return (
        message
    )