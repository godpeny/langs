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

from app.llms import llm
from app.embed import load_embed

# template
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.
also, remember to pass chat id to the next agent.
Follow Up Input: {question}
Chat ID(chat_id) : {chat_id}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context. Please put this 'chat_id' in the answer as prefix. 
For example, if the 'chat_id' is 'cidc598a120', the answer should be format of 'cidc598a120: ANSWER':
{context}

Chat ID(chat_id) : {chat_id}
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
    chat_id: str


@tool
def emb_finder(
        message: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    try:
        # message : [2023-08-23_cidc598a120.txt]:::(what happen last year?)
        # Step 1 : Separating message with date and question
        index_name, question = message.split(":::")
        # Step 2: Remove the brackets around the dates
        index_name_str = index_name.strip("[]")
        # Step 3: Convert the string of dates to a list
        index_name_str_list = index_name_str.split(", ")
        question = question.strip("()")

        answer_list = ""
        for index_name_str in index_name_str_list:
            # vector store of chat history.
            index_name_str = index_name_str.strip("''") # name : 2023-08-23_cidc598a120.txt
            if index_name_str.endswith(".txt"):
                index_name_str = index_name_str[:-4]
            index, cid = index_name_str.split("_")
            loaded_vectorstore = load_embed(index_name_str)
            retriever = loaded_vectorstore.as_retriever(search_type="mmr")

            input_generator = RunnablePassthrough.assign() | CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

            _inputs = RunnableMap(
                standalone_question=input_generator,
                chat_id = input_generator,
            )

            _context = {
                "context": itemgetter("standalone_question") | retriever | _combine_documents,
                "question": lambda x: x["standalone_question"],
                "chat_id": lambda x: x["chat_id"],
            }

            conversational_qa_chain = (
                    _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()
            )

            chain = conversational_qa_chain.with_types(input_type=ChatHistory)
            answer = chain.invoke({f"question": {question}, f"chat_id": {cid}})
            answer_list += answer + "\n"
        return answer_list
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return (
        message
    )