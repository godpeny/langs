import os
from datetime import datetime
import functools
from operator import itemgetter
import operator
from typing import Annotated, Sequence, TypedDict, Literal, List, Tuple
import json
from typing import List, Dict, Any, Union
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import RedirectResponse

from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document,MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from langserve import add_routes, APIHandler
from langserve.pydantic_v1 import BaseModel, Field

"""
    Define LLMs
"""
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])


def vector_store(path):
    raw_documents = TextLoader(path).load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    store = FAISS.from_documents(documents, emb)

    return store


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


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


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

        print("question : ", question)
        for date in date_list:
            # vector store of chat history.
            date = date.strip("''")
            print("date : ", date)
            chat_db_path = f"./app/data/{date}"
            vectorstore = vector_store(chat_db_path)
            retriever = vectorstore.as_retriever(search_type="mmr")

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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


"""
Data Nodes
"""
def agent_node(state, agent, name):
    """Helper function to create a node for a given agent"""
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


# Date Agent
current_date = datetime.now().strftime("%Y-%m-%d.txt")
file_list = os.listdir('./app/data')
prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You do not need to answer question itself."
                "You have two goals, one is to find date information from question and other is to pass the question for the next agent to answer."
                "Make '[date1.txt, date2.txt, ...]:::(question)' format to answer, when (question) is the original question from the user."
                "\n{system_message}"
                ,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
prompt = prompt.partial(system_message=f"The current date is {current_date}. If the question does not have any clue about date, use the current date. If the question has a clue about date, find all the related dates from the list {file_list}. For example, If current date is '2023-08-23' and question indicating this year, you have to answer with all the date list with 2023, like [2023-01-15.txt, 2023-02-11.txt, 2023-05-19.txt ...]")
data_agent =  prompt | llm

date_node = functools.partial(agent_node, agent=data_agent, name="date_finder")

"""
Tool Nodes
"""
tools = [emb_finder]
tool_node = ToolNode(tools)


"""
Embedding Node
"""
prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You have to use tools to find the embedding with the proper date,"
                "Remember, you must use tool with the answer received from previous agent without any editing or deleting with the format of '[date1, date2, ...]:::(question)'."
                "Remember, don't miss or ignore single date element from received answer when you pass the state to tools."
                "You have access to the following tools: {tool_names}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
chart_agent =  prompt | llm.bind_tools(tools)

chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")


"""
    Router
"""
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """
        This is the router
        Either agent can decide to end
    """
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    return "continue"


""" 
    Graph
"""
workflow = StateGraph(AgentState)

workflow.add_node("date_finder", date_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "date_finder",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": END, "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: "continue",
    {
        "continue": END,
    },
)

workflow.set_entry_point("date_finder")
graph = workflow.compile()


def extract_content_and_urls(value: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    result = []
    possible_keys = ['call_tool', 'date_finder', 'chart_generator']

    for key in possible_keys:
        if key in value:
            data = value[key]
            if 'messages' in data:
                messages = data['messages']
                if isinstance(messages, list) and len(messages) > 0:
                    message = messages[0]
                    content = message.content
                    # Check if the content is a JSON string
                    try:
                        json_content = json.loads(content)
                        # Handle case where content is a JSON string
                        for item in json_content:
                            url = item.get('url')
                            content = item.get('content')
                            result.append({'url': url, 'content': content})
                    except json.JSONDecodeError:
                        # Handle case where content is a regular string
                        result.append({'content': content})
            break  # Stop after finding the first valid key
    return result

"""
    Fast API with langserve
"""
app = FastAPI(
    title="Langs",
    version="0.0.0",
    description="Server for Lang-* components combined.",
)

api_handler = APIHandler(graph, path="/api/v1")


@app.post("/api/v1/invoke", include_in_schema=False)
def simple_invoke(request: Request) -> Response:
    """Handle a request."""
    # The API Handler validates the parts of the request
    # that are used by the runnnable (e.g., input, config fields)
    print("!")
    print(request.body)
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="what topic wre they talking about yesterday?"
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 5},
    )

    last_event = None
    for event in events:
        last_event = event

    return extract_content_and_urls(last_event)


# test
# print("TEST")
# print(chain.invoke({"input": {"question": "what do you know about harrison", "chat_history": []}}))
# print(conversational_qa_chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
