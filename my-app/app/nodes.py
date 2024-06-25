import os
from datetime import datetime
import functools

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from langgraph.prebuilt import ToolNode
from app.llms import llm
from app.tools import emb_finder


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


# Date Node
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
prompt = prompt.partial(system_message=f"The current date is {current_date}. If the question does not have any clue about date, use the current date. If the question has a clue about date, find all the related dates from the list {file_list}. For example, If current date is '2023-08-23' and question indicating this year, you have to answer with all the date list with 2023, like [2023-01-15_cidc598c830.txt, 2023-02-11_cidc523c830.txt, 2023-05-19_cidc598c8ab.txt ...]")
data_agent = prompt | llm

date_node = functools.partial(agent_node, agent=data_agent, name="date_finder")

# Tool Node
tools = [emb_finder]
tool_node = ToolNode(tools)


# Embedding Node
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
embedding_retriever_agent =  prompt | llm.bind_tools(tools)

embedding_retrieve_node = functools.partial(agent_node, agent=embedding_retriever_agent, name="embedding_retriever")
