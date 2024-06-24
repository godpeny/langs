import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

"""
    Define LLMs
"""
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])