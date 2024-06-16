import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_openai import OpenAIEmbeddings
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
chat_db_path = "./app/data/sample_text.txt"
vectorstore = vector_store(chat_db_path)
retriever = vectorstore.as_retriever(search_type="mmr")

# Add the chain you want to add
add_routes(app, retriever)

# test
print("TEST")
print(retriever.invoke("who are marrying?"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# curl -X POST http://localhost:8000/invoke -H "Content-Type: application/json" -d '{"key1": "value1", "key2": "value2"}'