from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.llms import emb


def embed(name):
    raw_documents = TextLoader(f"./app/data/{name}").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    store = FAISS.from_documents(documents, emb)

    return store


def save_embed(name, store):
    # remove '.txt' if exists
    if name.endswith(".txt"):
        name = name[:-4]
    store.save_local(f"app/data/embeddings/${name}.faiss")
    return store


def load_embed(name):
    if name.endswith(".txt"):
        name = name[:-4]
    store = FAISS.load_local(f"app/data/embeddings/${name}.faiss", emb, allow_dangerous_deserialization=True)
    return store
