from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from .config import EMBEDDING_MODEL


def build_or_load_vectorstore(chunks: List[Document], persist_dir: str = "vectorstore"):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if chunks:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
        try:
            vs.persist()
        except Exception:
            pass
        return vs
    # Fallback: load existing
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)


def retrieve_context(vs, topic: str, k: int = 6) -> str:
    results = vs.similarity_search(topic, k=k)
    return "\n\n".join([r.page_content for r in results])