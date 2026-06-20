# rag.py

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

TOP_K = 2

QUERY = "Who won the 2023 Cricket World Cup?"


# --- LOAD ENV ---
load_dotenv()


# --- CONNECT TO PINECONE ---
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

print("Loading Pinecone index...")
index = pc.Index("cwc-index")

NAMESPACE = "default"

# --- EMBEDDINGS ---
print("Creating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# --- VECTOR STORE ---
print("Connecting LangChain to Pinecone vector store...")
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE,
)


# --- RETRIEVE ---
print("Retrieving documents...")
docs = vector_store.similarity_search(
    QUERY,
    k=TOP_K,
)

contexts = [doc.page_content for doc in docs]


# --- PROMPT ---
context_text = "\n\n".join(contexts)

prompt = f"""
Answer the question based only on the context below.

Question:
{QUERY}

Context:
{context_text}

If the answer is not in the context, say: I don't know.
"""


# --- LLM ---
print("Generating answer...")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

response = llm.invoke([("human", prompt)])


print("\n=== ANSWER ===\n")
print(response.content)


print("\n=== SOURCES USED ===\n")
for i, doc in enumerate(docs, start=1):
    print(f"\n--- Source {i} ---")
    print(doc.page_content[:500])
