import os
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def rag_pipeline(query, folder="data/pdfs"):

    # =========================
    # 1. LOAD PDF TEXT
    # =========================
    text = ""

    if not os.path.exists(folder):
        return "PDF folder not found."

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    if not text.strip():
        return "No text extracted from PDFs."

    # =========================
    # 2. SPLIT TEXT
    # =========================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150)

    docs = splitter.create_documents([text])

    # =========================
    # 3. EMBEDDINGS + VECTOR DB
    # =========================
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 2})

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
    [d.page_content[:500] for d in retrieved_docs[:3]] )   # =========================
    # 4. LLM RESPONSE
    # =========================
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a senior Chemical Engineering researcher.

Analyze CO2 capture scientific documents.

====================
CONTEXT
====================
{context}

QUESTION:
{query}

====================
TASK
====================

Provide:

INTRODUCTION:
METHODS:
RESULTS:
CONCLUSION:
LIMITATIONS:
ENGINEERING_INSIGHT:
RECOMMENDED_NEXT_EXPERIMENT:
"""

    response = llm.invoke(prompt)

    return response.content