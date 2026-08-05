import os
import streamlit as st
import PyPDF2
import docx

from dotenv import load_dotenv
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI


# CONFIG
# ============================================
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
base_url = os.getenv("OPENROUTER_BASE_URL")

client = OpenAI(
    base_url=base_url,
    api_key=api_key)


# STREAMLIT SETUP
# ============================================
st.set_page_config(page_title="ChemInsight AI", layout="wide")
st.title("🔬 ChemInsight AI")


st.markdown("""
## ChemInsight AI: Scientific Research Assistant

AI-powered scientific document analysis platform using:

- Retrieval Augmented Generation (RAG)
- FAISS Vector Database
- HuggingFace Embeddings
- LLM GPT-4o Mini (OpenRouter)

Capabilities:

- PDF/DOCX/TXT analysis
- Scientific summarization
- Technical interpretation
- Document-based question answering
""")


# Session memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# FILE READER
# ============================================
def read_file(file):

    text = []

    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)

    elif file.name.endswith(".txt"):
        text.append(str(file.read(), "utf-8"))

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text.extend([p.text for p in doc.paragraphs if p.text])

    return "\n\n".join(text)


# CHUNKING
# ============================================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100)
    return splitter.create_documents([text])


# RETRIEVER
# ============================================
def build_retriever(text):

    docs = split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# QA CHAIN
# ============================================
def build_qa_chain(retriever):

    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        base_url=base_url,
        temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=False)


# ANALYSIS FUNCTIONS
# ============================================
def structured_summary(text):

    text = safe_text(text)

    prompt = f"""
Write a structured 500–600 word scientific summary.

Rules:
- be concise
- no hallucination
- if missing say "Not mentioned"

TEXT:
{text}
"""

    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400)

    return res.choices[0].message.content


def deep_analysis(text):

    text = safe_text(text)
    res = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": "Deep scientific analysis"},
            {"role": "user", "content": text}],
        max_tokens=400)
    return res.choices[0].message.content


# FILE UPLOAD
# ============================================
uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True)


# MAIN PIPELINE
# ============================================
if uploaded_files:

    all_text = ""

    for f in uploaded_files:
        all_text += read_file(f) + "\n\n"

    retriever = build_retriever(all_text)
    qa_chain = build_qa_chain(retriever)


    # ANALYSIS
    # =========================
    if st.button("Run Analysis"):

        with st.spinner("Analyzing..."):

            summary = structured_summary(all_text)
            analysis = deep_analysis(all_text)

        tab1, tab2 = st.tabs([
            "Summary",
            "Deep Analysis" ])

        with tab1:
            st.write(summary)

        with tab2:
            st.write(analysis)


