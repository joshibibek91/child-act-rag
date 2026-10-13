# Simple RAG as a web page — upload the Child Act PDF and ask questions
# Run:  streamlit run rag_streamlit.py
#
# The same 6 RAG steps as rag_terminal.py:
#   1. LOAD  2. CHUNK  3. EMBED  4. STORE  5. RETRIEVE  6. GENERATE
# Only the screen is different: Streamlit turns this script into a web page.

import os
import re
import time

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()                                   # reads GEMINI_API_KEY from .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"            # text -> numbers
CHAT_MODEL = "gemini-3.5-flash"                 # writes the answer

# SYSTEM PROMPT: the model's fixed rules. It is the same for every question.
SYSTEM_PROMPT = """You are a legal information assistant for a law office in Nepal.
You answer questions about the Act Relating to Children, 2075 (2018).

Rules:
- Answer ONLY from the text of the Act given with the question. Never use outside knowledge.
- Mention the section number and page you used, for example: (Section 42, page 20).
- If the answer is not in the given text, say: "I could not find this in the Act." Never guess.
- Answer in simple, clear language, in the same language as the question (English or Nepali).
- End every answer with: "This is information from the Act, not legal advice."
"""


def embed(texts):
    """EMBEDDING: a list of numbers that describes what a text means."""
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


# st.cache_resource: do steps 1-4 ONCE per uploaded file, not on every click.
@st.cache_resource
def build_database(pdf_file):
    # 1. LOAD
    pages = [p.extract_text(extraction_mode="layout") for p in PdfReader(pdf_file).pages]
    pages = [re.sub(r" +", " ", text) for text in pages]   # squeeze extra spaces

    # 2. CHUNK (recursive: paragraphs, then lines, then sentences, then words)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, page_numbers = [], []
    for number, text in enumerate(pages, start=1):
        for piece in splitter.split_text(text):
            if len(piece) < 50:                     # skip page numbers and headers
                continue
            chunks.append(piece)
            page_numbers.append(number)

    # 3. EMBED + 4. STORE in ChromaDB (a vector database, in memory)
    db = chromadb.Client()
    name = f"pdf-{len(db.list_collections()) + 1}"     # a new collection for each file
    collection = db.create_collection(name, metadata={"hnsw:space": "cosine"})
    # The FREE Gemini key allows 100 embeddings a minute (each chunk counts as one).
    # So we send 90 chunks, wait a minute, then send the next 90.
    for i in range(0, len(chunks), 90):
        if i > 0:
            time.sleep(60)
        batch = chunks[i:i + 90]
        collection.add(
            ids=[f"chunk-{n}" for n in range(i, i + len(batch))],
            documents=batch,
            embeddings=embed(batch),
            metadatas=[{"page": p} for p in page_numbers[i:i + 90]],
        )
    return collection


def ask(collection, question):
    # 5. RETRIEVE: the 4 chunks closest in meaning to the question
    found = collection.query(query_embeddings=embed([question]), n_results=4)
    pieces = found["documents"][0]
    pages_used = [m["page"] for m in found["metadatas"][0]]

    # 6. GENERATE: Gemini answers from those chunks only
    context = "\n\n".join(f"[Page {p}]\n{text}" for p, text in zip(pages_used, pieces))
    # USER PROMPT: changes every time — the retrieved text and the question.
    user_prompt = f"""TEXT FROM THE ACT:
{context}

QUESTION: {question}"""
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text, pages_used, pieces


# ---------------- The web page ----------------
st.title("⚖️ Child Act Assistant")

pdf = st.file_uploader("Upload the Child Act (PDF)", type="pdf")
if pdf is None:
    st.info("Upload the PDF to start.")
    st.stop()                                   # wait until a file is uploaded

with st.spinner("Reading the PDF and building the database (about 1 minute)..."):
    collection = build_database(pdf)
st.success(f"Ready: {collection.count()} chunks stored.")

question = st.text_input("Your question")
if st.button("Ask") and question:
    with st.spinner("Searching the Act..."):
        answer, pages_used, pieces = ask(collection, question)
    st.write(answer)
    st.caption(f"Pages used: {pages_used}")
    with st.expander("See the chunks that were retrieved"):
        for p, text in zip(pages_used, pieces):
            st.markdown(f"**Page {p}**")
            st.text(text)
