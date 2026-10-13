# Simple RAG in the terminal — ask questions about the Child Act PDF
# Run:  python rag_terminal.py
#
# RAG = Retrieval-Augmented Generation
#   1. LOAD      read the text from the PDF
#   2. CHUNK     cut the text into small pieces
#   3. EMBED     turn each piece into numbers that capture its meaning
#   4. STORE     keep the pieces and their numbers in ChromaDB
#   5. RETRIEVE  find the pieces closest in meaning to the question
#   6. GENERATE  Gemini answers using only those pieces

import os
import re
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()                                   # reads GEMINI_API_KEY from .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_FILE = "child_act_2075.pdf"
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
    """EMBEDDING: a list of numbers that describes what a text means.
    Texts with similar meaning get similar numbers."""
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


# ---------- 1. LOAD: read the text of every page ----------
reader = PdfReader(PDF_FILE)
pages = [page.extract_text(extraction_mode="layout") for page in reader.pages]
pages = [re.sub(r" +", " ", text) for text in pages]       # squeeze extra spaces
print(f"1. Loaded {len(pages)} pages")

# ---------- 2. CHUNK: recursive splitting ----------
# It tries to cut at paragraphs first, then lines, then sentences, then words,
# so a chunk rarely ends in the middle of a sentence.
# chunk_overlap repeats a little text between chunks, so nothing is lost at a cut.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks, page_numbers = [], []
for number, text in enumerate(pages, start=1):
    for piece in splitter.split_text(text):
        if len(piece) < 50:                     # skip page numbers and headers
            continue
        chunks.append(piece)
        page_numbers.append(number)             # remember where it came from
print(f"2. Cut into {len(chunks)} chunks")

# ---------- 3. EMBED + 4. STORE in ChromaDB ----------
# ChromaDB is a VECTOR DATABASE: it stores the numbers and finds the closest ones.
db = chromadb.Client()                          # lives in memory while the program runs
collection = db.create_collection("child_act", metadata={"hnsw:space": "cosine"})
# The FREE Gemini key allows 100 embeddings a minute (each chunk counts as one).
# So we send 90 chunks, wait a minute, then send the next 90.
for i in range(0, len(chunks), 90):
    if i > 0:
        print("   Waiting 60 seconds for the free-tier limit...")
        time.sleep(60)
    batch = chunks[i:i + 90]
    collection.add(
        ids=[f"chunk-{n}" for n in range(i, i + len(batch))],
        documents=batch,
        embeddings=embed(batch),
        metadatas=[{"page": p} for p in page_numbers[i:i + 90]],
    )
print(f"3-4. Embedded and stored {collection.count()} chunks\n")


def ask(question):
    # ---------- 5. RETRIEVE: the 4 chunks closest to the question ----------
    found = collection.query(query_embeddings=embed([question]), n_results=4)
    pieces = found["documents"][0]
    pages_used = [m["page"] for m in found["metadatas"][0]]

    # ---------- 6. GENERATE: answer from those chunks only ----------
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
    return response.text, pages_used


# ---------- Ask questions until the user types quit ----------
print("Ask about Nepal's Act Relating to Children, 2075. Type quit to stop.")
while True:
    question = input("\nQuestion: ").strip()
    if question.lower() == "quit":
        break
    if question:
        answer, pages_used = ask(question)
        print(f"\nAnswer: {answer}")
        print(f"(Pages used: {pages_used})")
