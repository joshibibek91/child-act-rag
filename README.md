# Child Act Assistant — a simple RAG

Ask questions about Nepal's **Act Relating to Children, 2075 (2018)** and get
answers from the Act itself, with the pages they came from.

| File | What it is |
|---|---|
| `rag_terminal.py` | The RAG system in the terminal |
| `rag_streamlit.py` | The same RAG system as a web page — upload the PDF, then ask |
| `child_act_2075.pdf` | The Act (English translation, Nepal Law Commission) |

Both files do the same 6 steps: **load → chunk → embed → store → retrieve → generate.**

## Setup

1. Get a free Gemini API key from Google AI Studio.
2. In this folder:

        python -m venv .venv
        .venv\Scripts\activate          (Windows)
        source .venv/bin/activate       (Mac / Linux)
        pip install -r requirements.txt

3. Copy `.env.example` to `.env` and paste your key.

## Run

    python rag_terminal.py
    streamlit run rag_streamlit.py

## Try these questions

- At what age is a person no longer a child?
- Can the police handcuff a child?
- Within how many days must the Juvenile Court decide a case?
- What is the capital of Australia?  *(it should say it could not find it)*

The answers are information from the Act, not legal advice.
