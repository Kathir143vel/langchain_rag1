# LangChain RAG Pipeline

A PEP 8 compliant RAG pipeline using Groq and BAAI embeddings.

## Setup
1. `python -m venv venv`
2. `source venv/Scripts/activate`
3. `pip install -r requirements.txt`
4. Create a `.env` file with `GROQ_API_KEY`.

## Usage
- Put files in `/data`.
- Run `python src/ingest.py` to index data.
- Run `python src/main.py` to chat.
