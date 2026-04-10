import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Updated to the non-deprecated version

load_dotenv()


def get_loader(file_path):
    """Factory to return the correct loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    if ext == ".csv":
        return CSVLoader(file_path)
    if ext == ".html":
        return UnstructuredHTMLLoader(file_path)
    return TextLoader(file_path)


def ingest_data():
    data_dir = "data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("📁 Created /data directory. Add your files there!")
        return

    raw_documents = []
    print("📂 Loading files from /data...")

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            try:
                loader = get_loader(file_path)
                raw_documents.extend(loader.load())
                print(f"✔️ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

    if not raw_documents:
        print("⚠️ No documents found. Ingestion stopped.")
        return

    # 1. First Pass: Markdown Header Splitting
    # Even if the file isn't MD, this prepares the structure if headers exist
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Combine text and split by headers
    full_text = "\n\n".join([doc.page_content for doc in raw_documents])
    md_header_splits = md_splitter.split_text(full_text)

    # 2. Second Pass: Recursive Character Splitting (The 'safety net')
    # chunk_size is 512 because BAAI/bge-small-en-v1.5 has a 512-token limit
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, add_start_index=True, strip_whitespace=True
    )
    final_chunks = recursive_splitter.split_documents(md_header_splits)

    # 3. Embedding Model (BAAI)
    print("🧠 Initializing BAAI Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 4. Vector Store (ChromaDB)
    print("💾 Saving to ChromaDB...")
    # This will overwrite/update the existing DB in chroma_db folder
    Chroma.from_documents(
        documents=final_chunks, embedding=embeddings, persist_directory="./chroma_db"
    )

    print(f"✅ Successfully ingested {len(final_chunks)} chunks.")


if __name__ == "__main__":
    ingest_data()
