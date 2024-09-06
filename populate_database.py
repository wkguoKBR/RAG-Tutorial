import argparse
import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_ollama_embedding import get_embedding_function
# from get_openai_embedding import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    # Check if the database should be cleared (using the --clear flag).
    args = parse_arguments()
    if args.reset:
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    return parser.parse_args()

def load_documents(data_path: str):
    # # PyPDF
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

    # Unstructured
    # documents = []
    # for filename in os.listdir(data_path):
    #     if filename.endswith(".pdf"):
    #         file_path = os.path.join(data_path, filename)
    #         loader = UnstructuredPDFLoader(file_path)
    #         documents.extend(loader.load())  # Load and add the documents to the list
    # return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_ids = set(db.get(include=[]).get("ids", []))
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

        # Inspect the embeddings of the newly added documents.
        # inspect_embeddings(db, new_chunk_ids)
    else:
        print("✅ No new documents to add")

def inspect_embeddings(db, chunk_ids):
    # Retrieve embeddings for the given chunk IDs.
    embeddings = db.get(ids=chunk_ids, include=["embeddings"])["embeddings"]
    
    # Print a few embeddings for inspection
    for i, embedding in enumerate(embeddings[:5]):
        print(f"Embedding {i + 1}: {embedding[:5]}... (truncated)")

def calculate_chunk_ids(chunks: list[Document]):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Add  chunk ID to the page meta-data.
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        print("✨ Clearing Database")
        shutil.rmtree(CHROMA_PATH)
    else:
        print("⚠️ Database not found.")

if __name__ == "__main__":
    main()
