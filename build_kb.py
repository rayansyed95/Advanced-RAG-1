import os
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


# -------------
# Configuration
# -------------

PDF_PATH = "data/Rag-database.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Helper Function for PDF Extraction

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
    
# Helper Function for Text Chunking

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []   
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Main Program

def main():
    print("Loading PDF.....")
    text = extract_text_from_pdf(PDF_PATH)

    print("Now chunking text ....")
    chunks = chunk_text(text)
    print("Generating Embeddings...")
    embeddings = EMBEDDING_MODEL.encode(chunks).tolist()

    print("Initializing Chroma DB")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to avoid duplicates
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Adding chunks to collection...")
    
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[{"chunk_id":i}]
        )
    print("Done!")


    print(f"Successfully created knowledge base with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
   
