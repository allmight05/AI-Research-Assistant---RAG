import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from extract_text import extract_text_from_pdf 

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_documents(pdf_folder):
    all_chunks = []
    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            print(f"Processing: {pdf_path}")
            raw_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)
    return all_chunks

def build_faiss_index(embeddings):

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    pdf_folder = os.path.join("data")
    
    print("Extracting and processing documents...")
    all_chunks = process_documents(pdf_folder)
    print(f"Total chunks created: {len(all_chunks)}")
    
    with open("chunks.txt", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n")
    print("Text chunks saved to 'chunks.txt'.")
    
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings for text chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embedding_array = np.array(embeddings).astype('float32')
    
    print("Building FAISS index...")
    index = build_faiss_index(embedding_array)
    print(f"Total vectors in FAISS index: {index.ntotal}")
    
    faiss.write_index(index, "faiss_index.bin")
    print("FAISS index saved as 'faiss_index.bin'.")
