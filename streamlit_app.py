import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM # Using LangChain's Ollama integration

@st.cache_resource(show_spinner=False)
def load_faiss_index():
    index = faiss.read_index("faiss_index.bin")
    return index

@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource(show_spinner=False)
def load_llama_model():
    llm = OllamaLLM(model="llama2", max_tokens=150)
    return llm

def retrieve_context(query, index, embedder, top_k=3):
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return indices[0] 

def generate_answer(query, context_text, llm):
    prompt = (
        f"Answer the following question based on the context below:\n\n"
        f"Context: {context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    # Call the Llama 2 model via Ollama using LangChain
    answer = llm(prompt)
    return answer

def main():
    st.title("AI Research Assistant - RAG")
    st.write("Ask a question about your research papers on machine learning, deep learning, LLMs, etc.")

    query = st.text_input("Enter your question:")

    if st.button("Get Answer") and query:
        st.info("Loading models and index...")
        index = load_faiss_index()
        embedder = load_sentence_transformer()
        llm = load_llama_model()

        st.info("Retrieving relevant context from documents...")
        indices = retrieve_context(query, index, embedder, top_k=3)
        
        if os.path.exists("chunks.txt"):
            with open("chunks.txt", "r", encoding="utf-8") as f:
                chunks = f.read().splitlines()
            context_chunks = [chunks[i] for i in indices if i < len(chunks)]
            context_text = "\n".join(context_chunks)
        else:
            st.error("Error: 'chunks.txt' not found. Please ensure your index-building script saved the chunks.")
            return

        st.info("Generating answer using Llama 2 via Ollama...")
        answer = generate_answer(query, context_text, llm)

        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
