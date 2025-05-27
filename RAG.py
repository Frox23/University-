import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import requests
import os

def get_rag_answer(query):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not os.path.exists('medicine_index.faiss') or not os.path.exists('medicine_index.pkl'):
            return "⚠️ Medicine database not found. Please run the indexing script first."
        
        index = faiss.read_index('medicine_index.faiss')
        with open('medicine_index.pkl', 'rb') as f:
            chunks = pickle.load(f)
        
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)
        
        retrieved_chunks = []
        for idx in I[0]:
            if idx < len(chunks):  
                retrieved_chunks.append(chunks[idx])
        
        if not retrieved_chunks:
            return "⚠️ No relevant information found in the database."
        
        retrieved_text = "\n\n".join(retrieved_chunks)
        
        prompt = f"""You are a helpful medical assistant. Use ONLY the following information to answer the user's question accurately and clearly. If the information doesn't contain the answer, say so.

MEDICAL INFORMATION:
{retrieved_text}

QUESTION: {query}

ANSWER:"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "⚠️ No answer generated.")
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ LLM API failed: {e}")
            return f"Based on the available information:\n\n{retrieved_text[:1000]}..."

    except Exception as e:
        print(f"⚠️ RAG Error: {e}")
        return "⚠️ Sorry, there was an error processing your query."

def update_index_with_new_text(new_text, model=None):
    try:
        if model is None:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if os.path.exists('medicine_index.faiss') and os.path.exists('medicine_index.pkl'):
            index = faiss.read_index('medicine_index.faiss')
            with open('medicine_index.pkl', 'rb') as f:
                existing_chunks = pickle.load(f)
        else:
            index = faiss.IndexFlatL2(384)
            existing_chunks = []
        
        new_chunks = [chunk.strip() for chunk in new_text.split("\n\n") if chunk.strip() and len(chunk) > 20]
        
        if new_chunks:
            new_embeddings = model.encode(new_chunks)
            
            index.add(new_embeddings)
            existing_chunks.extend(new_chunks)
            
            faiss.write_index(index, 'medicine_index.faiss')
            with open('medicine_index.pkl', 'wb') as f:
                pickle.dump(existing_chunks, f)
            
            print(f"✅ Index updated with {len(new_chunks)} new chunks")
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Index update failed: {e}")
        return False