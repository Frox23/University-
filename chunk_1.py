import pandas as pd
import numpy as np
import re
import pickle
from sentence_transformers import SentenceTransformer
import faiss

def create_medicine_index():
    with open('Medicines(1).txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    chunks = re.split(r'\n(?=(?:Name:|MEDICINE NAME:))', text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk) > 50]
    
    print(f"Created {len(chunks)} chunks")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    faiss.write_index(index, "medicine_index.faiss")
    with open("medicine_index.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"✅ Index created with {len(chunks)} entries")
    print(f"Embedding dimension: {dimension}")
    
    return chunks

if __name__ == "__main__":
    chunks = create_medicine_index()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("medicine_index.faiss")
    
    test_query = "What are the side effects of Advil?"
    query_embedding = model.encode([test_query])
    D, I = index.search(np.array(query_embedding), k=3)
    
    print(f"\n🔍 Test Query: {test_query}")
    print("Top 3 matches:")
    for i, idx in enumerate(I[0]):
        print(f"\n{i+1}. Distance: {D[0][i]:.4f}")
        print(f"Text: {chunks[idx][:200]}...")