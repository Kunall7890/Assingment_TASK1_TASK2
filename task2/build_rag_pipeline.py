from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import os
import re
import torch
import pandas as pd
from typing import List, Dict, Any

# Define the directory for Task 2
task2_dir = "task2"
fine_tuned_model_path = os.path.join(task2_dir, 'fine_tuned_model')

# --- RAG Pipeline Building --- #

# Load the fine-tuned model
try:
    model = SentenceTransformer(fine_tuned_model_path)
    print(f"Loaded fine-tuned model from {fine_tuned_model_path}")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    exit()

# Load and preprocess the dataset (re-using preprocessing logic)
try:
    def clean_text(text):
        if text is None:
            return ""
        text = text.lower()
        text = re.sub(r'["()\-,.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_entry(entry):
        entry['quote'] = clean_text(entry['quote'])
        # Clean author and tags as well, joining tags into a single string
        entry['author'] = clean_text(entry['author'])
        if isinstance(entry['tags'], list):
             entry['tags'] = " ".join([clean_text(tag) for tag in entry['tags']])
        elif entry['tags'] is None:
             entry['tags'] = ""
        else:
             entry['tags'] = clean_text(entry['tags'])
        return entry

    dataset = load_dataset("Abirate/english_quotes")
    processed_dataset = dataset.map(preprocess_entry)
    quotes = processed_dataset['train']['quote']
    # Store other relevant info like author and tags
    authors = processed_dataset['train']['author']
    tags = processed_dataset['train']['tags']
    print(f"Loaded and preprocessed {len(quotes)} quotes.")

except Exception as e:
    print(f"Error loading or preprocessing dataset: {e}")
    exit()

# Generate embeddings for the quotes
print("Generating embeddings for quotes...")
# Encode in batches to avoid memory issues for large datasets
embeddings = model.encode(quotes, show_progress_bar=True)
print("Embeddings generated.")

# Build FAISS index
print("Building FAISS index...")
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension) # Using L2 distance for similarity
index.add(np.array(embeddings).astype('float32'))
print(f"FAISS index built with {index.ntotal} vectors.")

# Save the FAISS index (optional but recommended)
faiss_index_path = os.path.join(task2_dir, 'faiss_index.bin')
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")

# Function to perform semantic search
def search_quotes(query, k=5):
    # Encode the query
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    # Perform search in FAISS index
    distances, indices = index.search(query_embedding, k)

    # Retrieve relevant quotes and their metadata
    results = []
    for i in range(k):
        idx = indices[0][i]
        results.append({
            'quote': quotes[idx],
            'author': authors[idx],
            'tags': tags[idx],
            'distance': distances[0][i] # Lower distance means higher similarity for L2
        })
    return results

# Function to generate a response using an LLM (Placeholder)
def generate_rag_response(query, search_results):
    """
    Generates a response to the user query based on the retrieved quotes.
    (Placeholder for LLM integration)

    Args:
        query (str): The user's natural language query.
        search_results (list): A list of dictionaries containing retrieved quote information.
    Returns:
        dict: A structured response containing relevant information.
    """
    # In a real RAG system, you would pass the query and search_results to an LLM
    # to generate a coherent and contextually relevant answer.
    # For this placeholder, we will structure the retrieved information.

    structured_response = {
        "query": query,
        "retrieved_quotes": [],
        "summary": "", # Placeholder for LLM-generated summary
        "authors": [],
        "tags": []
    }

    # Collect information from retrieved quotes
    for result in search_results:
        structured_response['retrieved_quotes'].append({
            'quote': result['quote'],
            'author': result['author'],
            'tags': result['tags']
        })
        if result['author'] not in structured_response['authors']:
            structured_response['authors'].append(result['author'])
        # Assuming tags are space-separated string, split and add unique tags
        for tag in result['tags'].split():
             if tag not in structured_response['tags']:
                 structured_response['tags'].append(tag)

    # Sort authors and tags for consistent output
    structured_response['authors'].sort()
    structured_response['tags'].sort()

    # Add a basic placeholder summary
    if search_results:
        structured_response['summary'] = f"Based on relevant quotes, here is information related to your query about '{query}'."
    else:
        structured_response['summary'] = f"No relevant quotes found for your query about '{query}'."

    return structured_response

# Example usage of the search and response generation
if __name__ == "__main__":
    example_query = "quotes about hope by Oscar Wilde"
    print(f"\nSearching for: '{example_query}'")
    search_results = search_quotes(example_query, k=3)

    print("\nSearch Results:")
    for result in search_results:
        print(f"Quote: {result['quote']}")
        print(f"Author: {result['author']}")
        print(f"Tags: {result['tags']}")
        print(f"Distance (L2): {result['distance']:.4f}")
        print("---")

    rag_response = generate_rag_response(example_query, search_results)
    print("\nGenerated RAG Response (Placeholder):")
    import json
    print(json.dumps(rag_response, indent=4))

def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the sentence transformer model."""
    return SentenceTransformer(model_name)

def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True)

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def search_quotes(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    quotes_df: pd.DataFrame,
    k: int = 5
) -> List[Dict[str, Any]]:
    """Search for relevant quotes using the query."""
    # Generate query embedding
    query_embedding = model.encode([query])
    
    # Search the index
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Get the results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        quote = quotes_df.iloc[idx]
        results.append({
            "quote": quote["quote"],
            "author": quote["author"],
            "tags": quote["tags"],
            "score": float(1 / (1 + distance))  # Convert distance to similarity score
        })
    
    return results

def load_index(index_path: str = "task2/models/faiss_index.bin") -> faiss.Index:
    """Load a saved FAISS index."""
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return None

def save_index(index: faiss.Index, index_path: str = "task2/models/faiss_index.bin"):
    """Save a FAISS index."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

def build_and_save_pipeline():
    """Build and save the complete RAG pipeline."""
    # Load data
    quotes_df = pd.read_csv("task2/data/quotes.csv")
    
    # Load model
    model = load_model()
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(model, quotes_df["quote"].tolist())
    
    # Build index
    print("Building FAISS index...")
    index = build_index(embeddings)
    
    # Save index
    print("Saving index...")
    save_index(index)
    
    print("Pipeline built and saved successfully!")

if __name__ == "__main__":
    build_and_save_pipeline() 