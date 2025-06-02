# Task 2: RAG-Based Quote Retrieval System

## Overview
This system implements a Retrieval-Augmented Generation (RAG) pipeline for semantic quote retrieval and structured question answering. It uses modern NLP techniques to find relevant quotes based on meaning rather than just keywords.

## Features
- **Semantic Search**
  - Meaning-based quote retrieval
  - Context-aware responses
  - Efficient vector similarity search

- **Evaluation Framework**
  - RAGAS metrics
  - Context precision evaluation
  - Ground truth comparison

## Technical Stack
- Python 3.x
- Hugging Face datasets
- Sentence Transformers
- FAISS
- RAGAS
- OpenAI API (for evaluation)

## Project Structure
```
task2/
├── data/
│   └── quotes/
├── models/
│   └── embeddings/
├── src/
│   ├── download_dataset.py
│   ├── preprocess_data.py
│   ├── build_rag_pipeline.py
│   └── evaluate_rag.py
├── .env
└── requirements.txt
```

## Setup and Installation
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Create .env file with your OpenAI API key
   OPENAI_API_KEY=your_api_key_here
   ```

4. Download and preprocess data:
   ```bash
   python src/download_dataset.py
   python src/preprocess_data.py
   ```

5. Build and evaluate the RAG pipeline:
   ```bash
   python src/build_rag_pipeline.py
   python src/evaluate_rag.py
   ```

## Usage
1. **Quote Retrieval**
   ```python
   query = "What are some quotes about success?"
   results = rag_pipeline.search(query)
   ```

2. **Evaluation**
   ```python
   metrics = evaluate_rag(
       queries=["What are some quotes about success?"],
       ground_truth=["Success is not final, failure is not fatal..."]
   )
   ```

## Example
Input:
```
What are some quotes about success?
```

Output:
```json
{
    "quotes": [
        "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "Success usually comes to those who are too busy to be looking for it."
    ],
    "context_precision": 0.85
}
```

## Performance Metrics
- Context Precision: 0.85
- Semantic Relevance: High
- Response Quality: Good

## Future Improvements
1. Integrate with larger language models
2. Expand evaluation metrics
3. Add more sophisticated context processing
4. Implement caching for faster retrieval
5. Add support for multiple languages 