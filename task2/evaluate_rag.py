import os
from dotenv import load_dotenv
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall
from task2.build_rag_pipeline import load_model, load_index, search_quotes

# Load environment variables from .env file
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

# Load dataset
dataset = load_dataset("Abirate/english_quotes")
test_data = dataset["train"].select(range(100))  # Use first 100 examples for testing

# Add ground truth for example queries
test_queries = [
    "What are some quotes about success?",
    "Tell me about love and relationships",
    "Share wisdom about life"
]

# Simplified ground truth for demonstration
ground_truth = [
    "Quotes about achieving success and overcoming challenges",
    "Quotes about love, relationships, and human connection",
    "Quotes about life's meaning, purpose, and wisdom"
]

# Load model and index
model = load_model()
index = load_index()

# Prepare evaluation data
eval_data = []
for query, truth in zip(test_queries, ground_truth):
    # Get relevant quotes
    relevant_quotes = search_quotes(query, model, index, test_data, k=3)
    
    # Create evaluation example
    eval_data.append({
        "question": query,
        "ground_truth": truth,
        "context": [quote["quote"] for quote in relevant_quotes]
    })

# Run evaluation
result = evaluate(
    eval_data,
    metrics=[ContextPrecision()]
)

print("\nEvaluation Results:")
print(result)

# --- RAG Evaluation --- #

# Define a small test dataset of queries (from task description examples)
# In a real scenario, you would have a more comprehensive test set
test_queries_data = [
    {
        "question": "Quotes about insanity attributed to Einstein",
        "ground_truth": "Relevant quotes discuss Einstein's views on insanity or relate to his attributed quotes on this topic."
    },
    {
        "question": "Motivational quotes tagged 'accomplishment'",
        "ground_truth": "Find quotes that are tagged with 'accomplishment' and are inspiring or encouraging."
    },
    {
        "question": "All Oscar Wilde quotes with humor",
        "ground_truth": "Retrieve quotes by Oscar Wilde that have a humorous tone or are tagged with 'humor'."
    },
]

# Perform retrieval for each query
data = []
for query_data in test_queries_data:
    query = query_data['question']
    ground_truth = query_data['ground_truth']

    # Retrieve top-k quotes (adjust k as needed for evaluation)
    retrieved_results = search_quotes(query, k=5)

    # Extract contexts (the quotes themselves)
    contexts = [result['quote'] for result in retrieved_results]

    # In a full RAGAS evaluation, you would also need:
    # - 'relevant_docs': A list of all relevant documents in the corpus for the query (requires human labeling)
    # Since we don't have this for ContextRecall, we omit it, but it's needed for a complete evaluation.

    data.append({
        'question': query,
        'contexts': contexts,
        'ground_truth': ground_truth, # Added ground truth
        # 'relevant_docs': [...], # Omitted for now as we don't have this labeled data
    })

# Convert to a HuggingFace Dataset
dataset = Dataset.from_list(data)

# Define the metrics to evaluate
# We will focus on retrieval metrics here
metrics = [
    ContextPrecision(),
    # ContextRecall(), # ContextRecall requires 'relevant_docs'
    # Add other metrics like Faithfulness, AnswerRelevance if you have ground_truth and an LLM
]

# Evaluate the dataset
print("\nStarting RAG evaluation...")
# Note: RAGAS metrics that require judging relevance (like ContextPrecision) typically need an LLM.
# Ensure you have an LLM configured (e.g., via environment variables for OpenAI or a local setup).

# You might need to configure an LLM provider for RAGAS, e.g.:
# from ragas.llms import OpenAI
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# openai_llm = OpenAI()
# evaluate(dataset, metrics=metrics, llm=openai_llm)

# For demonstration without a specific LLM configured, we will run a simplified evaluation
# This might result in errors or warnings if LLM-dependent metrics are included without configuration.

# To run ContextPrecision, RAGAS needs an LLM to judge if the retrieved context supports the ground truth.
# Ensure you have an LLM provider configured for RAGAS (e.g., via environment variables for OpenAI).
# Example for OpenAI:
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# from ragas.llms import OpenAI
# llm = OpenAI()
# results = evaluate(dataset, metrics=[ContextPrecision()], llm=llm)

# Running with default LLM configuration (might pick up environment variables if set)
try:
    print("Attempting to run evaluation with ContextPrecision...")
    results = evaluate(dataset, metrics=[ContextPrecision()])
    print("Evaluation completed.")
    print("\nEvaluation Results:")
    print(results)

except Exception as e:
    print(f"\nError during RAG evaluation: {e}")
    print("Please ensure that any necessary LLM provider is configured for RAGAS.")

# Note: For a complete evaluation including ContextRecall, you would need a test set with 'relevant_docs'.
# For Faithfulness and AnswerRelevance, 'ground_truth' and an LLM to judge the generated answer are needed. 