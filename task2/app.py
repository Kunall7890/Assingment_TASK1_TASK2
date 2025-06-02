import gradio as gr
import pandas as pd
from build_rag_pipeline import load_model, load_index, search_quotes

# Load the model and index
model = load_model()
index = load_index()
quotes_df = pd.read_csv("task2/data/quotes.csv")

def search_quotes_interface(query: str, num_results: int = 5):
    """Interface function for searching quotes."""
    results = search_quotes(query, model, index, quotes_df, k=num_results)
    
    # Format results for display
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"{i}. Quote: {result['quote']}\n"
            f"   Author: {result['author']}\n"
            f"   Tags: {result['tags']}\n"
            f"   Relevance Score: {result['score']:.2f}\n"
        )
    
    return "\n\n".join(formatted_results)

# Create the Gradio interface
interface = gr.Interface(
    fn=search_quotes_interface,
    inputs=[
        gr.Textbox(label="Enter your search query", placeholder="e.g., quotes about success"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of results")
    ],
    outputs=gr.Textbox(label="Search Results", lines=10),
    title="Quote Search System",
    description="Search through a collection of quotes using natural language queries.",
    examples=[
        ["quotes about success and achievement"],
        ["motivational quotes about perseverance"],
        ["famous quotes about love"]
    ]
)

if __name__ == "__main__":
    interface.launch() 