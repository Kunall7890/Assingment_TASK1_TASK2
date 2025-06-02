from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_from_disk
import os
import pandas as pd

# Define the directory for Task 2
task2_dir = "task2"
processed_dataset_path = os.path.join(task2_dir, 'processed_dataset')

# --- Model Fine-Tuning --- #

# Load the processed dataset
try:
    # Assuming you saved the processed dataset to disk in the previous step
    # If not, you would load and preprocess here
    # dataset = load_dataset("Abirate/english_quotes")
    # from preprocess_data import preprocess_entry # Assuming preprocess_entry is in preprocess_data.py
    # processed_dataset = dataset.map(preprocess_entry)

    # For now, let's load from the original source and re-preprocess in this script
    from datasets import load_dataset
    import re

    def clean_text(text):
        if text is None:
            return ""
        text = text.lower()
        text = re.sub(r'["()\-,.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_entry(entry):
        entry['quote'] = clean_text(entry['quote'])
        # Clean author and tags for potential use in training examples
        entry['author'] = clean_text(entry['author'])
        if isinstance(entry['tags'], list):
             entry['tags'] = " ".join([clean_text(tag) for tag in entry['tags']]) # Join tags into a single string
        elif entry['tags'] is None:
             entry['tags'] = ""
        else:
             entry['tags'] = clean_text(entry['tags']) # Handle cases where tags might not be a list initially

        return entry

    dataset = load_dataset("Abirate/english_quotes")
    processed_dataset = dataset.map(preprocess_entry)
    print("Dataset loaded and preprocessed successfully!")

except Exception as e:
    print(f"Error loading or preprocessing dataset: {e}")
    exit()

# Choose a pre-trained model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
print(f"Loaded base model: {model_name}")

# Prepare training data (Examples: Query, Relevant Text)
# We will create training examples where the query is a combination of author and tags, and the text is the quote.
# This helps the model learn to associate authors/tags with quote content.
training_examples = []
for i in range(len(processed_dataset['train'])):
    entry = processed_dataset['train'][i]
    query = f"{entry['author']} {entry['tags']}".strip()
    text = entry['quote']
    if query and text:
        # Add label=1.0 for positive pairs
        training_examples.append(InputExample(texts=[query, text], label=1.0))

print(f"Created {len(training_examples)} training examples.")

# Create DataLoader
train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)

# Define Loss Function
train_loss = losses.CosineSimilarityLoss(model)

# Define Training Arguments
num_epochs = 1
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of training data for warmup

# Fine-tune the model
print("\nStarting model fine-tuning...")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=os.path.join(task2_dir, 'fine_tuned_model'))

print("\nModel fine-tuning completed.")
print(f"Fine-tuned model saved to {os.path.join(task2_dir, 'fine_tuned_model')}") 