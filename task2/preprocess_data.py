from datasets import load_dataset, Dataset
import re
import os

# Define the directory for Task 2
task2_dir = "task2"

# Define the dataset name
dataset_name = "Abirate/english_quotes"

# Load the dataset (assuming it's already downloaded)
try:
    dataset = load_dataset(dataset_name)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Data Preprocessing --- #

def clean_text(text):
    if text is None:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove some common unwanted characters (optional, can be refined)
    text = re.sub(r'["()\-,.]', '', text) # remove quotes, parentheses, comma, hyphen, dot
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_entry(entry):
    entry['quote'] = clean_text(entry['quote'])
    # You might want to clean author and tags as well, depending on usage
    # entry['author'] = clean_text(entry['author'])
    # if isinstance(entry['tags'], list):
    #     entry['tags'] = [clean_text(tag) for tag in entry['tags']]
    # elif entry['tags'] is None:
    #      entry['tags'] = []
    # else:
    #      entry['tags'] = [clean_text(entry['tags'])] # Handle cases where tags might not be a list initially

    return entry

print("\nPreprocessing dataset...")
# Apply preprocessing to the training split
processed_dataset = dataset.map(preprocess_entry)

print("Preprocessing completed!")

print("\nFirst 5 preprocessed examples:")
print(processed_dataset['train'][:5])

# You would typically save the processed dataset here for later use
# processed_dataset['train'].save_to_disk(os.path.join(task2_dir, 'processed_dataset')) 