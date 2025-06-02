from datasets import load_dataset
import pandas as pd

def download_and_save_dataset():
    # Load the dataset
    dataset = load_dataset("Abirate/english_quotes")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset["train"])
    
    # Save to CSV
    df.to_csv("task2/data/quotes.csv", index=False)
    print(f"Dataset downloaded and saved to task2/data/quotes.csv")
    print(f"Total quotes: {len(df)}")

if __name__ == "__main__":
    download_and_save_dataset() 