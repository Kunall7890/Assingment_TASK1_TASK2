import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with NLTK components."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

class DataPreprocessor:
    def __init__(self, file_path: str):
        """
        Initialize the data preprocessor.
        
        Args:
            file_path (str): Path to the input Excel file
        """
        self.file_path = file_path
        self.text_preprocessor = TextPreprocessor()
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_excel(self.file_path)
            logger.info(f"Successfully loaded data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        # Fill missing text with empty string
        df['ticket_text'] = df['ticket_text'].fillna('')
        
        # Fill missing labels with 'Unknown'
        df['issue_type'] = df['issue_type'].fillna('Unknown')
        df['urgency_level'] = df['urgency_level'].fillna('Unknown')
        
        return df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the entire dataset.
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Load data
        df = self.load_data()
        
        # Handle missing data
        df = self.handle_missing_data(df)
        
        # Preprocess text
        logger.info("Preprocessing text data...")
        df['processed_text'] = df['ticket_text'].apply(self.text_preprocessor.preprocess_text)
        
        # Add text length feature
        df['text_length'] = df['ticket_text'].str.len()
        
        logger.info("Data preprocessing completed")
        return df

def main():
    """Main function to run preprocessing."""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor('data/ai_dev_assignment_tickets_complex_1000.xls')
        
        # Preprocess data
        processed_df = preprocessor.preprocess_data()
        
        # Save processed data
        processed_df.to_csv('data/processed_tickets.csv', index=False)
        logger.info("Processed data saved to 'data/processed_tickets.csv'")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 