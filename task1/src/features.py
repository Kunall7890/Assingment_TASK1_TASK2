import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer with vectorizers and encoders."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3)
        )
        self.issue_type_encoder = LabelEncoder()
        self.urgency_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Define urgency keywords
        self.urgency_keywords = {
            'low_urgency': ['delay', 'delayed', 'late', 'wait'],
            'medium_urgency': ['important', 'asap', 'soon', 'request'],
            'high_urgency': ['urgent', 'immediately', 'critical', 'down', 'broken', 'error'],
            'critical_urgency': ['critical', 'outage', 'urgent', 'down', 'broken', 'error', 'severe']
        }

    def extract_urgency_keywords_features(self, text: str) -> np.ndarray:
        """
        Extract features based on the presence of urgency keywords.

        Args:
            text (str): Preprocessed text

        Returns:
            np.ndarray: Feature vector based on keyword presence
        """
        features = []
        for level, keywords in self.urgency_keywords.items():
            # Check if any keyword from the level is in the text
            features.append(any(keyword in text for keyword in keywords))
        return np.array(features).astype(int)

    def extract_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features, text length, and urgency keyword features from text.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: Combined features (TF-IDF, text length, and urgency keywords)
        """
        # Calculate TF-IDF features
        if not self.is_fitted:
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
            self.is_fitted = True
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()

        # Calculate text length feature
        text_lengths = np.array([len(text) for text in texts]).reshape(-1, 1)

        # Calculate urgency keyword features for each text
        urgency_keyword_features = np.vstack([self.extract_urgency_keywords_features(text) for text in texts])

        # Combine features
        combined_features = np.hstack((tfidf_features, text_lengths, urgency_keyword_features))

        return combined_features
    
    def encode_labels(self, issue_types: List[str], urgency_levels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode categorical labels.
        
        Args:
            issue_types (List[str]): List of issue types
            urgency_levels (List[str]): List of urgency levels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Encoded issue types and urgency levels
        """
        logger.info("Encoding labels...")
        encoded_issues = self.issue_type_encoder.fit_transform(issue_types)
        encoded_urgency = self.urgency_encoder.fit_transform(urgency_levels)
        return encoded_issues, encoded_urgency
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from TF-IDF vectorizer.
        
        Returns:
            List[str]: List of feature names
        """
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def get_label_mappings(self) -> Dict[str, Dict[int, str]]:
        """
        Get mappings between encoded labels and original labels.
        
        Returns:
            Dict[str, Dict[int, str]]: Dictionary containing label mappings
        """
        return {
            'issue_type': dict(zip(
                self.issue_type_encoder.transform(self.issue_type_encoder.classes_),
                self.issue_type_encoder.classes_
            )),
            'urgency_level': dict(zip(
                self.urgency_encoder.transform(self.urgency_encoder.classes_),
                self.urgency_encoder.classes_
            ))
        }

class FeatureProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the feature processor.
        
        Args:
            data_path (str): Path to the preprocessed data
        """
        self.data_path = data_path
        self.feature_engineer = FeatureEngineer()
        
    def load_data(self) -> pd.DataFrame:
        """
        Load preprocessed data.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def process_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Process features and labels.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]: 
                - Text features (combined)
                - Encoded issue types
                - Encoded urgency levels
                - Label mappings
        """
        # Load data
        df = self.load_data()
        
        # Fill NaN in processed_text with empty string
        if 'processed_text' in df.columns:
            df['processed_text'] = df['processed_text'].fillna("")
        
        # Extract combined text features
        combined_text_features = self.feature_engineer.extract_text_features(df['processed_text'].tolist())
        
        # Encode labels
        issue_types, urgency_levels = self.feature_engineer.encode_labels(
            df['issue_type'].tolist(),
            df['urgency_level'].tolist()
        )
        
        # Get label mappings
        label_mappings = self.feature_engineer.get_label_mappings()
        
        logger.info("Feature processing completed")
        return combined_text_features, issue_types, urgency_levels, label_mappings

def main():
    """Main function to run feature processing."""
    try:
        # Initialize processor
        processor = FeatureProcessor('data/processed_tickets.csv')
        
        # Process features
        text_features, issue_types, urgency_levels, label_mappings = processor.process_features()
        
        # Save processed features
        np.save('data/text_features.npy', text_features)
        np.save('data/issue_types.npy', issue_types)
        np.save('data/urgency_levels.npy', urgency_levels)
        
        # Save label mappings
        import json
        with open('data/label_mappings.json', 'w') as f:
            json.dump(label_mappings, f)
        
        logger.info("Processed features saved to data directory")
        
    except Exception as e:
        logger.error(f"Error in feature processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 