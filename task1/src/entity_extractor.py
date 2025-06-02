import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor with spaCy model.
        
        Args:
            model_name (str): Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.info(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Define entity types to extract
        self.target_entities = {
            'PRODUCT': [],
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geo-Political Entities
            'DATE': [],
            'TIME': [],
            'MONEY': [],
            'PERCENT': [],
            'QUANTITY': []
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Dictionary of extracted entities by type
        """
        doc = self.nlp(text)
        entities = {entity_type: [] for entity_type in self.target_entities.keys()}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Process a batch of texts to extract entities.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Dict[str, List[str]]]: List of entity dictionaries
        """
        logger.info(f"Processing {len(texts)} texts for entity extraction...")
        return [self.extract_entities(text) for text in texts]
    
    def get_entity_statistics(self, entities_list: List[Dict[str, List[str]]]) -> Dict:
        """
        Calculate statistics about extracted entities.
        
        Args:
            entities_list (List[Dict[str, List[str]]]): List of entity dictionaries
            
        Returns:
            Dict: Dictionary containing entity statistics
        """
        stats = {
            entity_type: {
                'count': 0,
                'unique_entities': set(),
                'most_common': {}
            }
            for entity_type in self.target_entities.keys()
        }
        
        # Count entities
        for entities in entities_list:
            for entity_type, entity_list in entities.items():
                stats[entity_type]['count'] += len(entity_list)
                stats[entity_type]['unique_entities'].update(entity_list)
                
                # Count occurrences
                for entity in entity_list:
                    if entity in stats[entity_type]['most_common']:
                        stats[entity_type]['most_common'][entity] += 1
                    else:
                        stats[entity_type]['most_common'][entity] = 1
        
        # Convert sets to lists and sort most common entities
        for entity_type in stats:
            stats[entity_type]['unique_entities'] = list(stats[entity_type]['unique_entities'])
            stats[entity_type]['most_common'] = dict(
                sorted(
                    stats[entity_type]['most_common'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Keep top 10 most common entities
            )
        
        return stats

class EntityProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the entity processor.
        
        Args:
            data_path (str): Path to the preprocessed data
        """
        self.data_path = data_path
        self.extractor = EntityExtractor()
    
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
    
    def process_entities(self) -> Tuple[List[Dict], Dict]:
        """
        Process entities from the dataset.
        
        Returns:
            Tuple[List[Dict], Dict]: 
                - List of entity dictionaries
                - Entity statistics
        """
        # Load data
        df = self.load_data()
        
        # Fill NaN in ticket_text with empty string
        if 'ticket_text' in df.columns:
            df['ticket_text'] = df['ticket_text'].fillna("")
        
        # Extract entities
        entities_list = self.extractor.process_batch(df['ticket_text'].tolist())
        
        # Calculate statistics
        stats = self.extractor.get_entity_statistics(entities_list)
        
        logger.info("Entity processing completed")
        return entities_list, stats

def main():
    """Main function to run entity extraction."""
    try:
        # Initialize processor
        processor = EntityProcessor('data/processed_tickets.csv')
        
        # Process entities
        entities_list, stats = processor.process_entities()
        
        # Save entities
        with open('data/extracted_entities.json', 'w') as f:
            json.dump(entities_list, f, indent=4)
        
        # Save statistics
        with open('data/entity_statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        logger.info("Entity extraction completed and results saved")
        
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 