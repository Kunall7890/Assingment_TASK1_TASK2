import os
import logging
import numpy as np
from preprocess import DataPreprocessor
from features import FeatureProcessor
from model import MultiTaskModel
from entity_extractor import EntityProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_pipeline():
    """Run the complete machine learning pipeline."""
    try:
        # Create directories
        create_directories()
        
        # Step 1: Data Preprocessing
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor('data/ai_dev_assignment_tickets_complex_1000.xls')
        processed_df = preprocessor.preprocess_data()
        processed_df.to_csv('data/processed_tickets.csv', index=False)
        logger.info("Data preprocessing completed")
        
        # Step 2: Feature Engineering
        logger.info("Starting feature engineering...")
        feature_processor = FeatureProcessor('data/processed_tickets.csv')
        text_features, issue_types, urgency_levels, label_mappings = feature_processor.process_features()
        logger.info("Feature engineering completed")
        
        # Step 3: Model Training
        logger.info("Starting model training...")
        model = MultiTaskModel()
        metrics = model.train(text_features, np.column_stack((issue_types, urgency_levels)))
        model.save_model('models/multi_task_model.joblib')
        logger.info("Model training completed")
        
        # Step 4: Entity Extraction
        logger.info("Starting entity extraction...")
        entity_processor = EntityProcessor('data/processed_tickets.csv')
        entities_list, stats = entity_processor.process_entities()
        logger.info("Entity extraction completed")
        
        # Save final results
        logger.info("Saving final results...")
        import json
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        with open('data/extracted_entities.json', 'w') as f:
            json.dump(entities_list, f, indent=4)
        
        with open('data/entity_statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline() 