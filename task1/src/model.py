import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from typing import Tuple, Dict, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiTaskModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the multi-task model.
        
        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random seed for reproducibility
        """
        self.base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.model = MultiOutputClassifier(self.base_model)
        self.label_mappings = None
        
    def load_data(self, features_path: str, issue_types_path: str, 
                 urgency_levels_path: str, mappings_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load processed features and labels.
        
        Args:
            features_path (str): Path to text features
            issue_types_path (str): Path to issue type labels
            urgency_levels_path (str): Path to urgency level labels
            mappings_path (str): Path to label mappings
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and stacked labels
        """
        # Load features and labels
        X = np.load(features_path)
        issue_types = np.load(issue_types_path)
        urgency_levels = np.load(urgency_levels_path)
        
        # Stack labels for multi-task learning
        y = np.column_stack((issue_types, urgency_levels))
        
        # Load label mappings
        with open(mappings_path, 'r') as f:
            self.label_mappings = json.load(f)
            
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train the multi-task model.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        logger.info("Training multi-task model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'issue_type': classification_report(
                y_test[:, 0],
                y_pred[:, 0],
                output_dict=True
            ),
            'urgency_level': classification_report(
                y_test[:, 1],
                y_pred[:, 1],
                output_dict=True
            )
        }
        
        # Add accuracy scores
        metrics['issue_type_accuracy'] = accuracy_score(y_test[:, 0], y_pred[:, 0])
        metrics['urgency_accuracy'] = accuracy_score(y_test[:, 1], y_pred[:, 1])
        
        logger.info("Model training completed")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted issue types and urgency levels
        """
        predictions = self.model.predict(X)
        return predictions[:, 0], predictions[:, 1]
    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

def main():
    """Main function to train and evaluate the model."""
    try:
        # Initialize model
        model = MultiTaskModel()
        
        # Load data
        X, y = model.load_data(
            'data/text_features.npy',
            'data/issue_types.npy',
            'data/urgency_levels.npy',
            'data/label_mappings.json'
        )
        
        # Train model
        metrics = model.train(X, y)
        
        # Save model
        model.save_model('models/multi_task_model.joblib')
        
        # Save metrics
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Model training and evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 