import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, metrics_path: str, entity_stats_path: str):
        """
        Initialize the visualizer with paths to results.
        
        Args:
            metrics_path (str): Path to model metrics
            entity_stats_path (str): Path to entity statistics
        """
        self.metrics_path = metrics_path
        self.entity_stats_path = entity_stats_path
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def load_data(self) -> Tuple[Dict, Dict]:
        """
        Load metrics and entity statistics.
        
        Returns:
            Tuple[Dict, Dict]: Model metrics and entity statistics
        """
        try:
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
            
            with open(self.entity_stats_path, 'r') as f:
                entity_stats = json.load(f)
            
            return metrics, entity_stats
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def plot_classification_metrics(self, save_path: str = 'visualizations/classification_metrics.png'):
        """
        Plot classification metrics for both tasks.
        
        Args:
            save_path (str): Path to save the plot
        """
        metrics, _ = self.load_data()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot issue type metrics
        issue_metrics = pd.DataFrame(metrics['issue_type']).T
        issue_metrics[['precision', 'recall', 'f1-score']].plot(
            kind='bar',
            ax=ax1,
            title='Issue Type Classification Metrics'
        )
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot urgency level metrics
        urgency_metrics = pd.DataFrame(metrics['urgency_level']).T
        urgency_metrics[['precision', 'recall', 'f1-score']].plot(
            kind='bar',
            ax=ax2,
            title='Urgency Level Classification Metrics'
        )
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_entity_distribution(self, save_path: str = 'visualizations/entity_distribution.png'):
        """
        Plot distribution of extracted entities.
        
        Args:
            save_path (str): Path to save the plot
        """
        _, entity_stats = self.load_data()
        
        # Prepare data
        entity_counts = {
            entity_type: stats['count']
            for entity_type, stats in entity_stats.items()
        }
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=list(entity_counts.keys()),
            y=list(entity_counts.values())
        )
        plt.title('Distribution of Extracted Entities')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_most_common_entities(self, save_path: str = 'visualizations/most_common_entities.png'):
        """
        Plot most common entities for each type.
        
        Args:
            save_path (str): Path to save the plot
        """
        _, entity_stats = self.load_data()
        
        # Create figure with subplots
        n_entities = len(entity_stats)
        n_cols = 2
        n_rows = (n_entities + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for idx, (entity_type, stats) in enumerate(entity_stats.items()):
            if not stats['most_common']:
                continue
                
            # Get top 5 most common entities
            entities = list(stats['most_common'].keys())[:5]
            counts = list(stats['most_common'].values())[:5]
            
            # Plot
            sns.barplot(x=counts, y=entities, ax=axes[idx])
            axes[idx].set_title(f'Most Common {entity_type}')
            axes[idx].set_xlabel('Count')
            axes[idx].set_ylabel('Entity')
        
        # Remove empty subplots
        for idx in range(len(entity_stats), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        try:
            # Create visualizations directory
            import os
            os.makedirs('visualizations', exist_ok=True)
            
            # Generate plots
            self.plot_classification_metrics()
            self.plot_entity_distribution()
            self.plot_most_common_entities()
            
            logger.info("All visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def main():
    """Main function to generate visualizations."""
    try:
        # Initialize visualizer
        visualizer = Visualizer(
            metrics_path='models/metrics.json',
            entity_stats_path='data/entity_statistics.json'
        )
        
        # Generate all visualizations
        visualizer.generate_all_visualizations()
        
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 