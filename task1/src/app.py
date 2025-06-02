import gradio as gr
import numpy as np
import pandas as pd
import json
import joblib
from typing import Dict
from preprocess import TextPreprocessor
from features import FeatureEngineer
from model import MultiTaskModel
from entity_extractor import EntityExtractor
import logging
from flask import Flask, request, render_template, redirect, url_for
import os

# Get the absolute path to the task1 directory
TASK1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class TicketAnalyzer:
    def __init__(self, model_path: str, label_mappings_path: str):
        """
        Initialize the ticket analyzer with trained models.
        
        Args:
            model_path (str): Path to the trained classification model
            label_mappings_path (str): Path to the label mappings
        """
        # Load classification model
        self.model = MultiTaskModel()
        self.model.load_model(model_path)
        
        # Load label mappings
        with open(label_mappings_path, 'r') as f:
            self.label_mappings = json.load(f)
        
        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Initialize feature engineer with training data
        try:
            # Load processed data
            processed_data = pd.read_csv('task1/data/processed_tickets.csv')
            # Fill NaN values with empty string
            processed_data['processed_text'] = processed_data['processed_text'].fillna("")
            # Initialize feature engineer with training data
            self.feature_engineer.extract_text_features(processed_data['processed_text'].tolist())
        except Exception as e:
            logger.error(f"Error initializing feature engineer: {str(e)}")
            raise
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor()
    
    def analyze_ticket(self, ticket_text: str) -> Dict:
        """
        Analyze a support ticket.
        
        Args:
            ticket_text (str): The ticket text to analyze
            
        Returns:
            Dict: Analysis results including predictions and entities
        """
        try:
            # Preprocess text
            processed_text = self.text_preprocessor.preprocess_text(ticket_text)
            
            # Extract features
            text_features = self.feature_engineer.extract_text_features([processed_text])
            
            # Make predictions
            issue_type_pred, urgency_pred = self.model.predict(text_features)
            
            # Map predictions to labels
            issue_type = self.label_mappings['issue_type'][str(issue_type_pred[0])]
            urgency_level = self.label_mappings['urgency_level'][str(urgency_pred[0])]
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(ticket_text)
            
            return {
                'issue_type': issue_type,
                'urgency_level': urgency_level,
                'entities': entities
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ticket: {str(e)}")
            raise

def create_interface():
    """Create and launch the Gradio interface."""
    try:
        # Initialize analyzer
        analyzer = TicketAnalyzer(
            model_path=os.path.join(TASK1_DIR, 'models', 'multi_task_model.joblib'),
            label_mappings_path=os.path.join(TASK1_DIR, 'data', 'label_mappings.json')
        )
        
        def analyze_ticket(ticket_text: str) -> tuple:
            """
            Analyze a ticket and return the results.
            
            Args:
                ticket_text (str): The ticket text to analyze
                
            Returns:
                tuple: (issue_type, urgency_level)
            """
            results = analyzer.analyze_ticket(ticket_text)
            return results['issue_type'], results['urgency_level']
        
        # Create interface
        interface = gr.Interface(
            fn=analyze_ticket,
            inputs=gr.Textbox(
                label="Enter Ticket Description",
                placeholder="Enter your ticket description here...",
                lines=3
            ),
            outputs=[
                gr.Textbox(label="Issue Type"),
                gr.Textbox(label="Urgency Level")
            ],
            title="Ticket Analysis System",
            description="Enter a ticket description to analyze its issue type and urgency level.",
            examples=[
                ["Order #49712 for RoboChef Blender is 4 days late. Ordered on 25 April."],
                ["The new feature in version 2.1.0 is not working as expected. Can you help me troubleshoot?"],
                ["Customer reported that the mobile app crashes when trying to make a payment. This is affecting multiple users."],
                ["Need help with setting up the new printer in the office. Not urgent, can wait until tomorrow."],
                ["Critical security vulnerability found in the login system. Immediate attention required."],
                ["The website is loading very slowly today. Response time is over 10 seconds."],
                ["Can't access my account after the system update. Need help resetting password."],
                ["The delivery tracking system is showing incorrect status updates for all orders."],
                ["New employee needs access to the project management software. Please set up account."],
                ["The automatic backup system failed last night. Need to investigate the cause."]
            ]
        )
        
        return interface
        
    except Exception as e:
        logger.error(f"Error creating interface: {str(e)}")
        raise

def main():
    """Main function to launch the application."""
    try:
        # Create and launch interface
        interface = create_interface()
        interface.launch()
        
    except Exception as e:
        logger.error(f"Error launching application: {str(e)}")
        raise

@app.route('/upload_tickets', methods=['GET', 'POST'])
def upload_tickets():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith(('.xlsx', '.xls')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_excel(filepath)
            # Extract relevant columns
            if 'ticket_id' in df.columns and 'issue_type' in df.columns and 'urgency_level' in df.columns:
                tickets = df[['ticket_id', 'issue_type', 'urgency_level']].to_dict(orient='records')
                return render_template('result.html', tickets=tickets)
            else:
                return "Required columns not found in the uploaded file.", 400
    return render_template('upload.html')

if __name__ == "__main__":
    main() 