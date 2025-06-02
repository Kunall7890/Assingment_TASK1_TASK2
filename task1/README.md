# Task 1: Multi-Task Ticket Classification System

## Overview
This system automatically analyzes support tickets to classify issue types, determine urgency levels, and extract relevant entities. It uses machine learning to process and categorize customer support tickets in real-time.

## Features
- **Multi-Task Classification**
  - Issue Type Classification
  - Urgency Level Determination
  - Entity Extraction

- **User Interface**
  - Gradio web interface
  - Real-time ticket analysis
  - Example tickets for testing

## Technical Stack
- Python 3.x
- Scikit-learn
- spaCy
- Gradio
- NLTK
- Joblib

## Project Structure
```
task1/
├── data/
│   ├── processed_tickets.csv
│   └── label_mappings.json
├── models/
│   └── multi_task_model.joblib
├── src/
│   ├── app.py
│   ├── model.py
│   ├── features.py
│   ├── preprocess.py
│   └── entity_extractor.py
├── templates/
│   ├── upload.html
│   └── result.html
└── requirements.txt
```

## Setup and Installation
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training pipeline:
   ```bash
   python src/run_pipeline.py
   ```

4. Start the application:
   ```bash
   python src/app.py
   ```

## Usage
1. Access the web interface at http://127.0.0.1:7860
2. Enter a ticket description or use example tickets
3. View the analysis results:
   - Issue Type
   - Urgency Level
   - Extracted Entities

## Example
Input:
```
Order #49712 for RoboChef Blender is 4 days late. Ordered on 25 April.
```

Output:
```json
{
    "issue_type": "Late Delivery",
    "urgency_level": "Medium",
    "entities": {
        "ORDER_ID": ["49712"],
        "PRODUCT": ["RoboChef Blender"],
        "DATE": ["4 days", "25 April"]
    }
}
```

## Model Performance
- Issue Type Classification: Good accuracy
- Urgency Level Classification: Moderate accuracy
- Entity Extraction: High precision

## Future Improvements
1. Enhance urgency level classification
2. Add more domain-specific features
3. Implement model retraining pipeline
4. Add support for more entity types
5. Improve preprocessing pipeline 