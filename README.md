# AI Development Tasks

This repository contains two AI development tasks showcasing different aspects of natural language processing and machine learning.

## Task 1: Multi-Task Ticket Classification System
A machine learning system that automatically analyzes support tickets to classify issue types, determine urgency levels, and extract relevant entities.

### Key Features
- Multi-task classification (issue type, urgency level)
- Entity extraction
- Real-time analysis through Gradio interface
- Example tickets for testing

[View Task 1 Details](task1/README.md)

## Task 2: RAG-Based Quote Retrieval System
A Retrieval-Augmented Generation (RAG) pipeline for semantic quote retrieval and structured question answering.

### Key Features
- Semantic search for quotes
- Context-aware responses
- Evaluation framework with RAGAS metrics
- Efficient vector similarity search

[View Task 2 Details](task2/README.md)

## Getting Started

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vijaywfh
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies for both tasks:
   ```bash
   # Install Task 1 dependencies
   cd task1
   pip install -r requirements.txt
   cd ..

   # Install Task 2 dependencies
   cd task2
   pip install -r requirements.txt
   cd ..
   ```

## Running the Applications

### Task 1: Ticket Classification
1. Navigate to task1 directory:
   ```bash
   cd task1
   ```

2. Run the training pipeline:
   ```bash
   python src/run_pipeline.py
   ```

3. Start the application:
   ```bash
   python src/app.py
   ```

4. Access the interface at http://127.0.0.1:7860

### Task 2: Quote Retrieval
1. Navigate to task2 directory:
   ```bash
   cd task2
   ```

2. Set up environment variables:
   ```bash
   # Create .env file with your OpenAI API key
   OPENAI_API_KEY=your_api_key_here
   ```

3. Download and preprocess data:
   ```bash
   python src/download_dataset.py
   python src/preprocess_data.py
   ```

4. Build and evaluate the RAG pipeline:
   ```bash
   python src/build_rag_pipeline.py
   python src/evaluate_rag.py
   ```

## Project Structure
```
vijaywfh/
├── task1/                 # Multi-Task Ticket Classification System
│   ├── data/
│   ├── models/
│   ├── src/
│   └── README.md
├── task2/                 # RAG-Based Quote Retrieval System
│   ├── data/
│   ├── models/
│   ├── src/
│   └── README.md
└── README.md             # This file
```

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details. 