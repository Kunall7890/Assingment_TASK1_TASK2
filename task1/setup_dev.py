import os
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_virtual_environment():
    """Create a virtual environment."""
    try:
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating virtual environment: {str(e)}")
        raise

def install_dependencies():
    """Install project dependencies."""
    try:
        logger.info("Installing dependencies...")
        
        # Determine the pip command based on the platform
        if os.name == 'nt':  # Windows
            pip_cmd = os.path.join("venv", "Scripts", "pip")
        else:  # Unix-like
            pip_cmd = os.path.join("venv", "bin", "pip")
        
        # Upgrade pip
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install dependencies
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        # Install the package in development mode
        subprocess.run([pip_cmd, "install", "-e", "."], check=True)
        
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        raise

def download_nltk_data():
    """Download required NLTK data."""
    try:
        logger.info("Downloading NLTK data...")
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

def download_spacy_model():
    """Download required spaCy model."""
    try:
        logger.info("Downloading spaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        logger.info("spaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading spaCy model: {str(e)}")
        raise

def create_directories():
    """Create necessary directories."""
    try:
        logger.info("Creating project directories...")
        directories = [
            'data',
            'models',
            'visualizations',
            'logs',
            'tests',
            'notebooks',
            'docs'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("Project directories created successfully")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise

def setup_git():
    """Set up Git repository."""
    try:
        logger.info("Setting up Git repository...")
        
        # Initialize Git repository if it doesn't exist
        if not Path('.git').exists():
            subprocess.run(['git', 'init'], check=True)
        
        # Create .gitignore file
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/
models/
visualizations/
logs/
*.log
*.mp4
        """
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        
        logger.info("Git repository set up successfully")
    except Exception as e:
        logger.error(f"Error setting up Git repository: {str(e)}")
        raise

def main():
    """Main function to set up the development environment."""
    try:
        # Create virtual environment
        create_virtual_environment()
        
        # Install dependencies
        install_dependencies()
        
        # Download required data
        download_nltk_data()
        download_spacy_model()
        
        # Create directories
        create_directories()
        
        # Set up Git
        setup_git()
        
        logger.info("Development environment set up successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            logger.info("   .\\venv\\Scripts\\activate")
        else:  # Unix-like
            logger.info("   source venv/bin/activate")
        logger.info("2. Run the pipeline:")
        logger.info("   python src/run_pipeline.py")
        logger.info("3. Launch the application:")
        logger.info("   python src/app.py")
        
    except Exception as e:
        logger.error(f"Error setting up development environment: {str(e)}")
        raise

if __name__ == "__main__":
    main() 