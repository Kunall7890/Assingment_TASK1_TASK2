from setuptools import setup, find_packages

setup(
    name="ticket-analyzer",
    version="1.0.0",
    description="A machine learning pipeline for analyzing customer support tickets",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "nltk>=3.6.2",
        "spacy>=3.1.0",
        "gradio>=3.0.0",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "openpyxl>=3.0.7",
        "python-dotenv>=0.19.0",
        "opencv-python>=4.5.3",
        "Pillow>=8.3.1",
        "pytest>=6.2.5",
        "black>=21.7b0",
        "flake8>=3.9.2",
        "mypy>=0.910"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 