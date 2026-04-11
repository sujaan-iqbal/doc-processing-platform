#!/bin/bash
# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

# Create data directory
mkdir -p streamlit_app/data