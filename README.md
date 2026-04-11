
# Document Processing Platform

AI-Powered Document Analysis with NLP
- Text Summarization
- Sentiment Analysis  
- Batch Processing
- Analytics Dashboard

Check out demo here: https://huggingface.co/spaces/sujaan4596/doc-processing-platform

## Features

- 📝 **Text Summarization** - Extract key insights using LSA
- 😊 **Sentiment Analysis** - Detect emotional tone with polarity scores
- 🔄 **Batch Processing** - Handle multiple documents at once
- 📊 **Analytics Dashboard** - Visualize processing statistics
- 💾 **Export Options** - Download results as JSON

## How to Use

1. **Single Document**: Upload a .txt file or paste text directly
2. **Batch Processing**: Upload multiple files or enter multiple texts
3. **Analytics**: View sentiment distribution and processing trends

## Sample Text for Testing

The new product launch was incredibly successful! Customer feedback has been overwhelmingly positive, and sales have exceeded all expectations. The team worked very hard to make this happen. However, we did face some minor technical issues initially, but these were quickly resolved. Looking forward, we plan to expand to new markets.


## Technology Stack

- **Streamlit** - Web interface
- **TextBlob** - Sentiment analysis
- **Sumy** - Text summarization
- **NLTK** - Natural language processing
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

## Local Development

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
cd streamlit_app
streamlit run app.py

License
MIT License - feel free to use and modify!
