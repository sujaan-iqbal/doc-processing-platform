import re
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class NLPPipeline:
    def __init__(self):
        """Initialize NLP pipeline with lightweight models"""
        self.tokenizer = Tokenizer("english")
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        # Simple word and sentence counting
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Extract features
        features = {
            'word_count': len([w for w in words if w.isalnum()]),
            'sentence_count': len(sentences),
            'entities': self._extract_entities(text),
            'tokens': [w.lower() for w in words if w.isalnum() and len(w) > 2][:20]
        }
        
        return text, features
    
    def _extract_entities(self, text):
        """Simple entity extraction using regex"""
        entities = []
        
        # Find capitalized phrases (potential named entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for entity in capitalized[:5]:
            entities.append((entity, 'ENTITY'))
        
        # Find numbers/dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        for num in numbers[:3]:
            entities.append((num, 'NUMBER'))
        
        return entities
    
    def summarize_text(self, text, sentences_count=3):
        """Generate extractive summary using LSA"""
        try:
            parser = PlaintextParser.from_string(text, self.tokenizer)
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count)
            
            if summary:
                return ' '.join([str(sentence) for sentence in summary])
            else:
                # Fallback: return first few sentences
                sentences = nltk.sent_tokenize(text)
                return ' '.join(sentences[:sentences_count])
        except:
            # Fallback for any summarization errors
            sentences = nltk.sent_tokenize(text)
            return ' '.join(sentences[:min(sentences_count, len(sentences))])
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        
        # Get sentiment scores
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Calculate emotion scores
        emotion_scores = {
            'positive': max(0, min(1, (polarity + 1) / 2)),
            'negative': abs(min(0, polarity)),
            'neutral': 1 - abs(polarity)
        }
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment,
            'emotion_scores': emotion_scores
        }
    
    def process_document(self, text, summary_length=3):
        """Complete NLP pipeline"""
        # Preprocess
        cleaned_text, features = self.preprocess_text(text)
        
        # Summarize
        summary = self.summarize_text(cleaned_text, summary_length)
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(cleaned_text)
        
        # Aggregate results
        return {
            'original_text': text[:500] + '...' if len(text) > 500 else text,
            'cleaned_text': cleaned_text[:500] + '...' if len(cleaned_text) > 500 else cleaned_text,
            'features': features,
            'summary': summary,
            'sentiment': sentiment,
            'metadata': {
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0
            }
        }