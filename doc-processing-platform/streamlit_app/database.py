import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId
import json

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            self.client.server_info()  # Test connection
            self.db = self.client['doc_processing']
            self.collection = self.db['nlp_results']
            self.use_mongodb = True
            print("✅ Connected to MongoDB")
        except Exception as e:
            # Fallback to file-based storage if MongoDB not available
            print(f"⚠️ MongoDB not available, using file storage: {e}")
            self.use_mongodb = False
            self.file_path = os.path.join(os.path.dirname(__file__), 'data', 'results.json')
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            self._init_file_storage()
    
    def _init_file_storage(self):
        """Initialize JSON file storage"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)
    
    def _load_file_data(self):
        """Load data from JSON file"""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_file_data(self, data):
        """Save data to JSON file"""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, default=str, indent=2)
    
    def save_result(self, result):
        """Save NLP result to storage"""
        result['_id'] = str(ObjectId())
        result['document_id'] = result.get('document_id', result['_id'])
        result['created_at'] = datetime.now().isoformat()
        
        if self.use_mongodb:
            # Convert to MongoDB compatible format
            mongo_result = result.copy()
            mongo_result['created_at'] = datetime.now()
            self.collection.insert_one(mongo_result)
        else:
            # Save to JSON file
            data = self._load_file_data()
            data.append(result)
            self._save_file_data(data)
        
        return result['_id']
    
    def get_total_documents(self):
        """Get total number of processed documents"""
        if self.use_mongodb:
            return self.collection.count_documents({})
        else:
            data = self._load_file_data()
            return len(data)
    
    def get_average_polarity(self):
        """Get average sentiment polarity"""
        if self.use_mongodb:
            pipeline = [
                {'$match': {'sentiment.polarity': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'avg_polarity': {'$avg': '$sentiment.polarity'}
                }}
            ]
            result = list(self.collection.aggregate(pipeline))
            return result[0]['avg_polarity'] if result else 0
        else:
            data = self._load_file_data()
            if not data:
                return 0
            polarities = []
            for doc in data:
                if isinstance(doc.get('sentiment'), dict):
                    polarity = doc['sentiment'].get('polarity', 0)
                    if polarity is not None:
                        polarities.append(polarity)
            return sum(polarities) / len(polarities) if polarities else 0
    
    def get_analytics(self):
        """Get comprehensive analytics"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if self.use_mongodb:
            # MongoDB analytics
            total_docs = self.collection.count_documents({})
            today_docs = self.collection.count_documents({
                'created_at': {'$gte': today}
            })
            
            # Sentiment distribution
            sentiment_pipeline = [
                {'$match': {'sentiment.sentiment': {'$exists': True}}},
                {'$group': {
                    '_id': '$sentiment.sentiment',
                    'count': {'$sum': 1}
                }}
            ]
            sentiment_results = list(self.collection.aggregate(sentiment_pipeline))
            
            # Format sentiment distribution
            sentiment_dist = []
            sentiment_map = {'positive': 0, 'negative': 0, 'neutral': 0}
            for item in sentiment_results:
                sentiment_map[item['_id']] = item['count']
            
            for sentiment, count in sentiment_map.items():
                sentiment_dist.append({'sentiment': sentiment, 'count': count})
            
            # Recent documents
            recent_docs = list(self.collection.find(
                {},
                {'document_id': 1, 'sentiment.sentiment': 1, 'created_at': 1, '_id': 0}
            ).sort('created_at', -1).limit(10))
            
            # Format recent documents
            for doc in recent_docs:
                if 'created_at' in doc:
                    doc['created_at'] = doc['created_at'].isoformat()
                if 'sentiment' in doc and isinstance(doc['sentiment'], dict):
                    doc['sentiment'] = doc['sentiment'].get('sentiment', 'unknown')
            
            # Average word count
            word_count_pipeline = [
                {'$match': {'features.word_count': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'avg_words': {'$avg': '$features.word_count'}
                }}
            ]
            avg_words_result = list(self.collection.aggregate(word_count_pipeline))
            avg_words = int(avg_words_result[0]['avg_words']) if avg_words_result else 0
            
            # Daily counts
            daily_counts = []
            for i in range(7):
                date = today - timedelta(days=i)
                next_date = date + timedelta(days=1)
                count = self.collection.count_documents({
                    'created_at': {
                        '$gte': date,
                        '$lt': next_date
                    }
                })
                daily_counts.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'count': count
                })
            
        else:
            # File-based analytics
            data = self._load_file_data()
            
            total_docs = len(data)
            
            # Today's documents
            today_str = today.isoformat()
            today_docs = sum(1 for doc in data 
                           if doc.get('created_at', '')[:10] == today_str[:10])
            
            # Sentiment distribution
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for doc in data:
                sentiment = doc.get('sentiment', {})
                if isinstance(sentiment, dict):
                    sent = sentiment.get('sentiment', 'neutral')
                else:
                    sent = 'neutral'
                sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
            
            sentiment_dist = [
                {'sentiment': k, 'count': v} 
                for k, v in sentiment_counts.items()
            ]
            
            # Recent documents
            sorted_docs = sorted(
                data, 
                key=lambda x: x.get('created_at', '2000-01-01'), 
                reverse=True
            )
            recent_docs = []
            for doc in sorted_docs[:10]:
                sentiment_val = doc.get('sentiment', {})
                if isinstance(sentiment_val, dict):
                    sentiment_str = sentiment_val.get('sentiment', 'unknown')
                else:
                    sentiment_str = 'unknown'
                    
                recent_docs.append({
                    'document_id': doc.get('document_id', doc.get('_id', 'unknown')),
                    'sentiment': sentiment_str,
                    'created_at': doc.get('created_at', 'unknown')
                })
            
            # Average word count
            word_counts = []
            for doc in data:
                features = doc.get('features', {})
                if isinstance(features, dict):
                    wc = features.get('word_count', 0)
                    if wc:
                        word_counts.append(wc)
            avg_words = int(sum(word_counts) / len(word_counts)) if word_counts else 0
            
            # Daily counts
            daily_counts = []
            for i in range(7):
                date = today - timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
                count = sum(1 for doc in data 
                          if doc.get('created_at', '')[:10] == date_str)
                daily_counts.append({
                    'date': date_str,
                    'count': count
                })
        
        # Return consistent format
        return {
            'total_documents': total_docs,
            'today_count': today_docs,
            'avg_polarity': round(self.get_average_polarity(), 3),
            'avg_word_count': avg_words,
            'sentiment_distribution': sentiment_dist,
            'daily_counts': list(reversed(daily_counts)),
            'recent_documents': recent_docs
        }