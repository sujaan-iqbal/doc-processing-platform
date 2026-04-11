import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from nlp_pipeline import NLPPipeline
from database import DatabaseManager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Processing System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_nlp_pipeline():
    return NLPPipeline()

@st.cache_resource
def load_database():
    return DatabaseManager()

nlp = load_nlp_pipeline()
db = load_database()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background: #c6f6d5;
        color: #22543d;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .sentiment-negative {
        background: #fed7d7;
        color: #742a2a;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .sentiment-neutral {
        background: #e2e8f0;
        color: #4a5568;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 Document Processing System</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
        AI-Powered Text Analysis • Summarization • Sentiment Detection • Batch Processing
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    processing_mode = st.radio(
        "Select Mode",
        ["Single Document", "Batch Processing", "Analytics Dashboard"],
        help="Choose how you want to process documents"
    )
    
    st.divider()
    
    st.subheader("🔧 NLP Settings")
    summary_length = st.slider(
        "Summary Length (sentences)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of sentences in the summary"
    )
    
    st.divider()
    
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents Processed", db.get_total_documents())
    with col2:
        st.metric("Avg. Polarity", f"{db.get_average_polarity():.3f}")
    
    st.divider()
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.success("Cache cleared!")
        time.sleep(1)
        st.rerun()

# Main content area
if processing_mode == "Single Document":
    st.header("📝 Process Single Document")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload File", "✏️ Paste Text", "🎯 Try Sample"],
        horizontal=True
    )
    
    text_to_process = ""
    
    if input_method == "📄 Upload File":
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt'],
            help="Upload a .txt file to process"
        )
        
        if uploaded_file is not None:
            text_to_process = uploaded_file.read().decode('utf-8')
            st.success(f"✅ File loaded: {uploaded_file.name}")
            with st.expander("Preview text"):
                st.text(text_to_process[:500] + "..." if len(text_to_process) > 500 else text_to_process)
    
    elif input_method == "✏️ Paste Text":
        text_to_process = st.text_area(
            "Enter your text:",
            height=200,
            placeholder="Paste your document text here..."
        )
    
    else:  # Try Sample
        sample_texts = {
            "Product Review": """The new smartphone is absolutely amazing! The camera quality is outstanding and the battery life lasts all day. The user interface is smooth and intuitive. However, the price point is a bit high and the charging cable is too short. Overall, I'm very satisfied with my purchase and would recommend it to others.""",
            
            "News Article": """Breaking News: Global climate summit reaches historic agreement. World leaders have committed to reducing carbon emissions by 50% by 2030. Environmental activists celebrate the landmark decision while industry representatives express concerns about implementation costs. The agreement includes provisions for developing nations and establishes a new international monitoring framework.""",
            
            "Business Report": """Q3 earnings report shows mixed results. Revenue increased by 15% year-over-year, driven by strong performance in the cloud services division. However, hardware sales declined by 8% due to supply chain disruptions. The company maintains a positive outlook for Q4, citing new product launches and expanding market share in emerging economies."""
        }
        
        selected_sample = st.selectbox("Choose a sample:", list(sample_texts.keys()))
        text_to_process = sample_texts[selected_sample]
        st.info("📋 Sample text loaded. Click 'Process Document' to analyze.")
        with st.expander("Preview sample text"):
            st.text(text_to_process)
    
    # Process button
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if text_to_process:
            with st.spinner("🔄 Processing document... This may take a few seconds."):
                progress_bar = st.progress(0)
                
                # Simulate progress steps
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Process with NLP pipeline
                results = nlp.process_document(text_to_process, summary_length)
                
                # Save to database
                doc_id = db.save_result(results)
                
                progress_bar.empty()
            
            st.success(f"✅ Document processed successfully! (ID: {doc_id})")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Sentiment metrics
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment = results['sentiment']
            sentiment_class = f"sentiment-{sentiment['sentiment']}"
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0">{sentiment['sentiment'].upper()}</h3>
                    <p style="margin:0; color:#666">Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0">{sentiment['polarity']:.3f}</h3>
                    <p style="margin:0; color:#666">Polarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0">{results['features']['word_count']}</h3>
                    <p style="margin:0; color:#666">Words</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                compression = results['metadata']['compression_ratio']
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0">{compression:.1%}</h3>
                    <p style="margin:0; color:#666">Compression</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Summary
            st.subheader("📋 Summary")
            st.markdown(f"""
            <div class="result-card">
                <p style="font-size:1.1rem; line-height:1.6;">{results['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("😊 Sentiment Analysis")
                
                # Emotion scores chart
                emotion_df = pd.DataFrame({
                    'Emotion': list(sentiment['emotion_scores'].keys()),
                    'Score': list(sentiment['emotion_scores'].values())
                })
                
                fig = px.bar(
                    emotion_df,
                    x='Emotion',
                    y='Score',
                    color='Emotion',
                    color_discrete_map={
                        'positive': '#48bb78',
                        'negative': '#f56565',
                        'neutral': '#a0aec0'
                    }
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
            
            with col2:
                st.subheader("🔍 Text Features")
                
                features = results['features']
                
                # Entities
                if features['entities']:
                    st.write("**Named Entities:**")
                    for entity, label in features['entities'][:5]:
                        st.markdown(f"- **{entity}** ({label})")
                
                st.write("**Statistics:**")
                st.write(f"- Sentences: {features['sentence_count']}")
                st.write(f"- Avg. words/sentence: {features['word_count'] / max(1, features['sentence_count']):.1f}")
                
                # Top tokens
                if features['tokens']:
                    st.write("**Top Keywords:**")
                    keyword_string = ", ".join(features['tokens'][:10])
                    st.markdown(f"`{keyword_string}`")
            
            # Export options
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download Results as JSON", use_container_width=True):
                    import json
                    json_str = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="Click to Download",
                        data=json_str,
                        file_name=f"analysis_{doc_id}.json",
                        mime="application/json"
                    )
            
            with col2:
                # Generate shareable link (simulated)
                st.button("🔗 Copy Shareable Link", use_container_width=True)
            
        else:
            st.warning("⚠️ Please provide text to process")

elif processing_mode == "Batch Processing":
    st.header("📦 Batch Document Processing")
    
    st.info("Upload multiple documents or paste several texts for batch processing")
    
    batch_input_method = st.radio(
        "Choose batch input method:",
        ["📁 Multiple Files", "✏️ Multiple Texts"],
        horizontal=True
    )
    
    documents = []
    
    if batch_input_method == "📁 Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose text files",
            type=['txt'],
            accept_multiple_files=True,
            help="Select multiple .txt files"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                documents.append({
                    'name': file.name,
                    'text': file.read().decode('utf-8')
                })
            
            st.success(f"✅ {len(uploaded_files)} files loaded")
    
    else:
        st.write("Enter multiple texts (one per input box):")
        
        num_texts = st.number_input("Number of documents", min_value=1, max_value=10, value=2)
        
        for i in range(num_texts):
            text = st.text_area(
                f"Document {i+1}",
                height=100,
                key=f"batch_text_{i}",
                placeholder=f"Enter text for document {i+1}..."
            )
            if text:
                documents.append({
                    'name': f"Document {i+1}",
                    'text': text
                })
    
    if documents and st.button("🚀 Process Batch", type="primary", use_container_width=True):
        st.subheader("📊 Batch Results")
        
        results_list = []
        progress_bar = st.progress(0)
        
        for idx, doc in enumerate(documents):
            with st.spinner(f"Processing {doc['name']}..."):
                results = nlp.process_document(doc['text'], summary_length)
                results['document_name'] = doc['name']
                results_list.append(results)
                
                # Save to database
                db.save_result(results)
                
                progress_bar.progress((idx + 1) / len(documents))
        
        progress_bar.empty()
        st.success(f"✅ Processed {len(documents)} documents successfully!")
        
        # Display summary table
        summary_data = []
        for r in results_list:
            summary_data.append({
                'Document': r['document_name'][:30],
                'Sentiment': r['sentiment']['sentiment'],
                'Polarity': f"{r['sentiment']['polarity']:.3f}",
                'Words': r['features']['word_count'],
                'Summary': r['summary'][:100] + "..."
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Sentiment distribution chart
        sentiment_counts = df['Sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#48bb78',
                'negative': '#f56565',
                'neutral': '#a0aec0'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results in expanders
        for idx, result in enumerate(results_list):
            with st.expander(f"📄 {result['document_name']} - {result['sentiment']['sentiment'].upper()}"):
                st.write("**Summary:**", result['summary'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Polarity", f"{result['sentiment']['polarity']:.3f}")
                with col2:
                    st.metric("Word Count", result['features']['word_count'])


else:  # Analytics Dashboard
    st.header("📈 Analytics Dashboard")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    # Get analytics data
    analytics = db.get_analytics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", analytics['total_documents'])
    with col2:
        st.metric("Processed Today", analytics['today_count'])
    with col3:
        st.metric("Avg. Polarity", f"{analytics['avg_polarity']:.3f}")
    with col4:
        st.metric("Avg. Words/Doc", analytics['avg_word_count'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        if analytics['sentiment_distribution']:
            # Create DataFrame with correct column names
            sentiment_df = pd.DataFrame(analytics['sentiment_distribution'])
            
            # Ensure we have the right columns
            if 'sentiment' in sentiment_df.columns and 'count' in sentiment_df.columns:
                fig = px.pie(
                    sentiment_df,
                    values='count',
                    names='sentiment',  # Fixed: use 'sentiment' instead of 'sentiment'
                    color='sentiment',
                    color_discrete_map={
                        'positive': '#48bb78',
                        'negative': '#f56565',
                        'neutral': '#a0aec0'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available yet. Process some documents first!")
        else:
            st.info("No data available. Process some documents to see analytics!")
    
    with col2:
        st.subheader("📈 Processing Timeline")
        if analytics['daily_counts']:
            timeline_df = pd.DataFrame(analytics['daily_counts'])
            if not timeline_df.empty:
                fig = px.line(
                    timeline_df,
                    x='date',
                    y='count',
                    title="Documents Processed per Day"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data available")
    
    # Recent documents
    st.subheader("📋 Recent Documents")
    if analytics['recent_documents']:
        recent_df = pd.DataFrame(analytics['recent_documents'])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent documents")
    
    # Export analytics
    if st.button("📥 Export Analytics Report"):
        report = {
            'generated_at': datetime.now().isoformat(),
            'analytics': analytics
        }
        import json
        st.download_button(
            label="Download Report",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )