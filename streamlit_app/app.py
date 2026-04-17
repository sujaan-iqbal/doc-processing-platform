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
import json

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

    /* Summary card */
    .summary-card {
        border-left: 3px solid #4299e1;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        background: #f7fafc;
        margin-bottom: 1rem;
    }
    .summary-card p {
        font-size: 1.05rem;
        line-height: 1.75;
        color: #2d3748;
        margin: 0;
    }

    /* Translation card */
    .translation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .translation-content {
        background: white;
        padding: 20px;
        border-radius: 8px;
    }
    .translation-content h4 {
        color: #667eea;
        margin-top: 0;
        margin-bottom: 15px;
    }
    .translation-content p {
        color: #2d3748;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Status badges */
    .badge-row {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 1.25rem;
    }
    .badge {
        font-size: 12px;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-lang     { background: #ebf8ff; color: #2b6cb0; }
    .badge-positive { background: #c6f6d5; color: #22543d; }
    .badge-negative { background: #fed7d7; color: #742a2a; }
    .badge-neutral  { background: #e2e8f0; color: #4a5568; }
    .badge-id       { background: #f7fafc; color: #718096; border: 1px solid #e2e8f0; margin-left: auto; }

    /* Feature panel */
    .feature-panel {
        background: white;
        border: 0.5px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
    .panel-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #718096;
        margin-bottom: 10px;
    }

    /* Keyword chips */
    .kw-chip {
        display: inline-block;
        font-size: 11px;
        background: #f7fafc;
        color: #4a5568;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px 2px 2px 0;
    }
    .entity-chip {
        display: inline-block;
        font-size: 11px;
        background: #ebf8ff;
        color: #2b6cb0;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px 2px 2px 0;
    }

    /* Emotion bar */
    .emo-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    .emo-label {
        font-size: 13px;
        color: #4a5568;
        width: 60px;
        flex-shrink: 0;
    }
    .emo-track {
        flex: 1;
        height: 8px;
        background: #edf2f7;
        border-radius: 4px;
        overflow: hidden;
    }
    .emo-fill-pos { height: 100%; border-radius: 4px; background: #48bb78; }
    .emo-fill-neg { height: 100%; border-radius: 4px; background: #fc8181; }
    .emo-fill-neu { height: 100%; border-radius: 4px; background: #a0aec0; }
    .emo-val {
        font-size: 12px;
        color: #718096;
        width: 36px;
        text-align: right;
        flex-shrink: 0;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 1.5rem 0;
    }

    /* Info / success boxes */
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #4299e1;
        padding: 12px 20px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .info-box p { margin: 0; color: #2b6cb0; }

    .success-box {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        padding: 12px 20px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .success-box p { margin: 0; color: #22543d; }

    /* Button hover */
    .stButton > button {
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def render_badge_row(results, doc_id):
    sentiment = results['sentiment']['sentiment']
    lang_name = results.get('detected_language_name', 'Unknown')
    badge_class = f"badge-{sentiment}"
    sentiment_icon = {"positive": "↑", "negative": "↓", "neutral": "–"}.get(sentiment, "–")
    st.markdown(f"""
    <div class="badge-row">
        <span class="badge badge-lang">🌐 {lang_name}</span>
        <span class="badge {badge_class}">{sentiment_icon} {sentiment.capitalize()}</span>
        <span class="badge badge-id">doc #{doc_id}</span>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(results):
    sentiment = results['sentiment']
    compression = results['metadata']['compression_ratio']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_color = "normal" if sentiment['polarity'] >= 0 else "inverse"
        st.metric("Polarity", f"{sentiment['polarity']:.3f}")
    with col2:
        st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
    with col3:
        st.metric("Word count", results['features']['word_count'])
    with col4:
        st.metric("Compression", f"{compression:.1%}")


def render_summary(results):
    st.markdown('<p class="panel-label">Summary</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-card">
        <p>{results['summary']}</p>
    </div>
    """, unsafe_allow_html=True)


def render_emotion_bars(emotion_scores):
    fill_class = {
        'positive': 'emo-fill-pos',
        'negative': 'emo-fill-neg',
        'neutral':  'emo-fill-neu',
    }
    bars_html = ""
    for emotion, score in emotion_scores.items():
        pct = min(max(float(score) * 100, 0), 100)
        cls = fill_class.get(emotion, 'emo-fill-neu')
        bars_html += f"""
        <div class="emo-row">
            <span class="emo-label">{emotion.capitalize()}</span>
            <div class="emo-track"><div class="{cls}" style="width:{pct:.0f}%"></div></div>
            <span class="emo-val">{score:.2f}</span>
        </div>"""
    st.markdown(f'<div class="panel-label">Emotion scores</div>{bars_html}',
                unsafe_allow_html=True)


def render_text_features(features):
    st.markdown('<p class="panel-label">Text features</p>', unsafe_allow_html=True)

    avg_words = features['word_count'] / max(1, features['sentence_count'])

    st.markdown(f"""
    <div style="font-size:13px; color:#4a5568; margin-bottom:12px; line-height:1.8;">
        <span style="color:#2d3748;font-weight:500;">Sentences</span> {features['sentence_count']}
        &nbsp;&nbsp;
        <span style="color:#2d3748;font-weight:500;">Avg. words / sentence</span> {avg_words:.1f}
    </div>
    """, unsafe_allow_html=True)

    if features.get('entities'):
        chips = "".join(
            f'<span class="entity-chip">{ent} · {label}</span>'
            for ent, label in features['entities'][:6]
        )
        st.markdown(
            f'<div style="margin-bottom:10px"><span style="font-size:12px;color:#718096;'
            f'font-weight:600;">Named entities</span><br><div style="margin-top:4px">{chips}</div></div>',
            unsafe_allow_html=True)

    if features.get('tokens'):
        chips = "".join(
            f'<span class="kw-chip">{kw}</span>'
            for kw in features['tokens'][:10]
        )
        st.markdown(
            f'<div><span style="font-size:12px;color:#718096;font-weight:600;">Top keywords</span>'
            f'<br><div style="margin-top:4px">{chips}</div></div>',
            unsafe_allow_html=True)


def render_analysis_results(results, doc_id, text_to_process, summary_length):
    """Render the full analysis results section."""

    render_badge_row(results, doc_id)
    render_metrics(results)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    render_summary(results)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            render_emotion_bars(results['sentiment']['emotion_scores'])
    with col2:
        with st.container():
            render_text_features(results['features'])

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Export
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"analysis_{doc_id}.json",
            mime="application/json",
            use_container_width=True
        )

    # Translation
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🌐 Translate document")

    if hasattr(nlp, 'supported_languages'):
        col1, col2 = st.columns([2, 1])
        with col1:
            target_language = st.selectbox(
                "Target language",
                list(nlp.supported_languages.keys()),
                key="translate_lang"
            )
        with col2:
            translate_what = st.radio(
                "Translate",
                ["📝 Summary only", "📄 Full document"],
                horizontal=True,
                key="translate_what"
            )

        if st.button("🌐 Translate now", type="primary", use_container_width=True):
            with st.spinner(f"Translating to {target_language}…"):
                lang_code = nlp.supported_languages[target_language]
                text_to_translate = (
                    results['summary']
                    if translate_what == "📝 Summary only"
                    else text_to_process
                )
                translated_text = nlp.translate_text(text_to_translate, lang_code)

            st.success(f"✅ Translated to {target_language}")
            st.markdown(f"""
            <div class="translation-card">
                <div class="translation-content">
                    <h4>🔤 Translation ({target_language})</h4>
                    <p>{translated_text}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label="📥 Download translation",
                data=translated_text,
                file_name=f"translated_{target_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
    else:
        st.info("Translation requires `supported_languages` in `nlp_pipeline.py`.")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color:white;">📄 Document Processing System</h1>
    <p style="font-size:1.2rem; margin-top:0.5rem; color:#e2e8f0;">
        AI-Powered Text Analysis · Summarization · Sentiment Detection · Translation · Batch Processing
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    processing_mode = st.radio(
        "Select mode",
        ["📝 Single Document", "📦 Batch Processing", "📈 Analytics Dashboard"],
        help="Choose how you want to process documents"
    )

    st.divider()

    st.subheader("🔧 NLP Settings")
    summary_length = st.slider(
        "Summary length (sentences)",
        min_value=1, max_value=5, value=3,
        help="Number of sentences in the summary"
    )

    st.divider()

    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", db.get_total_documents())
    with col2:
        st.metric("Avg. polarity", f"{db.get_average_polarity():.3f}")

    st.divider()

    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.success("Cache cleared!")
        time.sleep(1)
        st.rerun()

# ── Single Document ───────────────────────────────────────────────────────────

if processing_mode == "📝 Single Document":
    st.header("📝 Process Single Document")

    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload File", "✏️ Paste Text", "🎯 Try Sample"],
        horizontal=True
    )

    text_to_process = ""

    if input_method == "📄 Upload File":
        uploaded_file = st.file_uploader(
            "Choose a text file", type=['txt'],
            help="Upload a .txt file to process"
        )
        if uploaded_file is not None:
            text_to_process = uploaded_file.read().decode('utf-8')
            st.success(f"✅ File loaded: {uploaded_file.name}")
            with st.expander("📄 Preview text"):
                st.text(text_to_process[:500] + ("…" if len(text_to_process) > 500 else ""))

    elif input_method == "✏️ Paste Text":
        text_to_process = st.text_area(
            "Enter your text:", height=200,
            placeholder="Paste your document text here…"
        )

    else:  # Try Sample
        sample_texts = {
            "Product Review": (
                "The new smartphone is absolutely amazing! The camera quality is outstanding "
                "and the battery life lasts all day. The user interface is smooth and intuitive. "
                "However, the price point is a bit high and the charging cable is too short. "
                "Overall, I'm very satisfied with my purchase and would recommend it to others."
            ),
            "News Article": (
                "Breaking News: Global climate summit reaches historic agreement. World leaders "
                "have committed to reducing carbon emissions by 50% by 2030. Environmental "
                "activists celebrate the landmark decision while industry representatives express "
                "concerns about implementation costs. The agreement includes provisions for "
                "developing nations and establishes a new international monitoring framework."
            ),
            "Business Report": (
                "Q3 earnings report shows mixed results. Revenue increased by 15% year-over-year, "
                "driven by strong performance in the cloud services division. However, hardware "
                "sales declined by 8% due to supply chain disruptions. The company maintains a "
                "positive outlook for Q4, citing new product launches and expanding market share "
                "in emerging economies."
            ),
        }

        selected_sample = st.selectbox("Choose a sample:", list(sample_texts.keys()))
        text_to_process = sample_texts[selected_sample]

        st.markdown("""
        <div class="info-box"><p>📋 Sample text loaded. Click 'Process Document' to analyse.</p></div>
        """, unsafe_allow_html=True)

        with st.expander("📄 Preview sample text"):
            st.text(text_to_process)

    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if text_to_process:
            with st.spinner("🔄 Processing document…"):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                results = nlp.process_document(text_to_process, summary_length)
                doc_id  = db.save_result(results)
                st.session_state['current_results'] = results
                st.session_state['current_text']    = text_to_process
                progress_bar.empty()

            st.markdown(f"""
            <div class="success-box"><p>✅ Document processed successfully!</p></div>
            """, unsafe_allow_html=True)

            render_analysis_results(results, doc_id, text_to_process, summary_length)

        else:
            st.warning("⚠️ Please provide text to process.")

# ── Batch Processing ──────────────────────────────────────────────────────────

elif processing_mode == "📦 Batch Processing":
    st.header("📦 Batch Document Processing")

    st.markdown("""
    <div class="info-box"><p>📦 Upload multiple documents or paste several texts for batch processing.</p></div>
    """, unsafe_allow_html=True)

    batch_input_method = st.radio(
        "Choose batch input method:",
        ["📁 Multiple Files", "✏️ Multiple Texts"],
        horizontal=True
    )

    documents = []

    if batch_input_method == "📁 Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose text files", type=['txt'],
            accept_multiple_files=True,
            help="Select multiple .txt files"
        )
        if uploaded_files:
            for file in uploaded_files:
                documents.append({'name': file.name, 'text': file.read().decode('utf-8')})
            st.success(f"✅ {len(uploaded_files)} files loaded")

    else:
        st.write("Enter multiple texts (one per input box):")
        num_texts = st.number_input("Number of documents", min_value=1, max_value=10, value=2)
        for i in range(num_texts):
            text = st.text_area(
                f"Document {i+1}", height=100,
                key=f"batch_text_{i}",
                placeholder=f"Enter text for document {i+1}…"
            )
            if text:
                documents.append({'name': f"Document {i+1}", 'text': text})

    if documents and st.button("🚀 Process Batch", type="primary", use_container_width=True):
        results_list = []
        progress_bar = st.progress(0)

        for idx, doc in enumerate(documents):
            with st.spinner(f"Processing {doc['name']}…"):
                res = nlp.process_document(doc['text'], summary_length)
                res['document_name'] = doc['name']
                results_list.append(res)
                db.save_result(res)
                progress_bar.progress((idx + 1) / len(documents))

        progress_bar.empty()

        st.markdown(f"""
        <div class="success-box"><p>✅ Processed {len(documents)} documents successfully!</p></div>
        """, unsafe_allow_html=True)

        summary_data = [{
            'Document':  r['document_name'][:30],
            'Sentiment': r['sentiment']['sentiment'],
            'Polarity':  f"{r['sentiment']['polarity']:.3f}",
            'Words':     r['features']['word_count'],
            'Summary':   r['summary'][:100] + "…",
        } for r in results_list]

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

        sentiment_counts = df['Sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#48bb78', 'negative': '#f56565', 'neutral': '#a0aec0'}
        )
        st.plotly_chart(fig, use_container_width=True)

        for result in results_list:
            with st.expander(f"📄 {result['document_name']} — {result['sentiment']['sentiment'].upper()}"):
                st.markdown(f"""
                <div class="summary-card"><p>{result['summary']}</p></div>
                """, unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Polarity",    f"{result['sentiment']['polarity']:.3f}")
                with c2:
                    st.metric("Word Count",  result['features']['word_count'])

# ── Analytics Dashboard ───────────────────────────────────────────────────────

else:
    st.header("📈 Analytics Dashboard")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    analytics = db.get_analytics()

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Documents",  analytics['total_documents'])
    with col2: st.metric("Processed Today",  analytics['today_count'])
    with col3: st.metric("Avg. Polarity",    f"{analytics['avg_polarity']:.3f}")
    with col4: st.metric("Avg. Words / Doc", analytics['avg_word_count'])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sentiment Distribution")
        if analytics['sentiment_distribution']:
            sentiment_df = pd.DataFrame(analytics['sentiment_distribution'])
            if 'sentiment' in sentiment_df.columns and 'count' in sentiment_df.columns:
                fig = px.pie(
                    sentiment_df, values='count', names='sentiment',
                    color='sentiment',
                    color_discrete_map={'positive': '#48bb78', 'negative': '#f56565', 'neutral': '#a0aec0'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available yet.")
        else:
            st.info("No data available. Process some documents first!")

    with col2:
        st.subheader("📈 Processing Timeline")
        if analytics.get('daily_counts'):
            timeline_df = pd.DataFrame(analytics['daily_counts'])
            if not timeline_df.empty:
                fig = px.line(timeline_df, x='date', y='count', title="Documents per Day")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Recent Documents")
    if analytics.get('recent_documents'):
        recent_df = pd.DataFrame(analytics['recent_documents'])
        st.dataframe(recent_df, use_container_width=True)

    if st.button("📥 Export Analytics Report", use_container_width=True):
        report = {'generated_at': datetime.now().isoformat(), 'analytics': analytics}
        st.download_button(
            label="Download Report",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
