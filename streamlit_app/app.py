import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
from nlp_pipeline import NLPPipeline
from database import DatabaseManager
from dotenv import load_dotenv
import json

load_dotenv()

st.set_page_config(
    page_title="Document Processing System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_nlp_pipeline():
    return NLPPipeline()

@st.cache_resource
def load_database():
    return DatabaseManager()

nlp = load_nlp_pipeline()
db  = load_database()

# ── Minimal CSS — only things Streamlit cannot do natively ────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    /* Emotion bars */
    .emo-wrap  { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
    .emo-label { font-size:13px; color:#4a5568; width:64px; flex-shrink:0; }
    .emo-track { flex:1; height:9px; background:#e2e8f0; border-radius:5px; overflow:hidden; }
    .emo-bar   { height:100%; border-radius:5px; }
    .emo-val   { font-size:12px; color:#718096; width:34px; text-align:right; flex-shrink:0; }
    /* Chips */
    .chip-wrap { display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }
    .chip      { font-size:11px; padding:2px 9px; border-radius:4px; }
    .chip-kw   { background:#f0f4f8; color:#4a5568; border:1px solid #e2e8f0; }
    .chip-ent  { background:#ebf8ff; color:#2b6cb0; }
    /* Summary / translation accent card */
    .accent-card {
        border-left: 3px solid #667eea;
        background: #f7f8ff;
        padding: 14px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 4px;
    }
    .accent-card p { margin:0; font-size:1.05rem; line-height:1.8; color:#2d3748; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

SENTIMENT_COLOR = {"positive": "#38a169", "negative": "#e53e3e", "neutral": "#718096"}
SENTIMENT_BG    = {"positive": "#f0fff4", "negative": "#fff5f5", "neutral": "#f7fafc"}
EMOTION_COLOR   = {"positive": "#48bb78", "negative": "#fc8181", "neutral": "#a0aec0"}


def _emotion_bars_html(emotion_scores: dict) -> str:
    rows = ""
    for emotion, score in emotion_scores.items():
        pct   = min(max(float(score) * 100, 0), 100)
        color = EMOTION_COLOR.get(emotion, "#a0aec0")
        rows += (
            f'<div class="emo-wrap">'
            f'  <span class="emo-label">{emotion.capitalize()}</span>'
            f'  <div class="emo-track">'
            f'    <div class="emo-bar" style="width:{pct:.0f}%;background:{color}"></div>'
            f'  </div>'
            f'  <span class="emo-val">{score:.2f}</span>'
            f'</div>'
        )
    return rows


def _chips_html(items, extra_class: str) -> str:
    return (
        '<div class="chip-wrap">'
        + "".join(f'<span class="chip {extra_class}">{item}</span>' for item in items)
        + "</div>"
    )


def render_analysis_results(results: dict, doc_id, text_to_process: str):
    sentiment = results["sentiment"]
    features  = results["features"]
    sent_word = sentiment["sentiment"]
    color     = SENTIMENT_COLOR.get(sent_word, "#718096")
    bg        = SENTIMENT_BG.get(sent_word, "#f7fafc")
    lang      = results.get("detected_language_name", "Unknown")

    # ── Status row ────────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([2, 2, 3])
    sc1.info(f"🌐 **Language:** {lang}")
    sc2.markdown(
        f'<div style="background:{bg};border:1px solid {color}55;border-radius:8px;'
        f'padding:10px 14px;font-weight:600;color:{color};font-size:0.95rem;">'
        f'{"🟢" if sent_word=="positive" else "🔴" if sent_word=="negative" else "⚪"}'
        f' {sent_word.capitalize()} sentiment</div>',
        unsafe_allow_html=True
    )
    sc3.caption(f"Document ID: `{doc_id}`")

    st.divider()

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Polarity",     f"{sentiment['polarity']:.3f}")
    m2.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
    m3.metric("Word count",   features["word_count"])
    m4.metric("Compression",  f"{results['metadata']['compression_ratio']:.1%}")

    st.divider()

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Summary")
    st.markdown(
        f'<div class="accent-card"><p>{results["summary"]}</p></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Emotion bars  +  Text features ───────────────────────────────────────
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**😊 Emotion scores**")
        st.markdown(_emotion_bars_html(sentiment["emotion_scores"]), unsafe_allow_html=True)
        st.caption(f"Subjectivity: **{sentiment['subjectivity']:.3f}**")

    with right:
        st.markdown("**🔍 Text features**")
        avg_wps = features["word_count"] / max(1, features["sentence_count"])
        st.markdown(
            f"Sentences: **{features['sentence_count']}** &nbsp;·&nbsp; "
            f"Avg words / sentence: **{avg_wps:.1f}**",
            unsafe_allow_html=True
        )

        if features.get("entities"):
            st.markdown("&nbsp;")
            st.markdown("**Named entities**")
            st.markdown(
                _chips_html(
                    [f"{e} · {l}" for e, l in features["entities"][:6]],
                    "chip-ent"
                ),
                unsafe_allow_html=True
            )

        if features.get("tokens"):
            st.markdown("&nbsp;")
            st.markdown("**Top keywords**")
            st.markdown(
                _chips_html(features["tokens"][:10], "chip-kw"),
                unsafe_allow_html=True
            )

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.download_button(
        label="📥 Download results (JSON)",
        data=json.dumps(results, indent=2, default=str),
        file_name=f"analysis_{doc_id}.json",
        mime="application/json",
    )

    st.divider()

    # ── Translation ───────────────────────────────────────────────────────────
    st.markdown("#### 🌐 Translate document")

    if not hasattr(nlp, "supported_languages"):
        st.info("Translation requires `supported_languages` in `nlp_pipeline.py`.")
        return

    tl1, tl2 = st.columns([2, 1])
    with tl1:
        target_language = st.selectbox(
            "Target language",
            list(nlp.supported_languages.keys()),
            key="translate_lang"
        )
    with tl2:
        translate_what = st.radio(
            "Translate",
            ["Summary only", "Full document"],
            key="translate_what"
        )

    if st.button("🌐 Translate now", type="primary", use_container_width=True):
        with st.spinner(f"Translating to {target_language}…"):
            lang_code       = nlp.supported_languages[target_language]
            source_text     = results["summary"] if translate_what == "Summary only" else text_to_process
            translated_text = nlp.translate_text(source_text, lang_code)

        st.success(f"✅ Translated to {target_language}")
        st.markdown(
            f'<div class="accent-card"><p>{translated_text}</p></div>',
            unsafe_allow_html=True
        )
        st.download_button(
            label="📥 Download translation",
            data=translated_text,
            file_name=f"translation_{target_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1 style="margin:0;color:white;">📄 Document Processing System</h1>
  <p style="font-size:1.1rem;margin-top:0.5rem;color:#e2e8f0;">
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
    )

    st.divider()
    st.subheader("🔧 NLP Settings")
    summary_length = st.slider("Summary length (sentences)", 1, 5, 3)

    st.divider()
    st.subheader("📊 System Status")
    sc1, sc2 = st.columns(2)
    sc1.metric("Documents",    db.get_total_documents())
    sc2.metric("Avg polarity", f"{db.get_average_polarity():.3f}")

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
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file:
            text_to_process = uploaded_file.read().decode("utf-8")
            st.success(f"✅ Loaded: {uploaded_file.name}")
            with st.expander("Preview"):
                st.text(text_to_process[:500] + ("…" if len(text_to_process) > 500 else ""))

    elif input_method == "✏️ Paste Text":
        text_to_process = st.text_area(
            "Enter your text:", height=200,
            placeholder="Paste your document text here…"
        )

    else:
        samples = {
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
        selected = st.selectbox("Choose a sample:", list(samples.keys()))
        text_to_process = samples[selected]
        st.info("📋 Sample loaded — click **Process Document** to analyse.")
        with st.expander("Preview sample"):
            st.text(text_to_process)

    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if not text_to_process.strip():
            st.warning("⚠️ Please provide text to process.")
        else:
            with st.spinner("Analysing document…"):
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    bar.progress(i + 1)
                results = nlp.process_document(text_to_process, summary_length)
                doc_id  = db.save_result(results)
                st.session_state["current_results"] = results
                st.session_state["current_text"]    = text_to_process
                bar.empty()

            st.success("✅ Document processed successfully!")
            render_analysis_results(results, doc_id, text_to_process)

# ── Batch Processing ──────────────────────────────────────────────────────────

elif processing_mode == "📦 Batch Processing":
    st.header("📦 Batch Document Processing")
    st.info("Upload multiple documents or paste several texts for batch processing.")

    batch_method = st.radio(
        "Input method:", ["📁 Multiple Files", "✏️ Multiple Texts"], horizontal=True
    )
    documents = []

    if batch_method == "📁 Multiple Files":
        files = st.file_uploader("Choose text files", type=["txt"], accept_multiple_files=True)
        if files:
            for f in files:
                documents.append({"name": f.name, "text": f.read().decode("utf-8")})
            st.success(f"✅ {len(files)} files loaded")
    else:
        n = st.number_input("Number of documents", 1, 10, 2)
        for i in range(n):
            t = st.text_area(f"Document {i+1}", height=100, key=f"bt_{i}",
                             placeholder=f"Text for document {i+1}…")
            if t:
                documents.append({"name": f"Document {i+1}", "text": t})

    if documents and st.button("🚀 Process Batch", type="primary", use_container_width=True):
        results_list = []
        bar = st.progress(0)
        for idx, doc in enumerate(documents):
            r = nlp.process_document(doc["text"], summary_length)
            r["document_name"] = doc["name"]
            results_list.append(r)
            db.save_result(r)
            bar.progress((idx + 1) / len(documents))
        bar.empty()

        st.success(f"✅ Processed {len(documents)} documents!")

        df = pd.DataFrame([{
            "Document":  r["document_name"][:30],
            "Sentiment": r["sentiment"]["sentiment"],
            "Polarity":  f"{r['sentiment']['polarity']:.3f}",
            "Words":     r["features"]["word_count"],
            "Summary":   r["summary"][:100] + "…",
        } for r in results_list])
        st.dataframe(df, use_container_width=True)

        counts = df["Sentiment"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index, title="Sentiment Distribution",
            color=counts.index,
            color_discrete_map={"positive": "#48bb78", "negative": "#f56565", "neutral": "#a0aec0"}
        )
        st.plotly_chart(fig, use_container_width=True)

        for r in results_list:
            with st.expander(f"📄 {r['document_name']} — {r['sentiment']['sentiment'].upper()}"):
                st.markdown(
                    f'<div class="accent-card"><p>{r["summary"]}</p></div>',
                    unsafe_allow_html=True
                )
                bc1, bc2 = st.columns(2)
                bc1.metric("Polarity",   f"{r['sentiment']['polarity']:.3f}")
                bc2.metric("Word count", r["features"]["word_count"])

# ── Analytics Dashboard ───────────────────────────────────────────────────────

else:
    st.header("📈 Analytics Dashboard")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    a = db.get_analytics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Documents", a["total_documents"])
    c2.metric("Processed Today", a["today_count"])
    c3.metric("Avg Polarity",    f"{a['avg_polarity']:.3f}")
    c4.metric("Avg Words / Doc", a["avg_word_count"])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        if a["sentiment_distribution"]:
            sdf = pd.DataFrame(a["sentiment_distribution"])
            if {"sentiment", "count"}.issubset(sdf.columns):
                fig = px.pie(
                    sdf, values="count", names="sentiment", color="sentiment",
                    color_discrete_map={"positive": "#48bb78", "negative": "#f56565", "neutral": "#a0aec0"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data yet.")
        else:
            st.info("Process some documents first!")

    with col2:
        st.subheader("Processing Timeline")
        if a.get("daily_counts"):
            tdf = pd.DataFrame(a["daily_counts"])
            if not tdf.empty:
                fig = px.line(tdf, x="date", y="count", title="Documents per Day")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Documents")
    if a.get("recent_documents"):
        st.dataframe(pd.DataFrame(a["recent_documents"]), use_container_width=True)

    if st.button("📥 Export Analytics Report", use_container_width=True):
        report = {"generated_at": datetime.now().isoformat(), "analytics": a}
        st.download_button(
            "Download",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
