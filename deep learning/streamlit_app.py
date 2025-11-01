import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="wide")

# --------------- Styles ---------------
CUSTOM_CSS = """
<style>
    html, body [data-testid="stAppViewContainer"] {
        background: radial-gradient(1000px 600px at 10% 10%, #0b1220 0%, #0a0f1a 35%, #0a0e17 100%) fixed;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(31,41,55,0.9) 0%, rgba(15,23,42,0.95) 100%);
        color: #fff;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
        box-shadow: 0 20px 40px rgba(2,6,23,0.35), inset 0 0 0 1px rgba(255,255,255,0.04);
    }
    .card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.75) 0%, rgba(255,255,255,0.6) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(15, 23, 42, 0.12);
        box-shadow: 0 14px 35px rgba(2,6,23,0.15), 0 2px 6px rgba(2,6,23,0.1);
        margin-bottom: 1rem;
        transition: transform 180ms ease, box-shadow 180ms ease;
        transform-style: preserve-3d;
    }
    .card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 22px 50px rgba(2,6,23,0.28), 0 4px 10px rgba(2,6,23,0.2);
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #06b6d4, #0ea5e9);
        color: white;
        font-size: 0.75rem;
        margin-left: 0.5rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.15);
    }
    .pill {
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #ede9fe, #e9d5ff);
        color: #5b21b6;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        box-shadow: inset 0 0 0 1px rgba(91,33,182,0.25);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------- Helpers ---------------
NUM_FEATURES = [
    "review_length",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "rating_deviation",
    "reviewer_history",
    "rating",
]

@st.cache_data(show_spinner=False)
def load_default_dataset():
    primary = "reviews_large.csv"
    fallback = os.path.join("dataset", "fake reviews dataset.csv")
    path = primary if os.path.exists(primary) else fallback
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = "deep_learning_model.keras"
    tok_path = "tokenizer.joblib"
    scaler_path = "scaler.joblib"
    model = load_model(model_path) if os.path.exists(model_path) else None
    tokenizer = joblib.load(tok_path) if os.path.exists(tok_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, tokenizer, scaler


def engineer_features(df: pd.DataFrame):
    df = df.copy()
    # Harmonize schema
    if "review_text" not in df.columns:
        if "text_" in df.columns:
            df["review_text"] = df["text_"].astype(str)
        else:
            # Attempt to locate any plausible text column
            cand = None
            for col in df.columns:
                if isinstance(col, str) and col.lower() in {"text", "review", "content", "comment"}:
                    cand = col
                    break
            df["review_text"] = df[cand].astype(str) if cand else ""
    if "rating" not in df.columns:
        df["rating"] = 0
    if "product_id" not in df.columns:
        df["product_id"] = df["category"] if "category" in df.columns else "unknown_product"
    if "reviewer_id" not in df.columns:
        df["reviewer_id"] = df["category"] if "category" in df.columns else "unknown_reviewer"
    if "is_fake" not in df.columns and "label" in df.columns:
        map_vals = {"CG": 1, "OR": 0, "FAKE": 1, "GENUINE": 0, "Y": 1, "N": 0, 1: 1, 0: 0}
        df["is_fake"] = df["label"].map(lambda v: map_vals.get(str(v).strip().upper(), 0)).astype(int)

    df["review_text"] = df["review_text"].astype(str)
    df["review_length"] = df["review_text"].apply(lambda x: len(x) if isinstance(x, str) else 0)

    def _sentiment(x):
        try:
            s = TextBlob(x).sentiment
            return pd.Series([s.polarity, s.subjectivity])
        except Exception:
            return pd.Series([0.0, 0.0])

    sent = df["review_text"].apply(_sentiment)
    df["sentiment_polarity"] = sent.iloc[:, 0]
    df["sentiment_subjectivity"] = sent.iloc[:, 1]

    prod_avg = df.groupby("product_id")["rating"].transform("mean")
    df["rating_deviation"] = (df["rating"] - prod_avg).abs().fillna(0)

    reviewer_counts = df.groupby("reviewer_id")["reviewer_id"].transform("count")
    df["reviewer_history"] = reviewer_counts.fillna(0)

    df = df.fillna(0)
    return df


def make_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUM_FEATURES].values.astype(float)


def predict_single(model, tokenizer, scaler, text: str, rating: float, product_id: str, reviewer_id: str, context_df: pd.DataFrame):
    # Build a one-row dataframe to compute contextual features
    row = pd.DataFrame({
        "review_text": [text],
        "rating": [rating],
        "product_id": [product_id],
        "reviewer_id": [reviewer_id],
    })

    if context_df is not None and {"product_id", "rating"}.issubset(context_df.columns):
        ctx = pd.concat([context_df[["product_id", "rating"]], row[["product_id", "rating"]]])
        ctx["review_text"] = ""
        ctx["reviewer_id"] = ""
        ctx = engineer_features(ctx)
        row_feat = ctx.tail(1)
    else:
        row_feat = engineer_features(row)
        # Without context, rating_deviation and reviewer_history rely only on the row

    X_num = make_numeric_matrix(row_feat)
    X_num_scaled = scaler.transform(X_num)

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=150, padding="post")

    prob = float(model.predict([pad, X_num_scaled], verbose=0).ravel()[0])
    pred = int(prob >= 0.5)
    return pred, prob, row_feat


# --------------- Sidebar ---------------
st.sidebar.title("⚙️ Controls")
with st.sidebar:
    st.markdown("Upload a dataset to power contextual features and batch analysis.")
    user_file = st.file_uploader("Upload CSV (optional)", type=["csv"])

    df_default = load_default_dataset()
    df_upload = None
    if user_file is not None:
        try:
            df_upload = pd.read_csv(user_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    data_source = "Uploaded" if df_upload is not None else ("Default" if df_default is not None else "None")
    st.caption(f"Data source: {data_source}")


# --------------- Header ---------------
st.markdown(
    """
<div class="hero">
  <h2>🕵️ Fake Review Detector</h2>
  <div>Hybrid LSTM + Numerical Features • Trained Keras model • Tokenizer + Scaler</div>
</div>
""",
    unsafe_allow_html=True,
)

model, tokenizer, scaler = load_artifacts()
artifacts_ok = all([model is not None, tokenizer is not None, scaler is not None])

# --------------- Top KPIs ---------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='card'>" + "<div><span class='pill'>Artifacts</span></div>" + f"<h3>{'Loaded' if artifacts_ok else 'Missing'}</h3>" + "</div>", unsafe_allow_html=True)
with col2:
    total_rows = (df_upload if df_upload is not None else df_default).shape[0] if ((df_upload is not None) or (df_default is not None)) else 0
    st.markdown("<div class='card'>" + "<div><span class='pill'>Rows</span></div>" + f"<h3>{total_rows}</h3>" + "</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'>" + "<div><span class='pill'>Vocab Size</span></div>" + f"<h3>{getattr(tokenizer, 'num_words', 0) or 0}</h3>" + "</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='card'>" + "<div><span class='pill'>Seq Len</span></div>" + f"<h3>150</h3>" + "</div>", unsafe_allow_html=True)

# --------------- Tabs ---------------
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔮 Single Prediction", "📥 Batch Prediction"])

# --------------- Analytics Tab ---------------
with tab1:
    st.subheader("Dataset Analytics")
    df_base = df_upload if df_upload is not None else df_default
    if df_base is None:
        st.info("Upload a dataset or place reviews_large.csv in the project root to see analytics.")
    else:
        try:
            df_feat = engineer_features(df_base)
            left, right = st.columns([2, 1])
            with left:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                fig1 = px.histogram(df_feat, x="rating", nbins=10, title="Rating Distribution", template="plotly_white")
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                fig2 = px.histogram(df_feat, x="review_length", nbins=50, title="Review Length", template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("Avg Rating", f"{df_feat['rating'].mean():.2f}")
                st.metric("Avg Review Length", f"{df_feat['review_length'].mean():.1f}")
                st.metric("Avg Polarity", f"{df_feat['sentiment_polarity'].mean():.3f}")
                st.metric("Avg Subjectivity", f"{df_feat['sentiment_subjectivity'].mean():.3f}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            sample_df = df_feat.sample(min(len(df_feat), 5000), random_state=42)
            fig3 = px.scatter(
                sample_df,
                x="sentiment_polarity",
                y="sentiment_subjectivity",
                color=sample_df["rating"].astype(str),
                title="Sentiment Polarity vs Subjectivity (sample)",
                template="plotly_white",
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Analytics failed: {e}")

# --------------- Single Prediction Tab ---------------
with tab2:
    st.subheader("Single Prediction")
    if not artifacts_ok:
        st.warning("Trained artifacts not found. Train the model first to enable predictions.")
    colA, colB = st.columns([2, 1])
    with colA:
        text = st.text_area("Review Text", height=180, placeholder="Paste a review here...")
    with colB:
        rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=5.0, step=0.5)
        product_id = st.text_input("Product ID", value="prod-1")
        reviewer_id = st.text_input("Reviewer ID", value="user-1")
        ctx_df = df_upload if df_upload is not None else df_default
        run = st.button("Predict", type="primary")

    if run and artifacts_ok:
        with st.spinner("Scoring..."):
            pred, prob, feats = predict_single(model, tokenizer, scaler, text, rating, product_id, reviewer_id, ctx_df)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Prediction", "Fake" if pred == 1 else "Genuine", delta=f"Prob {prob:.3f}")
            st.progress(min(max(prob if pred==1 else 1-prob, 0.0), 1.0))
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("Top Features")
            show = feats[NUM_FEATURES].T
            st.dataframe(show.rename(columns={show.columns[0]: "value"}))
            st.markdown("</div>", unsafe_allow_html=True)

# --------------- Batch Prediction Tab ---------------
with tab3:
    st.subheader("Batch Prediction")
    if not artifacts_ok:
        st.warning("Trained artifacts not found. Train the model first to enable predictions.")
    batch_file = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch")
    if batch_file is not None and artifacts_ok:
        try:
            batch_df = pd.read_csv(batch_file)
            # Expect at least: review_text, rating, product_id, reviewer_id
            bdf = engineer_features(batch_df)
            # numeric
            X_num = bdf[NUM_FEATURES].values.astype(float)
            X_num_scaled = scaler.transform(X_num)
            # text
            seqs = tokenizer.texts_to_sequences(bdf["review_text"].astype(str).tolist())
            pad = pad_sequences(seqs, maxlen=150, padding="post")
            probs = model.predict([pad, X_num_scaled], verbose=0).ravel()
            preds = (probs >= 0.5).astype(int)
            out = batch_df.copy()
            out["pred_is_fake"] = preds
            out["pred_prob_fake"] = probs

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("Preview", out.head(20))
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                label="Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = px.histogram(out, x="pred_prob_fake", nbins=30, title="Predicted Fake Probability", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.caption("© 2025 Fake Review Detector • Hybrid Deep Learning UI")
