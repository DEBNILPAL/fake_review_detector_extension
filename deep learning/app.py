import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob

st.set_page_config(page_title="Backend • Fake Review Detector", page_icon="🕵️", layout="wide")

# Styles
CUSTOM_CSS = """
<style>
  html, body [data-testid="stAppViewContainer"] {
      background: radial-gradient(1000px 600px at 10% 10%, #0b1220 0%, #0a0f1a 35%, #0a0e17 100%) fixed;
  }
  .hero { padding: 1.2rem 1.4rem; border-radius: 14px; background: linear-gradient(135deg, rgba(31,41,55,0.9) 0%, rgba(15,23,42,0.95) 100%); color: #fff; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem; box-shadow: 0 20px 40px rgba(2,6,23,0.35), inset 0 0 0 1px rgba(255,255,255,0.04); }
  .card { padding: 1rem 1.2rem; border-radius: 16px; background: linear-gradient(180deg, rgba(255,255,255,0.75) 0%, rgba(255,255,255,0.6) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(15, 23, 42, 0.12); box-shadow: 0 14px 35px rgba(2,6,23,0.15), 0 2px 6px rgba(2,6,23,0.1); margin-bottom: 1rem; }
  .pill { padding: 0.3rem 0.7rem; border-radius: 999px; background: linear-gradient(90deg, #ede9fe, #e9d5ff); color: #5b21b6; font-weight: 600; font-size: 0.8rem; margin-right: 0.4rem; box-shadow: inset 0 0 0 1px rgba(91,33,182,0.25); }
  .muted { opacity: 0.8; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar config
st.sidebar.title("⚙️ Backend Settings")
default_backend = os.getenv("BACKEND_URL", "http://localhost:3000")
backend_url = st.sidebar.text_input("Backend URL", value=default_backend, help="Your Node/Express server base URL")

if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user": None}

def api_post(path: str, data: dict):
    url = f"{backend_url}{path}"
    r = requests.post(url, json=data, timeout=30)
    r.raise_for_status()
    return r.json()

def api_get(path: str):
    url = f"{backend_url}{path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# Header
st.markdown(
    """
<div class="hero">
  <h2>🕵️ Backend Integration Tester</h2>
  <div class="muted">Calls your Express API endpoints to verify deep learning pipeline + database</div>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs for full flow
tab_auth, tab_predict, tab_analyze_save, tab_submit, tab_dataset = st.tabs([
    "🔐 Auth",
    "🔮 Predict (Backend)",
    "🧾 Analyze & Save",
    "📝 Submit Review",
    "📊 Dataset Analytics",
])

# Auth Tab
with tab_auth:
    st.subheader("Authentication")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'><h4>Sign Up</h4>", unsafe_allow_html=True)
        su_name = st.text_input("Full Name", key="su_name")
        su_email = st.text_input("Email", key="su_email")
        su_pass = st.text_input("Password", type="password", key="su_pass")
        if st.button("Create Account", key="signup_btn"):
            try:
                res = api_post("/api/signup", {"username": su_name, "email": su_email, "password": su_pass})
                st.success("Account created")
                st.json(res)
            except Exception as e:
                st.error(f"Signup failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'><h4>Login</h4>", unsafe_allow_html=True)
        li_email = st.text_input("Email", key="li_email")
        li_pass = st.text_input("Password", type="password", key="li_pass")
        cols = st.columns([1,1,2])
        with cols[0]:
            if st.button("Login", key="login_btn"):
                try:
                    res = api_post("/api/login", {"email": li_email, "password": li_pass})
                    st.session_state.auth = {"logged_in": True, "user": res.get("user")}
                    st.success("Logged in")
                except Exception as e:
                    st.error(f"Login failed: {e}")
        with cols[1]:
            if st.button("Logout", key="logout_btn"):
                st.session_state.auth = {"logged_in": False, "user": None}
        st.caption(f"Status: {'✅ Logged in' if st.session_state.auth['logged_in'] else '❌ Logged out'}")
        if st.session_state.auth.get("user"):
            st.write("User:")
            st.json(st.session_state.auth["user"])
        st.markdown("</div>", unsafe_allow_html=True)

# Predict Tab (Backend /api/predict)
with tab_predict:
    st.subheader("Single Prediction via Backend")
    text = st.text_area("Review Text", height=160, placeholder="Paste a review...")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=5.0, step=0.5)
    with col2:
        product_id = st.text_input("Product ID", value="prod-1")
    with col3:
        reviewer_id = st.text_input("Reviewer ID", value="user-1")
    with col4:
        st.write(" ")
        run = st.button("Predict", type="primary")
    if run:
        try:
            res = api_post("/api/predict", {"text": text, "rating": rating, "productId": product_id, "reviewerId": reviewer_id})
            st.success("Prediction complete")
            st.json(res)
        except Exception as e:
            st.error(f"Predict failed: {e}")

# Analyze & Save Tab (Backend POST /api/analytics)
with tab_analyze_save:
    st.subheader("Analyze and Persist Review")
    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    review = st.text_area("Review", height=160)
    col1, col2, col3 = st.columns(3)
    with col1:
        rating2 = st.number_input("Rating (optional)", min_value=0.0, max_value=5.0, value=5.0, step=0.5, key="rate2")
    with col2:
        product_id2 = st.text_input("Product ID (optional)", value="prod-1")
    with col3:
        reviewer_id2 = st.text_input("Reviewer ID (optional)", value="user-1")
    if st.button("Analyze & Save", type="primary"):
        try:
            payload = {"full_name": full_name, "email": email, "review": review, "rating": rating2, "productId": product_id2, "reviewerId": reviewer_id2}
            res = api_post("/api/analytics", payload)
            st.success("Saved to review_analysis")
            st.json(res)
        except Exception as e:
            st.error(f"Analyze & Save failed: {e}")

# Submit Review Tab (raw to /submit_review)
with tab_submit:
    st.subheader("Submit Review (raw table)")
    user_obj = st.session_state.auth.get("user") or {}
    user_id = st.text_input("User ID", value=str(user_obj.get("id") or user_obj.get("user_id") or "1"))
    client_name = st.text_input("Client Name")
    rating3 = st.number_input("Rating", min_value=0.0, max_value=5.0, value=5.0, step=0.5, key="rate3")
    review_text = st.text_area("Review Text", height=140)
    if st.button("Submit Review"):
        try:
            res = api_post("/submit_review", {"userId": user_id, "client_name": client_name, "rating": rating3, "reviewText": review_text})
            st.success("Review submitted")
            st.json(res)
        except Exception as e:
            st.error(f"Submit failed: {e}")

# Dataset Analytics Tab (GET /api/analytics)
with tab_dataset:
    st.subheader("Dataset Analytics (from Backend)")
    source = st.radio("Source", ["Backend", "Local"], horizontal=True)
    fetch = st.button("Fetch Analytics", type="primary")
    
    # Helpers for local mode
    def _load_local_dataset():
        primary = "reviews_large.csv"
        fallback = os.path.join("dataset", "fake reviews dataset.csv")
        path = primary if os.path.exists(primary) else fallback
        if not os.path.exists(path):
            raise FileNotFoundError("Place reviews_large.csv in this folder or dataset/fake reviews dataset.csv")
        return pd.read_csv(path)

    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # text column
        if "review_text" not in d.columns:
            for col in d.columns:
                if str(col).lower() in {"text", "review", "content", "comment"}:
                    d["review_text"] = d[col].astype(str)
                    break
            else:
                d["review_text"] = ""
        if "rating" not in d.columns:
            d["rating"] = 0
        if "product_id" not in d.columns:
            d["product_id"] = "unknown_product"
        if "reviewer_id" not in d.columns:
            d["reviewer_id"] = "unknown_reviewer"
        d["review_text"] = d["review_text"].astype(str)
        d["review_length"] = d["review_text"].apply(lambda x: len(x))
        def _sent(x):
            try:
                s = TextBlob(x).sentiment
                return pd.Series([s.polarity, s.subjectivity])
            except Exception:
                return pd.Series([0.0, 0.0])
        sent = d["review_text"].apply(_sent)
        d["sentiment_polarity"] = sent.iloc[:,0]
        d["sentiment_subjectivity"] = sent.iloc[:,1]
        # simple placeholders
        d["rating_deviation"] = 0.0
        d["reviewer_history"] = 0.0
        return d
    if fetch:
        try:
            if source == "Backend":
                res = api_get("/api/analytics")
            else:
                # Build a response-like dict from local dataset
                df = _load_local_dataset()
                df_feat = _engineer_features(df)
                # rating counts
                rc = df_feat["rating"].round().value_counts().sort_index().to_dict()
                # length histogram
                counts, bins = np.histogram(df_feat["review_length"], bins=20)
                res = {
                    "total_rows": int(df_feat.shape[0]),
                    "avg_rating": float(df_feat["rating"].mean()),
                    "avg_review_length": float(df_feat["review_length"].mean()),
                    "avg_polarity": float(df_feat["sentiment_polarity"].mean()),
                    "avg_subjectivity": float(df_feat["sentiment_subjectivity"].mean()),
                    "rating_counts": rc,
                    "review_length_hist": {"bins": bins.tolist(), "counts": counts.tolist()},
                }
            st.success("Fetched analytics")

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown("<div class='card'><div><span class='pill'>Rows</span></div>" + f"<h3>{res.get('total_rows', 0)}</h3>" + "</div>", unsafe_allow_html=True)
            with k2:
                st.markdown("<div class='card'><div><span class='pill'>Avg Rating</span></div>" + f"<h3>{res.get('avg_rating', 0):.2f}</h3>" + "</div>", unsafe_allow_html=True)
            with k3:
                st.markdown("<div class='card'><div><span class='pill'>Avg Review Len</span></div>" + f"<h3>{res.get('avg_review_length', 0):.1f}</h3>" + "</div>", unsafe_allow_html=True)
            with k4:
                st.markdown("<div class='card'><div><span class='pill'>Polarity</span></div>" + f"<h3>{res.get('avg_polarity', 0):.3f}</h3>" + "</div>", unsafe_allow_html=True)

            # Charts container
            c1, c2 = st.columns(2)

            # Rating distribution
            rating_counts = res.get("rating_counts") or {}
            if isinstance(rating_counts, dict) and rating_counts:
                rdf = pd.DataFrame({"rating": list(rating_counts.keys()), "count": list(rating_counts.values())})
                try:
                    rdf["rating"] = rdf["rating"].astype(float)
                except Exception:
                    pass
                rdf = rdf.sort_values("rating")
                with c1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.plotly_chart(px.bar(rdf, x="rating", y="count", title="Rating Distribution", template="plotly_white"), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Review length histogram
            rl = res.get("review_length_hist") or {}
            bins = rl.get("bins")
            counts = rl.get("counts")
            if bins and counts and len(bins) == len(counts) + 1:
                # Build ranges as labels
                labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(counts))]
                hdf = pd.DataFrame({"length_bin": labels, "count": counts})
                with c2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.plotly_chart(px.bar(hdf, x="length_bin", y="count", title="Review Length Histogram", template="plotly_white"), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Optional: accuracy by threshold curve, if provided
            eval_obj = res.get("evaluation") or {}
            acc_curve = (eval_obj.get("accuracy_by_threshold") or {})
            thr = acc_curve.get("thresholds")
            acc = acc_curve.get("accuracy")
            if thr and acc and len(thr) == len(acc):
                import pandas as pd
                import plotly.express as px
                adf = pd.DataFrame({"threshold": thr, "accuracy": acc})
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.plotly_chart(px.line(adf, x="threshold", y="accuracy", title="Accuracy by Threshold", markers=True, template="plotly_white"), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Raw JSON expander for debugging
            with st.expander("Raw analytics JSON"):
                st.json(res)
        except Exception as e:
            st.error(f"Fetch failed: {e}")
