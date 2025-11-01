import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import math
import re
import time

# --- Constants ---
NUM_FEATURES = [
    "review_length",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "rating_deviation",
    "reviewer_history",
    "rating",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deep_learning_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
GBC_PATH = os.path.join(BASE_DIR, "gbc_model.joblib")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
DATASET_PRIMARY = os.path.join(BASE_DIR, "reviews_large.csv")
DATASET_FALLBACK = os.path.join(BASE_DIR, "dataset", "fake reviews dataset.csv")


# --- Load artifacts globally (only once) ---
def load_artifacts():
    start = time.time()
    model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    gbc = joblib.load(GBC_PATH) if os.path.exists(GBC_PATH) else None
    tfidf = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None
    print(f"[INFO] Artifacts loaded in {time.time() - start:.2f}s", file=sys.stderr)
    return model, scaler, gbc, tfidf


MODEL, SCALER, GBC, TFIDF = load_artifacts()  # loaded globally


def load_dataset() -> pd.DataFrame | None:
    path = DATASET_PRIMARY if os.path.exists(DATASET_PRIMARY) else DATASET_FALLBACK
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"@[\w_]+", " ", s)
    s = re.sub(r"#[\w_]+", " ", s)
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def engineer_features(df: pd.DataFrame):
    df = df.copy()
    if "review_text" not in df.columns:
        for col in df.columns:
            if str(col).lower() in {"text", "review", "content", "comment"}:
                df["review_text"] = df[col].astype(str)
                break
        else:
            df["review_text"] = ""
    if "rating" not in df.columns:
        df["rating"] = 0
    if "product_id" not in df.columns:
        df["product_id"] = "unknown_product"
    if "reviewer_id" not in df.columns:
        df["reviewer_id"] = "unknown_reviewer"

    df["review_length"] = df["review_text"].apply(lambda x: len(x))
    sent = df["review_text"].apply(lambda x: pd.Series(TextBlob(x).sentiment) if x else pd.Series([0, 0]))
    df["sentiment_polarity"] = sent[0]
    df["sentiment_subjectivity"] = sent[1]
    df["rating_deviation"] = 0
    df["reviewer_history"] = 0
    return df.fillna(0)


def make_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUM_FEATURES].values.astype(float)


def predict_single(text: str, rating: float, product_id: str, reviewer_id: str, context_df: pd.DataFrame | None):
    row = pd.DataFrame({
        "review_text": [text],
        "rating": [rating],
        "product_id": [product_id],
        "reviewer_id": [reviewer_id],
    })
    row_feat = engineer_features(row)

    X_num = make_numeric_matrix(row_feat)
    X_num_scaled = SCALER.transform(X_num)

    clean_tensor = tf.constant([[clean_text(text)]], dtype=tf.string)

    nn_prob = float(MODEL.predict([clean_tensor, X_num_scaled], verbose=0).ravel()[0]) if MODEL is not None else None

    gbc_prob = None
    if GBC is not None:
        try:
            if TFIDF is not None:
                X_tfidf = TFIDF.transform([clean_text(text)])
                X_gbc = hstack([X_tfidf, X_num_scaled])
            else:
                X_gbc = X_num_scaled
            if hasattr(GBC, "predict_proba"):
                gbc_prob = float(GBC.predict_proba(X_gbc)[:, 1][0])
        except Exception:
            gbc_prob = None

    probs = [p for p in [nn_prob, gbc_prob] if p is not None]
    final_prob = float(sum(probs) / len(probs)) if probs else 0.0
    pred = int(final_prob >= 0.5)
    return pred, final_prob, row_feat.iloc[0].to_dict(), {"nn_prob": nn_prob, "gbc_prob": gbc_prob}


def do_predict(args):
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        text = payload.get("text", "")
        rating = float(payload.get("rating", 0))
        product_id = str(payload.get("productId", "prod-1"))
        reviewer_id = str(payload.get("reviewerId", "user-1"))
    except Exception as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        return 1

    if not SCALER or (MODEL is None and GBC is None):
        print(json.dumps({"error": "Model artifacts missing"}))
        return 2

    pred, prob, feats, parts = predict_single(text, rating, product_id, reviewer_id, None)
    out = {
        "prediction": "fake" if pred == 1 else "genuine",
        "prob_fake": prob,
        "features": feats,
        "components": parts,
    }
    print(json.dumps(out))
    return 0


def do_analytics(args):
    df = load_dataset()
    if df is None:
        print(json.dumps({"error": "Dataset not found"}))
        return 3

    try:
        df_feat = engineer_features(df)
        total = int(df_feat.shape[0])
        avg_rating = float(df_feat["rating"].mean())
        avg_len = float(df_feat["review_length"].mean())
        avg_pol = float(df_feat["sentiment_polarity"].mean())
        avg_subj = float(df_feat["sentiment_subjectivity"].mean())

        out = {
            "total_rows": total,
            "avg_rating": avg_rating,
            "avg_review_length": avg_len,
            "avg_polarity": avg_pol,
            "avg_subjectivity": avg_subj,
        }
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"Analytics failed: {e}"}))
        return 4


def main():
    parser = argparse.ArgumentParser(description="Fake Review Inference & Analytics Service")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("predict")
    sub.add_parser("analytics")

    args = parser.parse_args()

    if args.command == "predict":
        rc = do_predict(args)
    elif args.command == "analytics":
        rc = do_analytics(args)
    else:
        print(json.dumps({"error": "Use 'predict' or 'analytics'."}))
        rc = 64

    sys.exit(rc)


if __name__ == "__main__":
    main()
