import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from textblob import TextBlob
import re
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

import joblib

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback


def load_dataset():
    primary = "reviews_large.csv"
    fallback = os.path.join("dataset", "fake reviews dataset.csv")
    path = primary if os.path.exists(primary) else fallback
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find reviews_large.csv. Ensure the dataset is present.")
    df = pd.read_csv(path)
    return df


def engineer_features(df):
    # Harmonize schema to expected columns
    df = df.copy()
    # Map text column
    if "review_text" not in df.columns:
        if "text_" in df.columns:
            df["review_text"] = df["text_"].astype(str)
        else:
            df["review_text"] = ""
    # Ensure rating
    if "rating" not in df.columns:
        df["rating"] = 0
    # Label mapping -> is_fake
    if "is_fake" not in df.columns:
        if "label" in df.columns:
            map_vals = {
                "CG": 1,  # likely computer-generated / fake
                "OR": 0,  # original / genuine
                "FAKE": 1,
                "GENUINE": 0,
                "Y": 1,
                "N": 0,
                1: 1,
                0: 0,
            }
            df["is_fake"] = df["label"].map(lambda v: map_vals.get(str(v).strip().upper(), 0)).astype(int)
        else:
            df["is_fake"] = 0
    # Product/reviewer fallbacks
    if "product_id" not in df.columns:
        df["product_id"] = df["category"] if "category" in df.columns else "unknown_product"
    if "reviewer_id" not in df.columns:
        # Use category as a proxy group for reviewer history if real reviewer_id missing
        df["reviewer_id"] = df["category"] if "category" in df.columns else "unknown_reviewer"

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
    df["rating_deviation"] = (df["rating"] - prod_avg).abs()

    reviewer_counts = df.groupby("reviewer_id")["reviewer_id"].transform("count")
    df["reviewer_history"] = reviewer_counts

    df = df.fillna(0)

    numerical_features = [
        "review_length",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "rating_deviation",
        "reviewer_history",
        "rating",
    ]

    text_data = df["review_text"].values
    labels = df["is_fake"].values.astype(int)

    return df, numerical_features, text_data, labels


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


def load_glove_embeddings(path: str, embedding_dim: int) -> dict:
    if not os.path.exists(path):
        return {}
    embeddings_index = {}
    try:
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                if coefs.shape[0] == embedding_dim:
                    embeddings_index[word] = coefs
    except Exception:
        return {}
    return embeddings_index


def build_embedding_matrix(vocab: list[str], embeddings_index: dict, embedding_dim: int, max_vocab_size: int) -> np.ndarray:
    # Keras TextVectorization vocab[0] = "" (padding), vocab[1] = "[UNK]"
    matrix = np.random.normal(loc=0.0, scale=0.05, size=(max_vocab_size, embedding_dim)).astype("float32")
    matrix[0] = np.zeros(embedding_dim, dtype="float32")
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    for tok, idx in token_to_idx.items():
        if idx >= max_vocab_size:
            continue
        vec = embeddings_index.get(tok)
        if vec is not None:
            matrix[idx] = vec
    return matrix


def prepare_splits(df, numerical_features, text_data, labels):
    X_num = df[numerical_features].values.astype(float)
    X_train_num, X_test_num, X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_num, text_data, labels, test_size=0.2, random_state=42, stratify=labels if len(np.unique(labels))>1 else None
    )
    return X_train_num, X_test_num, X_train_text, X_test_text, y_train, y_test


def preprocess_numeric(X_train_num, X_test_num):
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    return scaler, X_train_num_scaled, X_test_num_scaled


def preprocess_text(X_train_text, X_test_text):
    # Clean texts
    X_train_clean = np.array([clean_text(t) for t in X_train_text], dtype=object)
    X_test_clean = np.array([clean_text(t) for t in X_test_text], dtype=object)

    # TextVectorization inside the model will handle tokenization; just return cleaned arrays
    # Provide config values for model building
    MAX_VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 200
    return X_train_clean, X_test_clean, MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH


def build_model(max_vocab_size, max_seq_len, num_num_features, embedding_dim=64):
    # Text branch: raw strings -> TextVectorization -> Embedding -> LSTM
    input_text = Input(shape=(1,), dtype=tf.string, name="text_input")
    tv = TextVectorization(
        max_tokens=max_vocab_size,
        output_sequence_length=max_seq_len,
        standardize=None,
        name="tv_layer",
    )
    # Note: we'll adapt tv outside the model after building it
    x = tv(input_text)
    embedding = Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, name="embedding")(x)
    lstm_layer = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedding)
    text_dense = Dense(32, activation="relu")(lstm_layer)

    input_num = Input(shape=(num_num_features,), name="num_input")
    num_dense = Dense(32, activation="relu")(input_num)

    merged = concatenate([text_dense, num_dense])
    dropout = Dropout(0.5)(merged)
    output_dense = Dense(16, activation="relu")(dropout)
    output = Dense(1, activation="sigmoid")(output_dense)

    model = Model(inputs=[input_text, input_num], outputs=output)
    return model


def train_and_evaluate():
    df = load_dataset()
    df, numerical_features, text_data, labels = engineer_features(df)

    X_train_num, X_test_num, X_train_text, X_test_text, y_train, y_test = prepare_splits(
        df, numerical_features, text_data, labels
    )

    scaler, X_train_num_scaled, X_test_num_scaled = preprocess_numeric(X_train_num, X_test_num)
    X_train_clean, X_test_clean, max_vocab, max_seq_len = preprocess_text(
        X_train_text, X_test_text
    )

    # Use 100d GloVe if available
    GLOVE_DIR = os.path.join("embeddings")
    os.makedirs(GLOVE_DIR, exist_ok=True)
    GLOVE_PATHS = [
        os.path.join(GLOVE_DIR, "glove.6B.100d.txt"),
        os.path.join(".", "glove.6B.100d.txt"),
    ]
    embedding_dim = 100
    model = build_model(max_vocab, max_seq_len, X_train_num_scaled.shape[1], embedding_dim=embedding_dim)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Print high-level dataset stats for visibility
    print(f"Training samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"Numeric features: {X_train_num_scaled.shape[1]} | Vocab: {max_vocab} | Seq len: {max_seq_len}")

    class PrintProgress(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\n===== Epoch {epoch+1}/{self.params.get('epochs', '?')} =====")
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            msg = (
                f"Epoch {epoch+1} end - "
                f"loss: {logs.get('loss'):.4f} | acc: {logs.get('accuracy'):.4f} | "
                f"val_loss: {logs.get('val_loss'):.4f} | val_acc: {logs.get('val_accuracy'):.4f}"
            )
            print(msg)
        def on_train_batch_end(self, batch, logs=None):
            # Throttle batch logs to avoid excessive spam
            if batch % 50 == 0:
                logs = logs or {}
                print(
                    f"Batch {batch} - loss: {logs.get('loss'):.4f}, acc: {logs.get('accuracy'):.4f}"
                )

    # Adapt TextVectorization on training corpus
    tv_layer = model.get_layer("tv_layer")
    tv_layer.adapt(tf.data.Dataset.from_tensor_slices(X_train_clean).batch(64))

    # Try to load GloVe and set embedding weights
    glove_path = next((p for p in GLOVE_PATHS if os.path.exists(p)), None)
    if glove_path is not None:
        print(f"Loading GloVe embeddings from: {glove_path}")
        embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
        vocab = tv_layer.get_vocabulary()
        emb_matrix = build_embedding_matrix(vocab, embeddings_index, embedding_dim, max_vocab)
        emb_layer = model.get_layer("embedding")
        try:
            emb_layer.set_weights([emb_matrix])
            # Keep trainable to allow fine-tuning
            emb_layer.trainable = True
            print("Initialized embedding layer with GloVe vectors.")
        except Exception as e:
            print(f"Warning: Failed to set GloVe weights: {e}")
    else:
        print("GloVe file not found; using randomly initialized embeddings.")

    history = model.fit(
        [X_train_clean, X_train_num_scaled],
        y_train,
        batch_size=32,
        epochs=20,
        validation_data=([X_test_clean, X_test_num_scaled], y_test),
        callbacks=[es, PrintProgress()],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate([X_test_clean, X_test_num_scaled], y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    y_pred_proba = model.predict([X_test_clean, X_test_num_scaled]).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("Neural Net Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Build TF-IDF features on cleaned text for GBC
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    # Combine numeric + TF-IDF features for GBC
    X_train_gbc = hstack([X_train_tfidf, X_train_num_scaled])
    X_test_gbc = hstack([X_test_tfidf, X_test_num_scaled])

    # Gradient Boosting with hyperparameter search
    gbc = GradientBoostingClassifier()
    param_dist = {
        "n_estimators": [50, 100, 150, 200, 300],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_depth": [1, 2, 3],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 4],
    }
    gbc_search = RandomizedSearchCV(
        estimator=gbc,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    gbc_search.fit(X_train_gbc, y_train)
    best_gbc = gbc_search.best_estimator_
    print(f"Best GBC params: {gbc_search.best_params_}")
    gbc_pred = best_gbc.predict(X_test_gbc)
    print("Gradient Boosting Classification Report (numerical features, tuned):")
    print(classification_report(y_test, gbc_pred, digits=4))

    # Save artifacts
    model.save("deep_learning_model.keras")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(best_gbc, "gbc_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")

    print("Saved artifacts: deep_learning_model.keras, scaler.joblib, gbc_model.joblib, tfidf_vectorizer.joblib")


if __name__ == "__main__":
    train_and_evaluate()
