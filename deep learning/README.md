# Fake Review Detector - Hybrid Deep Learning + Boosting

This project trains a hybrid model to detect fake reviews using:
- A Keras LSTM branch over tokenized review text
- A dense branch over engineered numerical/behavioral features
- A separate Gradient Boosting baseline on numerical features

Artifacts saved after training:
- `deep_learning_model.keras`
- `tokenizer.joblib`
- `scaler.joblib`

## Dataset
- Expected CSV: `reviews_large.csv` in project root.
- If not found, the script will fall back to `dataset/fake reviews dataset.csv`.
- Required columns: `review_text`, `rating`, `product_id`, `reviewer_id`, `is_fake`.

## Features Engineered
- `review_length`: Character length of `review_text`.
- `sentiment_polarity`, `sentiment_subjectivity`: From TextBlob.
- `rating_deviation`: |rating - product average rating|.
- `reviewer_history`: Number of reviews by `reviewer_id`.

## Split
- 20% train, 80% test. Random state 42.

## Requirements
Install dependencies (Windows PowerShell):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train
Run the training script:

```powershell
python train_hybrid_model.py
```

It will print model summary, train with early stopping, evaluate on test data (loss/accuracy), and print classification reports for both the neural network and the Gradient Boosting baseline.

## Output
- Metrics and classification reports printed to console.
- Model and preprocessors saved in the project root as listed above.

## Environment setup (Windows)
- Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Python 3.13 is supported with TensorFlow >= 2.20.0 (already pinned).

## Streamlit UI (Multipage)
- Landing page: `app.py`
- Detector page: `pages/Detector.py` (reuses `streamlit_app.py` for logic)

Run either:

```powershell
streamlit run app.py
```

Click "Get Started" to navigate to the detector UI. If switching pages is not supported in your Streamlit version, use the sidebar to open "Detector".

Alternatively, run the detector directly:

```powershell
streamlit run streamlit_app.py
```

## Dataset schema mapping
The code automatically harmonizes common schemas:
- Text: uses `review_text`, or falls back to `text_`, or any of {`text`, `review`, `content`, `comment`}.
- Label: maps `label` to `is_fake` with `CG`->1, `OR`->0 (also supports `FAKE`/`GENUINE`/`Y`/`N`/`0`/`1`).
- Product/Reviewer: falls back to `category` if `product_id`/`reviewer_id` are absent.
- Missing numeric fields are filled with 0.

## Training progress
- The script prints dataset stats and adds a custom callback that logs:
  - Epoch headers and end-of-epoch metrics (loss/acc and val_loss/val_acc)
  - Batch metrics every 50 batches

## Boosting baseline
- Trains a `GradientBoostingClassifier` on the engineered numerical features and prints a classification report on the same test split.

## Troubleshooting
- Streamlit deprecation: `use_container_width` -> use `width='stretch'` or `'content'`. Current UI may show a warning; it does not affect functionality.
- TensorFlow messages about oneDNN and CPU instructions are informational.
- If you upload a dataset in the UI and see DataFrame ambiguity errors, ensure you are not using Python's `or` on DataFrames; this has been handled in code.
- If analytics show a length mismatch, ensure the same sampled dataframe is used for both the data and the color series; this has been fixed in code.

## Files
- `train_hybrid_model.py` — training pipeline, feature engineering, hybrid Keras model, boosting baseline, artifact saving.
- `streamlit_app.py` — main detector UI (analytics, single/batch prediction) with stylish 3D/glass cards.
- `app.py` — landing page with "Get Started" button that navigates to Detector.
- `pages/Detector.py` — multipage wrapper to reuse `streamlit_app.py`.
- `requirements.txt` — dependencies for Python 3.13 + TensorFlow >= 2.20.
