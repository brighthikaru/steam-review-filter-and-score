"""
train_models.py

Trains the quality/review-bomb filter on the project's 4-game dataset
and saves the fitted pieces to disk, so the live demo app can load and
score brand-new games without re-running the notebook.

WHY a separate script, not "just import the notebook": a notebook is
for exploring and justifying decisions with narrative and charts; this
script is for reproducibly producing one artifact (a folder of fitted
model files) that a completely different program (app.py) depends on.
Mixing the two would mean the app breaks any time someone edits the
notebook's markdown or adds an exploratory cell.

WHY the app doesn't also need the sentiment model (LSTM / TF-IDF+LogReg
from Section 6 of the notebook): Steam's API already returns each
review's own thumbs up/down (`voted_up`) alongside the text. The
notebook's Section 7 (Aggregate Scoring) uses that real label as ground
truth for both "before filter" and "after filter" percentages -- it
never needs the sentiment model's *prediction* to compute a score, only
the quality model's flag to decide which reviews to keep. So the only
model this app needs to load is the quality/bomb filter. This keeps the
app lightweight: no TensorFlow/Keras dependency at all.

USAGE:
    python train_models.py
    (writes fitted artifacts to ./models/)
"""

import glob
import os

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import (
    drop_non_english_reviews,
    add_daily_review_count,
    build_features_for_modeling,
)

# This script lives in src/, alongside features.py and collect_reviews.py --
# data/ and the models/ output directory are both one level up, at the
# project root, matching collect_reviews.py's existing OUTPUT_DIR convention.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLUMNS = [
    "review_length_chars", "review_length_words", "is_very_short",
    "playtime_at_review_hours", "is_low_playtime", "is_duplicate_text",
    "daily_volume_zscore", "is_new_reviewer",
]


def load_raw_data():
    raw_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "*.csv")))
    df_raw = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True)
    return df_raw


def train():
    print("Loading raw data...")
    df_raw = load_raw_data()

    print("Dropping non-English-script reviews...")
    df, dropped = drop_non_english_reviews(df_raw)
    print(f"  dropped {len(dropped)} of {len(df_raw)} rows")

    df_raw = df.copy()
    df_raw = add_daily_review_count(df_raw)
    df_raw["label"] = (df_raw["slice"] == "bomb_window").astype(int)

    strat_key = df_raw["game_name"] + "_" + df_raw["label"].astype(str)
    train_raw, test_raw = train_test_split(
        df_raw, test_size=0.3, random_state=42, stratify=strat_key
    )

    print("Building leakage-safe features...")
    train_feat, test_feat, fitted_stats = build_features_for_modeling(train_raw, test_raw)

    X_train_raw = train_feat[FEATURE_COLUMNS].astype(float)
    y_train = train_feat["label"]
    X_test_raw = test_feat[FEATURE_COLUMNS].astype(float)
    y_test = test_feat["label"]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURE_COLUMNS, index=X_train_raw.index)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=FEATURE_COLUMNS, index=X_test_raw.index)

    print("Fitting TF-IDF vectorizer + text-enhanced Logistic Regression (final quality model)...")
    quality_tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
    train_text_tfidf = quality_tfidf.fit_transform(train_feat["review"].fillna(""))
    test_text_tfidf = quality_tfidf.transform(test_feat["review"].fillna(""))

    X_train_combined = hstack([csr_matrix(X_train.values), train_text_tfidf])
    X_test_combined = hstack([csr_matrix(X_test.values), test_text_tfidf])

    log_reg_text = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg_text.fit(X_train_combined, y_train)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    pred = log_reg_text.predict(X_test_combined)
    score = log_reg_text.predict_proba(X_test_combined)[:, 1]
    print(f"  test precision={precision_score(y_test, pred):.3f} recall={recall_score(y_test, pred):.3f} "
          f"f1={f1_score(y_test, pred):.3f} roc_auc={roc_auc_score(y_test, score):.3f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(quality_tfidf, os.path.join(MODELS_DIR, "quality_tfidf.joblib"))
    joblib.dump(log_reg_text, os.path.join(MODELS_DIR, "quality_model.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print(f"Saved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    train()
