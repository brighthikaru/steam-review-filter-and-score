"""
train_models.py

Trains both deployed models -- the language-based review-quality filter
and the sentiment classifier -- on the project's 6-game dataset, and
saves the fitted pieces to disk so the live demo app can load and score
brand-new games without re-running the notebook.

WHY a separate script, not "just import the notebook": a notebook is
for exploring and justifying decisions with narrative and charts; this
script is for reproducibly producing one artifact (a folder of fitted
model files) that a completely different program (app.py) depends on.
Mixing the two would mean the app breaks any time someone edits the
notebook's markdown or adds an exploratory cell.

WHAT changed from the earlier version of this script (2026-07-29): the
quality filter used to be a structural-features-plus-text model trained
on "was this review posted during one specific game's bombing event."
That approach was dropped -- see the notebook's Section 5.7 for why
(it didn't generalise to games outside its training set). The quality
filter here is now trained on review TEXT ALONE, predicting a
game-agnostic "low-effort" proxy label (very-short OR low-playtime OR
duplicate-text) -- see notebook Section 4.2/5. This also means the app
no longer needs a StandardScaler or structural feature columns at
inference time: the quality model only needs the review's text.

This script also now trains and saves the SENTIMENT model (TF-IDF +
Logistic Regression, notebook Section 6), which the earlier version
deliberately skipped since the app used to rely solely on Steam's own
`voted_up` field as ground truth. The app now shows the sentiment
model's own prediction side-by-side with Steam's real vote (per
Hikaru's request, 2026-07-29), so both models need to ship together.

USAGE:
    python train_models.py
    (writes fitted artifacts to ./models/)
"""

import glob
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from features import (
    drop_non_english_reviews,
    add_length_features,
    add_playtime_features,
    add_author_features,
    _normalize_text,
)

# This script lives in src/, alongside features.py and collect_reviews.py --
# data/ and the models/ output directory are both one level up, at the
# project root, matching collect_reviews.py's existing OUTPUT_DIR convention.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_raw_data():
    raw_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "*.csv")))
    df_raw = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True)
    return df_raw


def add_low_effort_label(df):
    """
    Builds the game-agnostic "low-effort" proxy label used to train the
    quality filter -- see notebook Section 4.2 for the full justification.
    Purely structural (length/playtime/duplicate-text), no dependency on
    any one game's controversy or timing.
    """
    df = add_length_features(df)
    df = add_playtime_features(df)
    df = add_author_features(df)

    df = df.copy()
    df["_normalized_review"] = df["review"].apply(_normalize_text)
    non_empty = df["_normalized_review"] != ""
    dup_counts = (
        df[non_empty].groupby(["game_name", "_normalized_review"])["_normalized_review"]
        .transform("count")
    )
    df["is_duplicate_text"] = False
    df.loc[non_empty, "is_duplicate_text"] = dup_counts.gt(1).values
    df = df.drop(columns=["_normalized_review"])

    df["label"] = (df["is_very_short"] | df["is_low_playtime"] | df["is_duplicate_text"]).astype(int)
    return df


def train_quality_model(df):
    print("\n--- Quality (low-effort review) filter ---")
    quality_df = add_low_effort_label(df)

    strat_key = quality_df["game_name"] + "_" + quality_df["label"].astype(str)
    train_df, test_df = train_test_split(quality_df, test_size=0.3, random_state=42, stratify=strat_key)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(train_df["review"].fillna(""))
    X_test = vectorizer.transform(test_df["review"].fillna(""))
    y_train = train_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    print(f"  test precision={precision_score(y_test, pred):.3f} recall={recall_score(y_test, pred):.3f} "
          f"f1={f1_score(y_test, pred):.3f} roc_auc={roc_auc_score(y_test, score):.3f}")

    return vectorizer, model


def train_sentiment_model(df):
    print("\n--- Sentiment model ---")
    sentiment_df = df[df["review"].fillna("").str.strip() != ""].copy()
    sentiment_df["label"] = sentiment_df["voted_up"].astype(int)

    strat_key = sentiment_df["game_name"] + "_" + sentiment_df["label"].astype(str)
    train_df, test_df = train_test_split(sentiment_df, test_size=0.3, random_state=42, stratify=strat_key)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(train_df["review"])
    X_test = vectorizer.transform(test_df["review"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    print(f"  test precision={precision_score(y_test, pred):.3f} recall={recall_score(y_test, pred):.3f} "
          f"f1={f1_score(y_test, pred):.3f} roc_auc={roc_auc_score(y_test, score):.3f}")

    return vectorizer, model


def train():
    print("Loading raw data...")
    df_raw = load_raw_data()
    print(f"  {df_raw.shape[0]} rows across {df_raw['game_name'].nunique()} games")

    print("Dropping non-English-script reviews...")
    df, dropped = drop_non_english_reviews(df_raw)
    print(f"  dropped {len(dropped)} of {len(df_raw)} rows")

    quality_vectorizer, quality_model = train_quality_model(df)
    sentiment_vectorizer, sentiment_model = train_sentiment_model(df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(quality_vectorizer, os.path.join(MODELS_DIR, "quality_tfidf.joblib"))
    joblib.dump(quality_model, os.path.join(MODELS_DIR, "quality_model.joblib"))
    joblib.dump(sentiment_vectorizer, os.path.join(MODELS_DIR, "sentiment_tfidf.joblib"))
    joblib.dump(sentiment_model, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    print(f"\nSaved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    train()
