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
duplicate-text) -- see notebook Section 4.2/5.

WHAT changed again (2026-07-30): the quality filter is now a CNN
(Keras/TensorFlow), not TF-IDF + Logistic Regression. A comparison in
the notebook (Section 5) found a CNN substantially outperforms every
classical model tried (ROC-AUC 0.986 vs. 0.962 for the best classical
candidate), consistent across all six game genres -- a real, checked
result, not just a higher headline number. The CNN's TextVectorization
layer is trained as part of the model itself, so there's no separate
TF-IDF vectorizer artifact to save for the quality filter anymore --
just one Keras model that takes raw review text directly.

The sentiment model stays classical (Section 6) -- deep learning's
extra capacity doesn't pay off for sentiment at this data volume,
unlike the quality filter above.

WHAT changed again (2026-07-30, later same day): the sentiment model
is now a Stacking ensemble (Logistic Regression + Naive Bayes + Random
Forest -> Logistic Regression meta-learner), not plain Logistic
Regression. Stacking scores higher on every metric and was initially
passed over on the assumption its ~30x longer training time was a real
deployment cost -- that assumption was wrong, and was corrected after
actually benchmarking per-request inference time (~24ms vs ~0.1ms,
both negligible next to the multi-second Steam API call already in
the pipeline) and serialized size (~3.2MB, trivial next to the ~6.6MB
CNN already shipped). See train_sentiment_model()'s docstring below.

USAGE:
    python train_models.py
    (writes fitted artifacts to ./models/)
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

from features import (
    drop_non_english_reviews,
    add_length_features,
    add_playtime_features,
    add_author_features,
    _normalize_text,
)

VOCAB_SIZE = 8000
SEQ_LEN = 60

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
    """
    Trains the CNN quality filter (notebook Section 5): a Keras
    Sequential model whose first layer is a TextVectorization layer, so
    the fitted vocabulary is saved as part of the model itself -- no
    separate vectorizer artifact needed. Takes raw review text strings
    directly as input.
    """
    print("\n--- Quality (low-effort review) filter: CNN ---")
    quality_df = add_low_effort_label(df)

    strat_key = quality_df["game_name"] + "_" + quality_df["label"].astype(str)
    train_df, test_df = train_test_split(quality_df, test_size=0.3, random_state=42, stratify=strat_key)

    train_text = train_df["review"].fillna("").values
    test_text = test_df["review"].fillna("").values
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    tf.random.set_seed(42)
    text_vectorizer = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=SEQ_LEN)
    text_vectorizer.adapt(train_text)

    model = keras.Sequential([
        keras.Input(shape=(1,), dtype=tf.string),
        text_vectorizer,
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=False),
        layers.Conv1D(64, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)

    model.fit(
        train_text, y_train,
        validation_split=0.1, epochs=10, batch_size=128,
        class_weight=class_weight, callbacks=[early_stop], verbose=2,
    )

    score = model.predict(test_text, verbose=0).flatten()
    pred = (score >= 0.5).astype(int)
    print(f"  test precision={precision_score(y_test, pred):.3f} recall={recall_score(y_test, pred):.3f} "
          f"f1={f1_score(y_test, pred):.3f} roc_auc={roc_auc_score(y_test, score):.3f}")

    return model


def train_sentiment_model(df):
    """
    Trains the deployed sentiment model: a Stacking ensemble (Logistic
    Regression + Naive Bayes + Random Forest, combined by a Logistic
    Regression meta-learner) over the same TF-IDF features a plain
    Logistic Regression would use.

    WHY Stacking, not plain Logistic Regression: Stacking scores higher
    on every metric (F1 0.947 vs 0.927, and -- the more user-facing
    number -- 91.2% vs 88.4% agreement with Steam's own vote). Logistic
    Regression was deployed initially instead, on the assumption that
    Stacking's ~30x longer *training* time meant a real deployment cost.
    That assumption was wrong and was corrected after actually
    benchmarking it: training time is a one-time, offline cost that
    never touches a live request. What actually matters for the app --
    inference time on a real request (400 reviews) and serialized model
    size -- turned out to be a non-issue: ~24ms vs ~0.1ms (both
    imperceptible next to the multi-second Steam API call that already
    happens per lookup), and ~3.2MB vs ~40KB (trivial next to the ~6.6MB
    CNN quality filter already shipped). Once measured rather than
    assumed, there was no real reason not to take the accuracy gain.
    """
    print("\n--- Sentiment model (Stacking ensemble) ---")
    sentiment_df = df[df["review"].fillna("").str.strip() != ""].copy()
    sentiment_df["label"] = sentiment_df["voted_up"].astype(int)

    strat_key = sentiment_df["game_name"] + "_" + sentiment_df["label"].astype(str)
    train_df, test_df = train_test_split(sentiment_df, test_size=0.3, random_state=42, stratify=strat_key)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(train_df["review"])
    X_test = vectorizer.transform(test_df["review"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    model = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ("nb", MultinomialNB()),
            ("rf", RandomForestClassifier(n_estimators=150, class_weight="balanced", max_depth=12,
                                           min_samples_leaf=5, n_jobs=-1, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1,
    )
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

    quality_model = train_quality_model(df)
    sentiment_vectorizer, sentiment_model = train_sentiment_model(df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Keras' own format -- a single .keras file, vocabulary and weights
    # both included. No separate vectorizer artifact for the quality
    # model anymore (see train_quality_model's docstring).
    quality_model.save(os.path.join(MODELS_DIR, "quality_cnn.keras"))
    joblib.dump(sentiment_vectorizer, os.path.join(MODELS_DIR, "sentiment_tfidf.joblib"))
    joblib.dump(sentiment_model, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    print(f"\nSaved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    train()
