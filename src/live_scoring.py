"""
live_scoring.py

The inference-time pipeline for the demo app: given a Steam appid, pull
fresh reviews right now, run them through the trained models, and report
a "before filter" vs "after filter" sentiment score, plus a side-by-side
comparison between the sentiment model's own prediction and Steam's real
vote.

WHY this is a separate module from the notebook/training script: the
notebook and train_models.py answer "does this approach work, and how
well" using historical data. This module answers a different question
at a different time -- "what does this say about a game a user just
typed in, right now" -- using only the trained artifacts.

WHAT changed from the earlier version of this script (2026-07-29): the
quality filter used to need a per-game temporal baseline (a "what's
normal daily review volume" reference), computed on the fly for
new games since only the 4 training games had one from labeled data.
That whole design is gone now -- the quality filter (see notebook
Section 5) is trained on review TEXT ALONE, predicting a game-agnostic
"low-effort" label. It needs nothing but the review text at inference
time, so there's no per-game baseline to fit, no timing features, and
no risk of the baseline being wrong for a brand-new game.

This module also now loads a SENTIMENT model (previously skipped,
since the app relied solely on Steam's own `voted_up` as ground
truth). The app now shows the sentiment model's own text-based
prediction side-by-side with Steam's real vote -- per Hikaru's
request, 2026-07-29 -- both as a demonstration that the model works,
and because a model that predicts sentiment from text alone is
portable to any source of review text, not just Steam's API.
"""

import os
import sys

import joblib
import pandas as pd
from tensorflow import keras

sys.path.append(os.path.dirname(__file__))
from features import drop_non_english_reviews
from collect_reviews import pull_reviews, review_to_row

# This module lives in src/, alongside features.py and collect_reviews.py --
# models/ is the output of train_models.py, one level up at the project root.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# How many recent reviews to pull for a live lookup. Smaller than the
# 8,000-per-game training pulls -- this needs to feel responsive in an
# app, not exhaustive like the training data collection was.
LIVE_PULL_TARGET = 400


def to_steam_category(pct_positive):
    """Same rough bucketing used in the notebook's Section 7 -- kept
    identical here so the app's categories match what's discussed in
    the notebook/documentation."""
    if pct_positive is None:
        return None
    if pct_positive >= 95:
        return "Overwhelmingly Positive"
    if pct_positive >= 80:
        return "Very Positive"
    if pct_positive >= 70:
        return "Mostly Positive"
    if pct_positive >= 40:
        return "Mixed"
    if pct_positive >= 20:
        return "Mostly Negative"
    return "Overwhelmingly Negative"


class QualityModel:
    """Loads the CNN low-effort/junk quality filter saved by
    train_models.py. Needs nothing but review text -- the
    TextVectorization vocabulary is baked into the saved Keras model,
    so there's no separate vectorizer artifact to load (unlike the
    earlier TF-IDF + Logistic Regression version)."""

    def __init__(self, models_dir=MODELS_DIR):
        self.model = keras.models.load_model(os.path.join(models_dir, "quality_cnn.keras"))

    def predict(self, review_text):
        text_values = review_text.fillna("").values
        score = self.model.predict(text_values, verbose=0).flatten()
        pred = (score >= 0.5).astype(int)
        return pred, score


class SentimentModel:
    """Loads the text-based sentiment classifier saved by
    train_models.py. Predicts positive/negative from review text alone
    -- independent of Steam's own `voted_up`, so this same model could
    in principle score reviews from a source that doesn't provide a
    thumbs-up vote at all."""

    def __init__(self, models_dir=MODELS_DIR):
        self.vectorizer = joblib.load(os.path.join(models_dir, "sentiment_tfidf.joblib"))
        self.model = joblib.load(os.path.join(models_dir, "sentiment_model.joblib"))

    def predict(self, review_text):
        X = self.vectorizer.transform(review_text.fillna(""))
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]


def fetch_reviews_df(appid, game_name, target=LIVE_PULL_TARGET):
    """Pulls up to `target` recent reviews for `appid` via the same
    fetch/pull logic collect_reviews.py uses for training data
    collection (filter="recent", language="english"), then flattens
    them into a DataFrame with review_to_row() -- identical row shape
    to the training CSVs."""
    raw_reviews = pull_reviews(appid, game_name, target)
    if not raw_reviews:
        return pd.DataFrame(), 0
    rows = [review_to_row(r, appid, game_name) for r in raw_reviews]
    df = pd.DataFrame(rows)
    return df, len(raw_reviews)


def score_game(appid, game_name, quality_model, sentiment_model, target=LIVE_PULL_TARGET):
    """
    End-to-end: pull -> clean -> filter -> score, both models.

    Returns a dict with:
      - the before/after "real sentiment" percentages (Steam's own
        voted_up, filtered by the quality model) and Steam-style
        categories,
      - a side-by-side comparison of the sentiment model's own
        prediction against Steam's real vote (agreement rate),
      - how many reviews were pulled/dropped/flagged,
      - a small sample of flagged reviews for transparency.
    """
    df_raw, n_pulled = fetch_reviews_df(appid, game_name, target=target)
    if n_pulled == 0:
        return {"error": f"No reviews returned for appid {appid} -- check the ID and try again."}

    df, dropped = drop_non_english_reviews(df_raw)
    n_dropped_non_english = len(dropped)

    if len(df) < 20:
        return {"error": f"Only {len(df)} usable English-language reviews found -- "
                          f"too few for a reliable score. Try a more-reviewed game."}

    df = df.copy()
    quality_pred, quality_score = quality_model.predict(df["review"])
    df["quality_pred"] = quality_pred
    df["quality_score"] = quality_score

    sentiment_pred, sentiment_score = sentiment_model.predict(df["review"])
    df["predicted_sentiment"] = sentiment_pred
    df["sentiment_confidence"] = sentiment_score

    pct_before = df["voted_up"].mean() * 100
    kept = df[df["quality_pred"] == 0]
    pct_after = kept["voted_up"].mean() * 100 if len(kept) else None

    agreement_pct = (df["predicted_sentiment"] == df["voted_up"].astype(int)).mean() * 100

    flagged = df[df["quality_pred"] == 1].sort_values("quality_score", ascending=False)

    return {
        "appid": appid,
        "game_name": game_name,
        "n_pulled": n_pulled,
        "n_dropped_non_english": n_dropped_non_english,
        "n_scored": len(df),
        "n_flagged": int(df["quality_pred"].sum()),
        "pct_flagged": round(df["quality_pred"].mean() * 100, 1),
        "pct_positive_before": round(pct_before, 1),
        "pct_positive_after": round(pct_after, 1) if pct_after is not None else None,
        "category_before": to_steam_category(pct_before),
        "category_after": to_steam_category(pct_after) if pct_after is not None else None,
        "sentiment_agreement_pct": round(agreement_pct, 1),
        "sample_flagged_reviews": flagged[["review", "voted_up", "quality_score"]].head(10),
    }
