"""
live_scoring.py

The inference-time pipeline for the demo app: given a Steam appid, pull
fresh reviews right now, run them through the trained quality filter,
and report a "before filter" vs "after filter" sentiment score.

WHY this is a separate module from the notebook/training script: the
notebook and train_models.py answer "does this approach work, and how
well" using historical, labeled data for 4 specific games. This module
answers a different question at a different time -- "what does this
say about a game a user just typed in, right now" -- using only the
trained artifacts (never re-fitting anything on the fly except the
one thing that has to be per-game: the temporal baseline, see below).

KEY DESIGN DECISION (confirmed with Hikaru, 2026-07-28): the quality
model's temporal feature needs a "what's a normal day" reference for
the game in question. That reference only exists, from training, for
the 4 games in data/raw/. A brand-new game the user searches for has
no such history in this app. Rather than restrict the app to those 4
games, we compute the baseline on the fly from the same batch of
reviews we just pulled (fit_temporal_baseline_from_sample() in
features.py) -- a realistic compromise for a live product, with a
documented trade-off: if the entire freshly-pulled sample happens to
be one long, unbroken bombing event with no quiet days in it, this
baseline won't "know" that, because there's no true pre-bomb reference
for an event you're seeing for the first time. That's a real
limitation, not a hidden one -- see the app's own "how this works"
panel.

WHY no sentiment model is loaded here: Steam's API already returns
each review's own thumbs up/down (`voted_up`). That's real ground
truth, not something that needs predicting -- Section 7 of the
notebook uses it the same way. This app only needs the quality model
to decide which reviews to trust before averaging `voted_up`.
"""

import os
import sys

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

sys.path.append(os.path.dirname(__file__))
from features import drop_non_english_reviews, build_features_for_inference
from collect_reviews import pull_reviews, review_to_row

# This module lives in src/, alongside features.py and collect_reviews.py --
# models/ is the output of train_models.py, one level up at the project root.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLUMNS = [
    "review_length_chars", "review_length_words", "is_very_short",
    "playtime_at_review_hours", "is_low_playtime", "is_duplicate_text",
    "daily_volume_zscore", "is_new_reviewer",
]

# How many recent reviews to pull for a live lookup. Smaller than the
# 4,000-per-game training pulls -- this needs to feel responsive in an
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
    """Loads the artifacts saved by train_models.py once, so a
    Streamlit app can reuse the same loaded model across searches
    instead of re-reading from disk every time."""

    def __init__(self, models_dir=MODELS_DIR):
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.quality_tfidf = joblib.load(os.path.join(models_dir, "quality_tfidf.joblib"))
        self.model = joblib.load(os.path.join(models_dir, "quality_model.joblib"))
        self.feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.joblib"))

    def predict(self, feat_df):
        """feat_df must already have the structural feature columns
        (from build_features_for_inference) and a `review` column."""
        X_structural = self.scaler.transform(feat_df[self.feature_columns].astype(float))
        X_text = self.quality_tfidf.transform(feat_df["review"].fillna(""))
        X_combined = hstack([csr_matrix(X_structural), X_text])
        return self.model.predict(X_combined), self.model.predict_proba(X_combined)[:, 1]


def fetch_reviews_df(appid, game_name, target=LIVE_PULL_TARGET):
    """Pulls up to `target` recent reviews for `appid` via the same
    fetch/pull logic collect_reviews.py uses for training data
    collection (filter="recent", language="english",
    filter_offtopic_activity=0 so bomb-window reviews aren't silently
    hidden), then flattens them into a DataFrame with review_to_row()
    -- identical row shape to the training CSVs."""
    raw_reviews = pull_reviews(appid, game_name, target, window=None)
    if not raw_reviews:
        return pd.DataFrame(), 0
    rows = [review_to_row(r, appid, game_name) for r in raw_reviews]
    df = pd.DataFrame(rows)
    return df, len(raw_reviews)


def score_game(appid, game_name, quality_model, target=LIVE_PULL_TARGET):
    """
    End-to-end: pull -> clean -> feature-engineer -> filter -> score.

    Returns a dict with the before/after percentages, Steam-style
    categories, how many reviews were pulled/dropped/flagged, and a
    small sample of flagged reviews for transparency (so a user isn't
    just told "trust the model" with no way to sanity-check it).
    """
    df_raw, n_pulled = fetch_reviews_df(appid, game_name, target=target)
    if n_pulled == 0:
        return {"error": f"No reviews returned for appid {appid} -- check the ID and try again."}

    df, dropped = drop_non_english_reviews(df_raw)
    n_dropped_non_english = len(dropped)

    if len(df) < 20:
        return {"error": f"Only {len(df)} usable English-language reviews found -- "
                          f"too few for a reliable score. Try a more-reviewed game."}

    feat = build_features_for_inference(df)
    quality_pred, quality_score = quality_model.predict(feat)
    feat = feat.copy()
    feat["quality_pred"] = quality_pred
    feat["quality_score"] = quality_score

    pct_before = feat["voted_up"].mean() * 100
    kept = feat[feat["quality_pred"] == 0]
    pct_after = kept["voted_up"].mean() * 100 if len(kept) else None

    flagged = feat[feat["quality_pred"] == 1].sort_values("quality_score", ascending=False)

    return {
        "appid": appid,
        "game_name": game_name,
        "n_pulled": n_pulled,
        "n_dropped_non_english": n_dropped_non_english,
        "n_scored": len(feat),
        "n_flagged": int(feat["quality_pred"].sum()),
        "pct_flagged": round(feat["quality_pred"].mean() * 100, 1),
        "pct_positive_before": round(pct_before, 1),
        "pct_positive_after": round(pct_after, 1) if pct_after is not None else None,
        "category_before": to_steam_category(pct_before),
        "category_after": to_steam_category(pct_after) if pct_after is not None else None,
        "sample_flagged_reviews": flagged[["review", "voted_up", "quality_score"]].head(10),
    }
