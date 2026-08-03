"""
live_scoring.py

The inference-time pipeline for the demo app: given a Steam appid, pull
fresh reviews right now, run them through the trained models, and report
a "before filter" vs "after filter" sentiment score, a curated sample of
the substantive (kept) reviews so a player can actually read what people
said about the game, plus a side-by-side comparison between the
sentiment model's own prediction and Steam's real vote.

WHY this is a separate module from the notebook/training script: the
notebook and train_models.py answer "does this approach work, and how
well" using historical data. This module answers a different question
at a different time -- "what does this say about a game a user just
typed in, right now" -- using only the trained artifacts.

The quality filter (see notebook Section 5) is trained on review TEXT
ALONE, predicting a game-agnostic "low-effort" label. It needs nothing
but the review text at inference time, so there's no per-game baseline
to fit, no timing features, and no risk of a baseline being wrong for
a brand-new game.

This module also loads a SENTIMENT model. The app shows the sentiment
model's own text-based prediction side-by-side with Steam's real vote,
both as a demonstration that the model works, and because a model that
predicts sentiment from text alone is portable to any source of review
text, not just Steam's API.
"""

import os
import sys

import joblib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

sys.path.append(os.path.dirname(__file__))
from features import drop_non_english_reviews
from collect_reviews import pull_reviews, review_to_row

# This module lives in src/, alongside features.py and collect_reviews.py --
# models/ is the output of train_models.py, one level up at the project root.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# How many recent reviews to pull for a live lookup. Much smaller than
# the 20,000-per-game training pulls -- this needs to feel responsive in
# an app, not exhaustive like the training data collection was.
LIVE_PULL_TARGET = 1000


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
    so there's no separate vectorizer artifact to load."""

    def __init__(self, models_dir=MODELS_DIR):
        self.model = keras.models.load_model(os.path.join(models_dir, "quality_cnn.keras"))

    def predict(self, review_text):
        # Neither a pandas .values array (NumPy object dtype -- rejected
        # with "Invalid dtype: object") nor a bare Python list
        # ("Unrecognized data type") is reliably accepted as input here
        # across Keras versions. An explicit tf.string tensor is what
        # the model's string Input layer actually expects, and works
        # regardless of Keras/TensorFlow version quirks.
        text_list = review_text.fillna("").astype(str).tolist()
        text_tensor = tf.constant(text_list, dtype=tf.string)
        score = self.model.predict(text_tensor, verbose=0).flatten()
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


class SummaryModel:
    """Generates a short natural-language summary of what the surfaced
    substantive reviews actually say, using a pretrained t5-small
    (TensorFlow backend -- no separate PyTorch dependency needed since
    the app already ships TensorFlow for the quality CNN). Only ever
    runs on the ~6 already-filtered kept reviews (see score_game), not
    the full pulled batch, so inference stays fast even though the
    model itself is a generative one.

    MEMORY, tested live 2026-08-03: deployed and scored two real games
    (Baldur's Gate 3, Helldivers 2) on Streamlit Community Cloud's free
    tier alongside the CNN quality filter and Stacking sentiment model
    -- no crash, no OOM restart. `tensorflow-cpu` on Linux turned out
    much lighter than the `tensorflow` + Windows combination used for
    an earlier local estimate (763-926MB), which had suggested more
    risk than actually exists. Confirmed safe to deploy.

    Summarizes positive and negative kept reviews SEPARATELY (see
    summarize_sides) rather than blending both into one call -- t5-small
    is a small, non-instruction-tuned model, and mixing "great gameplay"
    with "terrible bugs" in one prompt produced incoherent, repetitive
    output in testing (e.g. "it's a powerful and powerful game"). Two
    focused summaries read more coherently than one blended one.
    """

    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, reviews, max_new_tokens=50):
        if not len(reviews):
            return None
        combined = "summarize: " + ". ".join(str(r).rstrip(". ") for r in reviews)
        inputs = self.tokenizer(combined, return_tensors="tf", max_length=512, truncation=True)
        # Beam search + no_repeat_ngram_size directly targets the
        # repetition failure mode greedy decoding showed in testing
        # ("powerful and powerful") -- considers multiple candidate
        # continuations instead of always taking the single most-likely
        # next word, and hard-blocks repeating any 3-word phrase.
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            early_stopping=True,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def summarize_sides(self, positive_reviews, negative_reviews):
        """Returns (positive_summary, negative_summary), each generated
        independently so the model isn't asked to reconcile conflicting
        sentiment in a single pass."""
        return self.summarize(positive_reviews), self.summarize(negative_reviews)


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


def score_game(appid, game_name, quality_model, sentiment_model, target=LIVE_PULL_TARGET, summary_model=None):
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

    `summary_model` is optional (a SummaryModel instance) -- when
    provided, a couple-of-sentences summary is generated from the kept
    reviews. Optional because this is the piece still being validated
    for deployment memory fit; callers that don't pass one just get
    "general_sentiment_summary": None.
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

    # The kept (substantive) reviews are the actual point of the app for a
    # player: text detailed enough to explain *why* someone liked or
    # disliked the game (gameplay mechanics, bugs, pacing, etc.), not just
    # a thumbs up/down. Longer kept reviews are, on average, the ones most
    # likely to carry that detail, so surface the longest few on each side
    # of the vote rather than a random sample -- this is what a player
    # would actually want to read before buying, which the flagged-junk
    # sample (kept purely for filter transparency) was never meant to be.
    kept = df[df["quality_pred"] == 0].copy()
    kept["review_len"] = kept["review"].fillna("").str.len()
    kept_positive = kept[kept["voted_up"] == True].sort_values("review_len", ascending=False).head(3)
    kept_negative = kept[kept["voted_up"] == False].sort_values("review_len", ascending=False).head(3)
    sample_kept_reviews = pd.concat([kept_positive, kept_negative])

    positive_summary, negative_summary = None, None
    if summary_model is not None:
        positive_summary, negative_summary = summary_model.summarize_sides(
            kept_positive["review"].tolist(), kept_negative["review"].tolist()
        )

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
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "sample_kept_reviews": sample_kept_reviews[["review", "voted_up", "review_len"]],
        "sample_flagged_reviews": flagged[["review", "voted_up", "quality_score"]].head(10),
    }
