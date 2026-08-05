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
    """Generates a plain-English "Players liked: ..." / "Players disliked:
    ..." summary from the kept reviews, using Falconsai/text_summarization
    -- a t5-small checkpoint (60.5M params, 242MB) fine-tuned specifically
    for summarization. Only ever runs on the ~6 already-filtered kept
    reviews (see score_game), so per-request cost stays small even though
    generation is heavier than a lookup.

    THE ROAD HERE: five pretrained summarizers were tried, not one:
      - t5-small (generic, not summarization-tuned): safe on memory, but
        copy-pasted raw fragments of the source reviews rather than
        genuinely condensing them.
      - bart-large-cnn (~400M params): genuinely coherent prose, but
        OOM-crashed a throwaway Streamlit Community Cloud deployment on
        its very first real request.
      - distilbart-cnn-12-6: dead end before it even loaded -- only ships
        legacy PyTorch weights, no TF/safetensors option, would have
        required adding PyTorch as a second deep-learning framework just
        to convert it.
      - flan-t5-base / flan-t5-small: flan-t5-base hallucinated a detail
        ("graphics") that appeared in none of the source reviews;
        flan-t5-small produced vague, one-sided platitudes that ignored
        genuinely negative reviews in the input entirely.
    Given that history, the deployed version briefly switched to an
    extractive TF-IDF phrase-list approach instead (see git history) --
    it couldn't hallucinate, but couldn't produce real prose either.
    Falconsai/text_summarization was tried next specifically because it's
    the SAME size/memory class as the already-proven-safe generic
    t5-small (so no new OOM risk), but fine-tuned specifically for
    summarization rather than generic multi-task text-to-text -- and
    local testing showed no hallucination across multiple runs.

    THE REMAINING PROBLEM, AND THE FIX: summarizing several reviews in one
    pass (concatenate-then-summarize) reliably dropped at least one review
    from the output entirely -- the model would lock onto whichever single
    review read as most "quotable" (concrete numbers, in testing) and
    ignore the rest, the same failure mode that sank the original generic
    t5-small. The fix is METHOD, not model: summarize each kept review
    INDIVIDUALLY (a task much closer to what this model was actually
    fine-tuned on -- single-document summarization), then join the short
    results. This guarantees every review is represented in the output,
    since no single review can be dropped once each one gets its own
    generation pass.
    """

    MODEL_NAME = "Falconsai/text_summarization"

    def __init__(self, max_new_tokens=45):
        # Imported here, not at module level, so importing live_scoring.py
        # doesn't require transformers/tensorflow just to use QualityModel
        # or SentimentModel -- only SummaryModel actually needs them.
        from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _clean_trim(text):
        """A generation cut off by max_new_tokens ends mid-sentence (e.g.
        "...keep ruining"), which reads badly once several parts are
        joined together. Trims back to the last complete sentence (ending
        in . ! or ?) if one exists in a reasonably-sized fragment;
        otherwise adds a period rather than leaving a dangling clause."""
        text = text.strip()
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct >= 20:
            return text[: last_punct + 1]
        return text if text.endswith((".", "!", "?")) else text + "."

    def _summarize_one(self, review_text):
        inputs = self.tokenizer(
            "summarize: " + review_text, return_tensors="tf", max_length=512, truncation=True
        )
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_new_tokens,
            min_length=6,
            length_penalty=1.5,
            num_beams=4,
        )
        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cleaned = self._clean_trim(raw)
        # Generated text sometimes starts lowercase (the model doesn't
        # reliably capitalize the first word of its own output) -- since
        # several of these get joined into one paragraph, an uncapitalized
        # sentence mid-paragraph reads as a typo.
        return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned

    def _summarize_side(self, reviews):
        parts = [self._summarize_one(r) for r in reviews]
        return " ".join(parts) if parts else None

    def summarize_sides(self, positive_reviews, negative_reviews):
        """Returns (positive_summary, negative_summary) as short
        "Players liked/disliked: ..." prose summaries, or None for a
        side with no usable reviews."""
        pos_texts = [str(r) for r in positive_reviews if str(r).strip()]
        neg_texts = [str(r) for r in negative_reviews if str(r).strip()]
        if not pos_texts and not neg_texts:
            return None, None

        pos_summary = self._summarize_side(pos_texts)
        neg_summary = self._summarize_side(neg_texts)

        positive_summary = f"Players liked: {pos_summary}" if pos_summary else None
        negative_summary = f"Players disliked: {neg_summary}" if neg_summary else None
        return positive_summary, negative_summary


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
