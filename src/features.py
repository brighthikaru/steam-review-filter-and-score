"""
features.py

Turns raw review rows (as pulled by collect_reviews.py) into named,
justified features for the quality/bomb-detection and sentiment models.

WHY this is its own module, not notebook cells:
    The rubric explicitly grades "separation of the modelling pipeline
    from code for exploring and analysing the data." Feature logic is
    part of the pipeline -- it needs to run identically whether it's
    called from the training notebook, a test script, or (eventually)
    the live demo app. If it lived in notebook cells, the demo app would
    have to reimplement it, and any bug fix would need to happen twice.

Each function below adds one or a small group of related columns and is
documented with the *business* reason the feature exists -- not just
what the code does, but why a real reviewer behaving suspiciously would
show up differently on that feature.
"""

import re

import pandas as pd

# Unicode blocks for scripts that are unambiguously not English: CJK
# (Chinese/Japanese Kanji), Hiragana/Katakana, Hangul (Korean), Cyrillic,
# and Arabic. This deliberately does NOT flag accented Latin letters
# (e.g. "cafe" vs "café") -- those are still readable English-adjacent
# text. It only flags scripts a non-speaker genuinely cannot read.
_NON_ENGLISH_SCRIPT = re.compile(
    r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]"
)


def drop_non_english_reviews(df, text_col="review"):
    """
    WHY: Steam's `language=english` API parameter (used in
    collect_reviews.py) filters on a tag the REVIEWER selects when they
    post -- not on the actual text. People mislabel it, especially
    during review-bomb events where reviews get posted in bulk. Manual
    inspection of the collected data found ~1-2% of rows tagged
    "english" that were actually written in Chinese, Korean, Russian, or
    Arabic.

    This project's scope is English-only because the reviews need to be
    manually verifiable -- if a reviewer can't read the text, they can't
    judge whether a model's sentiment call is right or wrong, and can't
    defend that judgment. So this is a hard drop, not a soft flag: any
    review containing script from a non-Latin block gets removed
    entirely, rather than kept with a "maybe not English" feature.

    Returns (clean_df, dropped_df) so the caller can report/inspect what
    was removed rather than silently losing rows.
    """
    text = df[text_col].fillna("")
    is_non_english = text.apply(lambda t: bool(_NON_ENGLISH_SCRIPT.search(t)))
    return df[~is_non_english].copy(), df[is_non_english].copy()


def add_length_features(df, very_short_chars=20):
    """
    WHY: The single cheapest signal for "did this person put any thought
    into their review" is how much they wrote. One-word reviews, emoji
    spam, and "10/10" tell us almost nothing about the game itself.

    Adds:
        review_length_chars  -- raw character count
        review_length_words  -- raw word count
        is_very_short        -- True if the review is at/under
                                 `very_short_chars` characters (default
                                 20 -- chosen because it's short enough to
                                 exclude "great game 10/10" style
                                 non-reviews while still letting genuinely
                                 short-but-real opinions through)
    """
    df = df.copy()
    review_text = df["review"].fillna("")
    df["review_length_chars"] = review_text.str.len()
    df["review_length_words"] = review_text.str.split().str.len().fillna(0).astype(int)
    df["is_very_short"] = df["review_length_chars"] <= very_short_chars
    return df


def add_playtime_features(df, low_playtime_hours=1.0):
    """
    WHY: Someone leaving a strong opinion after a handful of minutes of
    playtime is a meaningfully different signal than someone doing so
    after dozens of hours. It doesn't prove the review is fake, but
    combined with other signals (very short text, arriving during a
    volume spike) it's a real component of a "low-engagement reviewer"
    profile that shows up disproportionately during bombing events.

    Adds:
        playtime_at_review_hours -- playtime_at_review converted to hours
        is_low_playtime           -- True if under `low_playtime_hours`
                                       (default 1 hour)
    """
    df = df.copy()
    df["playtime_at_review_hours"] = df["playtime_at_review"] / 60
    df["is_low_playtime"] = df["playtime_at_review_hours"] < low_playtime_hours
    return df


def _normalize_text(text):
    """Lowercase, strip punctuation/extra whitespace, for duplicate matching."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_author_features(df, new_reviewer_max_reviews=1):
    """
    WHY: An account with zero or one lifetime reviews posting during a
    volume spike is more consistent with a throwaway/pile-on account than
    an engaged community member. This is a weak signal on its own (lots
    of genuine players are first-time reviewers) but useful in
    combination with the other features above.

    Adds:
        is_new_reviewer -- True if the author has at most
                            `new_reviewer_max_reviews` reviews total
                            (default 1, i.e. this is their first or only
                            review)
    """
    df = df.copy()
    df["is_new_reviewer"] = df["num_reviews"].fillna(0) <= new_reviewer_max_reviews
    return df


def build_features(df):
    """
    EXPLORATION-ONLY convenience function. Adds the four row-local
    features (length, playtime, author) plus a version of the two
    "aggregate statistic" features (duplicate text, temporal z-score)
    computed globally across whatever dataframe you pass in.

    DO NOT use this to prepare features for model training/evaluation.
    Computing those three aggregate features globally means every row
    (including whatever you later call "test data") already influenced
    the statistics used to score it -- e.g. a bomb-window review's own
    contribution to that day's count is baked into the mean/std used to
    judge whether that day looks abnormal. That's leakage: the model
    would look better than it actually is, because test rows quietly
    helped compute their own features.

    This function exists purely for EDA in Section 3/4, where we're just
    describing the dataset, not training or evaluating a model on it.
    For anything that gets split into train/test and evaluated, use
    fit_group_stats() on the training split, then apply_group_features()
    on both splits -- see Section 5.
    """
    df = add_length_features(df)
    df = add_playtime_features(df)
    df = add_author_features(df)

    df = df.copy()
    df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s").dt.date
    df["_normalized_review"] = df["review"].apply(_normalize_text)

    non_empty = df["_normalized_review"] != ""
    dup_counts = (
        df[non_empty]
        .groupby(["game_name", "_normalized_review"])["_normalized_review"]
        .transform("count")
    )
    df["is_duplicate_text"] = False
    df.loc[non_empty, "is_duplicate_text"] = dup_counts.gt(1).values

    daily_counts = (
        df.groupby(["game_name", "review_date"]).size()
        .rename("daily_review_count").reset_index()
    )
    stats = daily_counts.groupby("game_name")["daily_review_count"].agg(["mean", "std"])
    stats["std"] = stats["std"].replace(0, pd.NA)
    daily_counts = daily_counts.merge(stats, on="game_name")
    daily_counts["daily_volume_zscore"] = (
        (daily_counts["daily_review_count"] - daily_counts["mean"]) / daily_counts["std"]
    ).fillna(0.0)
    df = df.merge(
        daily_counts[["game_name", "review_date", "daily_review_count", "daily_volume_zscore"]],
        on=["game_name", "review_date"], how="left",
    )

    df = df.drop(columns=["_normalized_review"])
    return df


def add_daily_review_count(df):
    """
    Computes `review_date` and `daily_review_count` -- how many reviews
    (in our collected data) landed on that calendar day for that game.

    IMPORTANT: call this ONCE, on the full dataset, BEFORE splitting into
    train/test. This is deliberately not folded into fit_group_stats()/
    apply_group_features() below, because of a subtle bug those two
    would otherwise have: if `daily_review_count` were instead computed
    separately within each split (e.g. once on the training rows, once
    on the test rows), a test set that's only ~30% of the full data
    would show artificially low counts for every single day compared to
    a training reference fit on the ~70% split -- making bomb days look
    artificially "quiet" in test regardless of whether they were
    actually abnormal. That's not label leakage, but it is a real
    scale-mismatch bug, and it's just as capable of making a model's
    reported performance meaningless.

    "How many reviews arrived on a given day" is a fact about the
    collected data itself, not something derived from held-out labels --
    in a real production system you'd know this in real time as reviews
    stream in, independent of any later train/test split you might make
    for evaluation. So it's safe, and in fact necessary for correctness,
    to compute this once on the whole dataset up front.
    """
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s").dt.date
    df["daily_review_count"] = df.groupby(["game_name", "review_date"])["review_date"].transform("count")
    return df


def fit_group_stats(train_df):
    """
    Fits every "what's normal for this game" reference statistic from
    the TRAINING split only. Call this once, then pass the result to
    apply_group_features() for both the training and test splits (and,
    later, for any brand-new review pulled in the live demo).

    Assumes `train_df` already has `daily_review_count` attached via
    add_daily_review_count() -- called on the full dataset before the
    train/test split (see that function's docstring for why).

    WHY the reference stats come from `slice == "baseline"` rows only,
    not all of train_df: the whole point of the temporal z-score feature
    is to measure "how far does this differ from normal." If we computed
    "normal" using data that already includes the bombing event, the
    bomb's own volume would inflate what we consider a typical day,
    diluting the exact signal we're trying to detect. Grounding "normal"
    in known-clean (baseline) data keeps the reference uncontaminated.

    NOTE: an earlier version of this pipeline also fitted a
    `language_frequency` reference here (how common each review's
    language was for a given game, on the theory that a language-mix
    shift could signal a coordinated campaign). That feature was removed
    once the project scope was restricted to English-only reviews (see
    `drop_non_english_reviews()` above) -- with only one language left in
    the data, "frequency of this language" is a constant 1.0 for every
    row and carries zero information.

    Duplicate-phrase detection is the exception: coordinated slogans are
    a property of bombing events themselves, so restricting duplicate
    detection to baseline-only data would mean it almost never fires.
    Instead we fit known duplicates from the *entire* training split
    (baseline + bomb-window rows), which still only uses the training
    data -- the test split's own reviews never influence what counts as
    a "known duplicate," so there's no leakage into test evaluation.
    """
    train_df = train_df.copy()
    train_df["_normalized_review"] = train_df["review"].apply(_normalize_text)

    baseline = train_df[train_df["slice"] == "baseline"]

    # --- temporal reference: mean/std of (already-computed, full-scale)
    # daily counts, baseline days only. Dedupe to one row per (game,
    # date) first -- otherwise a single busy day would have its count
    # value counted once per review on that day, badly overweighting it.
    unique_days = baseline.drop_duplicates(subset=["game_name", "review_date"])
    temporal_stats = unique_days.groupby("game_name")["daily_review_count"].agg(["mean", "std"])
    temporal_stats["std"] = temporal_stats["std"].replace(0, pd.NA)

    # --- known duplicate phrases: from the full training split ---
    non_empty = train_df["_normalized_review"] != ""
    dup_counts = train_df[non_empty].groupby(["game_name", "_normalized_review"]).size()
    known_duplicate_keys = set(dup_counts[dup_counts > 1].index)

    return {
        "temporal_stats": temporal_stats,
        "known_duplicate_keys": known_duplicate_keys,
    }


def apply_group_features(df, stats):
    """
    Applies statistics fitted by fit_group_stats() (from a TRAINING
    split) onto `df`, which can be the training split itself, a held-out
    test split, or brand-new reviews in production. This is the
    leakage-safe counterpart to the aggregate portion of build_features().

    Assumes `df` already has `daily_review_count` attached via
    add_daily_review_count() -- called on the full dataset before
    splitting, so the count reflects true volume rather than whatever
    fraction of that day happened to land in this particular split.
    """
    df = df.copy()
    # pandas' merge() resets the index to a fresh 0..n-1 range, which
    # silently breaks anything that relies on the original row index
    # afterwards (e.g. concatenating a train and test dataframe that
    # were both independently reset -- their indices would collide,
    # and .loc[] against them would multiply rows instead of selecting
    # cleanly). Stash the original index as a column and restore it
    # after all the merges below.
    df["_orig_index"] = df.index
    df["_normalized_review"] = df["review"].apply(_normalize_text)

    # --- temporal z-score, using the already-correct daily_review_count ---
    df = df.merge(stats["temporal_stats"], on="game_name", how="left")
    df["daily_volume_zscore"] = ((df["daily_review_count"] - df["mean"]) / df["std"]).fillna(0.0)
    df = df.drop(columns=["mean", "std"])

    # --- duplicate flag, using train-known duplicate phrases only ---
    keys = list(zip(df["game_name"], df["_normalized_review"]))
    known = stats["known_duplicate_keys"]
    df["is_duplicate_text"] = [k in known and k[1] != "" for k in keys]

    df = df.drop(columns=["_normalized_review"])
    df = df.set_index("_orig_index")
    df.index.name = None
    return df


def fit_temporal_baseline_from_sample(df):
    """
    Fits a temporal "what's normal" reference the same way fit_group_stats()
    does, but from a single freshly-pulled sample instead of a labeled
    baseline/bomb-window split.

    WHY this exists separately from fit_group_stats(): that function
    needs `slice == "baseline"` rows to know which days are "known
    clean" -- that label only exists for the 4 games in this project's
    training data. A brand-new game a user searches for in the live app
    has no such label; we don't know in advance which of its days (if
    any) were a bombing event. The practical compromise (Hikaru's
    "option 1", 2026-07-28): treat the whole freshly-pulled sample as
    its own baseline. A day that's a genuine outlier within that sample
    will still show up as a high z-score; the trade-off is that if the
    entire pulled sample is dominated by an ongoing, unbroken bombing
    event with no quiet days to contrast against, this baseline won't
    "know" that -- there's no substitute for a true pre-bomb reference,
    which by definition doesn't exist for a bombing event you're seeing
    for the first time.

    Assumes `df` already has `daily_review_count` attached via
    add_daily_review_count().
    """
    df = df.copy()
    unique_days = df.drop_duplicates(subset=["game_name", "review_date"])
    temporal_stats = unique_days.groupby("game_name")["daily_review_count"].agg(["mean", "std"])
    temporal_stats["std"] = temporal_stats["std"].replace(0, pd.NA)

    df["_normalized_review"] = df["review"].apply(_normalize_text)
    non_empty = df["_normalized_review"] != ""
    dup_counts = df[non_empty].groupby(["game_name", "_normalized_review"]).size()
    known_duplicate_keys = set(dup_counts[dup_counts > 1].index)

    return {
        "temporal_stats": temporal_stats,
        "known_duplicate_keys": known_duplicate_keys,
    }


def build_features_for_modeling(train_df, other_df=None):
    """
    Convenience wrapper for the common case: fit reference statistics on
    `train_df`, then apply row-local + leakage-safe group features to
    `train_df` and (optionally) `other_df` (typically your test split).

    IMPORTANT: both `train_df` and `other_df` must already have
    `daily_review_count` attached (via add_daily_review_count(), called
    on the FULL dataset before splitting) -- see that function's
    docstring for why this can't safely be computed after the split.

    Returns (train_features, other_features, fitted_stats) -- the fitted
    stats are returned too so you can re-apply the exact same reference
    to brand-new data later (e.g. in the live demo app), without
    re-fitting on whatever new reviews come in.
    """
    train_df = add_length_features(train_df)
    train_df = add_playtime_features(train_df)
    train_df = add_author_features(train_df)

    stats = fit_group_stats(train_df)
    train_features = apply_group_features(train_df, stats)

    other_features = None
    if other_df is not None:
        other_df = add_length_features(other_df)
        other_df = add_playtime_features(other_df)
        other_df = add_author_features(other_df)
        other_features = apply_group_features(other_df, stats)

    return train_features, other_features, stats


def build_features_for_inference(df):
    """
    Feature pipeline for the live demo app: a single freshly-pulled
    sample for one game, no train/test split, no baseline/bomb-window
    labels. Combines the row-local features with a self-fit temporal
    baseline (fit_temporal_baseline_from_sample()) rather than the
    labeled-baseline approach used in training -- see that function's
    docstring for why, and the trade-off it accepts.
    """
    df = add_length_features(df)
    df = add_playtime_features(df)
    df = add_author_features(df)
    df = add_daily_review_count(df)

    stats = fit_temporal_baseline_from_sample(df)
    return apply_group_features(df, stats)
