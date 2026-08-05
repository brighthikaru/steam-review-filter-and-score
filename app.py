"""
app.py

Streamlit demo: type a game's name (or its Steam appid), pull its most
recent English-language reviews live, filter out likely low-effort/junk
reviews with the trained quality model, and surface the substantive
reviews underneath -- so a player can actually read what people say
about the gameplay before buying, instead of just a blended score. Also
shows the "real sentiment" score before vs. after filtering, plus a
side-by-side comparison of our own sentiment model's prediction against
Steam's real vote.

This is the "end-to-end solution (UI, model, data, infrastructure)"
deliverable -- it reuses the exact same src/features.py and
live_scoring.py pipeline code the notebook's approach is built on, and
the exact models trained by train_models.py, so there is exactly one
implementation of "how do we score this review" in the whole project.

The quality filter is a purely language-based model (no structural
features, no per-game timing baseline) -- see the notebook's Section 5
for why. The app also loads a sentiment model and shows its prediction
next to Steam's real vote, as a built-in accuracy check.

RUN:
    streamlit run app.py
    (requires: pip install streamlit joblib, and train_models.py
    already run once so models/ exists)
"""

import sys

import requests
import streamlit as st

# app.py lives at the project root; the pipeline code lives in src/,
# same convention the notebook uses (sys.path.append("../src") there
# vs. "src" here since this file's own location is the project root).
sys.path.append("src")
from live_scoring import QualityModel, SentimentModel, SummaryModel, score_game

st.set_page_config(page_title="Steam Review Filter & Score", page_icon="🎮", layout="centered")


@st.cache_resource
def load_models():
    return QualityModel(), SentimentModel(), SummaryModel()


def truncate_review(text, limit=400):
    """Trims a review for display without cutting off mid-word or mid-
    sentence with no indication anything was cut. Breaks at the last
    space before `limit` and appends an ellipsis when truncation
    actually happened, so "gameplay is great but..." doesn't just stop
    dead with no signal there's more."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(".,;: ") + "…"


def search_appid_by_name(term):
    """
    Looks up a Steam appid from a free-text game name via Steam's public
    store-search endpoint. Returns a list of (appid, name) candidates.
    Not the same endpoint as appreviews -- this one is purely for
    letting a user type a name instead of memorizing an appid, and is
    best-effort: if it fails (network, rate limit, etc.), the app falls
    back to asking for a numeric appid directly, since that's all the
    scoring pipeline actually needs.
    """
    try:
        resp = requests.get(
            "https://store.steampowered.com/api/storesearch/",
            params={"term": term, "cc": "US", "l": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [(item["id"], item["name"]) for item in items[:8]]
    except requests.RequestException:
        return []


st.title("🎮 Steam Review Filter & Score")
st.caption(
    "A one-word 'good' or 'bad' doesn't tell you why. Pulls a game's recent "
    "English-language reviews, filters out the ones too thin to explain anything, and "
    "surfaces the substantive reviews underneath -- so you can actually read what people "
    "say about the gameplay before you buy, not just a blended score."
)

with st.expander("How this works, and what it can't do"):
    st.markdown(
        """
- Reviews are pulled live from Steam's public API.
- The **quality filter** is a CNN (convolutional neural network) trained on review text
  alone to predict a game-agnostic "low-effort" label (very short, low playtime, or
  duplicate text) -- see the project notebook, Section 5 (ROC-AUC 0.987, consistent
  across seven different game genres in testing -- it beat every classical ML model tried).
  It needs nothing but the text, so it works the same way for any game, not just ones
  it was trained on.
- The **sentiment model** (Section 6) is a Stacking ensemble (Logistic Regression +
  Naive Bayes + Random Forest) that predicts positive/negative from text alone,
  independent of Steam's own vote -- shown here side-by-side with Steam's real vote
  as a built-in accuracy check (they agree 82-96% of the time across the games tested).
- Scoring is scoped to **English-language reviews only** -- both by requesting
  `language=english` from Steam and by dropping any review containing non-Latin
  script, since a review that can't be manually verified isn't defensible. For games
  with a large non-English review base, this score reflects English-speaking
  reviewers specifically, not the full community.
- Negativity driven by something other than gameplay (e.g. a monetisation
  controversy) is treated here as genuine sentiment, not noise to filter out.
- The **"Players liked/disliked"** lines are generated by a small summarization model
  (Falconsai/text_summarization, a t5-small checkpoint fine-tuned for summarization)
  -- each kept review is summarized individually, then joined, so every review gets
  represented instead of the model picking one and ignoring the rest. This model was
  chosen after four other generative summarizers were tried and rejected first (see
  the project README for the full comparison) for either hallucinating details not in
  the source reviews or exceeding the free-tier host's memory budget.
        """
    )

query = st.text_input("Game name or Steam appid", placeholder="e.g. Baldur's Gate 3, or 1086940")

appid = None
game_name = None

if query:
    if query.strip().isdigit():
        appid = int(query.strip())
        game_name = query.strip()
    else:
        candidates = search_appid_by_name(query)
        if candidates:
            options = [f"{name} (appid {aid})" for aid, name in candidates]
            choice = st.selectbox("Select the game:", options)
            idx = options.index(choice)
            appid, game_name = candidates[idx][0], candidates[idx][1]
        else:
            st.warning(
                "Couldn't search by name right now -- try entering the numeric Steam "
                "appid directly instead (find it in the game's Steam store URL)."
            )

if appid and st.button("Score this game", type="primary"):
    with st.spinner(f"Pulling reviews for {game_name or appid}..."):
        quality_model, sentiment_model, summary_model = load_models()
        result = score_game(
            appid, game_name or str(appid), quality_model, sentiment_model, summary_model=summary_model
        )

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader(result["game_name"])

        # Generated by SummaryModel (see live_scoring.py for the full model
        # history) -- the text already reads as a complete sentence
        # ("Players liked: ..."), so no extra "TL;DR" label is needed.
        if result.get("positive_summary") or result.get("negative_summary"):
            sum_col1, sum_col2 = st.columns(2)
            with sum_col1:
                if result.get("positive_summary"):
                    st.success(f"👍 {result['positive_summary']}")
            with sum_col2:
                if result.get("negative_summary"):
                    st.warning(f"👎 {result['negative_summary']}")

        # Steam's own category label (e.g. "Mostly Positive") is the headline
        # here, not the raw percentage -- that's how a player actually reads
        # a Steam score, and it's the framing this app is meant to mirror.
        # The percentage is still shown, just as supporting detail underneath,
        # since a category alone can hide a real swing that doesn't happen to
        # cross a bucket boundary (e.g. 41% -> 48% is still "Mixed" both ways).
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before filtering", result["category_before"])
            st.caption(f"{result['pct_positive_before']}% positive")
        with col2:
            if result["pct_positive_after"] is not None:
                delta = round(result["pct_positive_after"] - result["pct_positive_before"], 1)
                st.metric(
                    "After filtering",
                    result["category_after"],
                    delta=f"{delta:+.1f} pts positive",
                )
                st.caption(f"{result['pct_positive_after']}% positive")
            else:
                st.metric("After filtering", "n/a")
                st.caption("Every pulled review was flagged -- try a larger pull.")

        if (
            result["pct_positive_after"] is not None
            and result["category_after"] != result["category_before"]
        ):
            st.info(
                f"**Category shift:** {result['category_before']} → {result['category_after']} "
                f"once low-effort reviews are filtered out -- a real change in how you'd read "
                f"this game's reputation, not just a small percentage move."
            )

        st.markdown(
            f"Pulled **{result['n_pulled']}** reviews, dropped **{result['n_dropped_non_english']}** "
            f"for non-English content, scored **{result['n_scored']}**, and flagged "
            f"**{result['n_flagged']}** ({result['pct_flagged']}%) as likely low-effort/junk."
        )

        st.divider()
        st.markdown("##### What players are actually saying")
        st.caption(
            "The longest substantive reviews on each side -- the ones with enough detail "
            "(gameplay, bugs, pacing, value) to actually help you decide, not just a score."
        )
        if len(result["sample_kept_reviews"]):
            col_pos, col_neg = st.columns(2)
            kept = result["sample_kept_reviews"]
            with col_pos:
                st.markdown("**👍 Positive**")
                for _, row in kept[kept["voted_up"] == True].iterrows():
                    st.markdown(f"> {truncate_review(row['review'])}")
            with col_neg:
                st.markdown("**👎 Negative**")
                for _, row in kept[kept["voted_up"] == False].iterrows():
                    st.markdown(f"> {truncate_review(row['review'])}")
        else:
            st.caption("No substantive reviews survived filtering for this pull -- try a larger pull or a more-reviewed game.")

        st.divider()
        st.markdown("##### Our sentiment model vs. Steam's real vote")
        st.markdown(
            f"Predicting sentiment from review text alone (no access to Steam's vote), our "
            f"model agrees with the reviewer's actual thumbs up/down **{result['sentiment_agreement_pct']}%** "
            f"of the time for this game. This is a built-in accuracy check: it's the same model "
            f"that would let this pipeline judge sentiment from review text pulled from anywhere -- "
            f"not just a source that happens to provide its own up/down vote like Steam does."
        )

        if len(result["sample_flagged_reviews"]):
            with st.expander("See a sample of the reviews filtered out (so you can judge for yourself)"):
                for _, row in result["sample_flagged_reviews"].iterrows():
                    vote = "👍" if row["voted_up"] else "👎"
                    st.markdown(f"**{vote} (flag confidence {row['quality_score']:.2f})** — {truncate_review(row['review'], limit=300)}")
                    st.divider()

st.caption(
    "Capstone Project — Data Science & AI. Model and methodology detailed in the "
    "accompanying notebook and project documentation."
)
