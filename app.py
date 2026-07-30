"""
app.py

Streamlit demo: type a game's name (or its Steam appid), pull its most
recent English-language reviews live, filter out likely low-effort/junk
reviews with the trained quality model, and show the "real sentiment"
score before vs. after filtering -- plus a side-by-side comparison of
our own sentiment model's prediction against Steam's real vote.

This is the "end-to-end solution (UI, model, data, infrastructure)"
deliverable -- it reuses the exact same src/features.py and
live_scoring.py pipeline code the notebook's approach is built on, and
the exact models trained by train_models.py, so there is exactly one
implementation of "how do we score this review" in the whole project.

WHAT changed (2026-07-29): the quality filter is now a purely
language-based model (no structural features, no per-game timing
baseline) -- see the notebook's Section 5 for why. The app also now
loads a sentiment model and shows its prediction next to Steam's real
vote, rather than relying solely on Steam's vote as before.

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
from live_scoring import QualityModel, SentimentModel, score_game

st.set_page_config(page_title="Steam Review Filter & Score", page_icon="🎮", layout="centered")


@st.cache_resource
def load_models():
    return QualityModel(), SentimentModel()


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
    "Pulls a game's recent English-language reviews, filters out ones our model flags "
    "as low-effort/junk, and shows the real sentiment score before vs. after -- alongside "
    "our own sentiment model's prediction compared to Steam's real vote."
)

with st.expander("How this works, and what it can't do"):
    st.markdown(
        """
- Reviews are pulled live from Steam's public API.
- The **quality filter** is a CNN (convolutional neural network) trained on review text
  alone to predict a game-agnostic "low-effort" label (very short, low playtime, or
  duplicate text) -- see the project notebook, Section 5 (ROC-AUC 0.986, consistent
  across six different game genres in testing -- it beat every classical ML model tried).
  It needs nothing but the text, so it works the same way for any game, not just ones
  it was trained on.
- The **sentiment model** (Section 6) is a Stacking ensemble (Logistic Regression +
  Naive Bayes + Random Forest) that predicts positive/negative from text alone,
  independent of Steam's own vote -- shown here side-by-side with Steam's real vote
  as a built-in accuracy check (they agree 83-96% of the time across the games tested).
- Scoring is scoped to **English-language reviews only** -- both by requesting
  `language=english` from Steam and by dropping any review containing non-Latin
  script, since a review that can't be manually verified isn't defensible. For games
  with a large non-English review base, this score reflects English-speaking
  reviewers specifically, not the full community.
- Negativity driven by something other than gameplay (e.g. a monetisation
  controversy) is treated here as genuine sentiment, not noise to filter out.
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
        quality_model, sentiment_model = load_models()
        result = score_game(appid, game_name or str(appid), quality_model, sentiment_model)

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader(result["game_name"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Before filtering",
                f"{result['pct_positive_before']}% positive",
                help=result["category_before"],
            )
            st.caption(result["category_before"])
        with col2:
            if result["pct_positive_after"] is not None:
                delta = round(result["pct_positive_after"] - result["pct_positive_before"], 1)
                st.metric(
                    "After filtering",
                    f"{result['pct_positive_after']}% positive",
                    delta=f"{delta:+.1f} pts",
                    help=result["category_after"],
                )
                st.caption(result["category_after"])
            else:
                st.metric("After filtering", "n/a")
                st.caption("Every pulled review was flagged -- try a larger pull.")

        st.markdown(
            f"Pulled **{result['n_pulled']}** reviews, dropped **{result['n_dropped_non_english']}** "
            f"for non-English content, scored **{result['n_scored']}**, and flagged "
            f"**{result['n_flagged']}** ({result['pct_flagged']}%) as likely low-effort/junk."
        )

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
            with st.expander("See a sample of flagged reviews (so you can judge for yourself)"):
                for _, row in result["sample_flagged_reviews"].iterrows():
                    vote = "👍" if row["voted_up"] else "👎"
                    st.markdown(f"**{vote} (flag confidence {row['quality_score']:.2f})** — {row['review'][:300]}")
                    st.divider()

st.caption(
    "Capstone Project — Data Science & AI. Model and methodology detailed in the "
    "accompanying notebook and project documentation."
)
