"""
app.py

Streamlit demo: type a game's name (or its Steam appid), pull its most
recent English-language reviews live, filter out likely low-quality /
review-bomb reviews with the trained model, and show the filtered
sentiment score next to the raw one.

This is the "end-to-end solution (UI, model, data, infrastructure)"
deliverable -- it reuses the exact same src/features.py pipeline code
the notebook does, and the exact model trained by train_models.py, so
there is exactly one implementation of "how do we compute this
feature" or "how do we score this review" in the whole project.

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
from live_scoring import QualityModel, score_game, to_steam_category

st.set_page_config(page_title="Steam Review Filter & Score", page_icon="🎮", layout="centered")


@st.cache_resource
def load_model():
    return QualityModel()


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
    "Pulls a game's recent English-language reviews, filters out reviews our model "
    "flags as low-quality or review-bomb-driven, and compares the sentiment score "
    "before vs. after filtering."
)

with st.expander("How this works, and what it can't do"):
    st.markdown(
        """
- Reviews are pulled live from Steam's public API (`filter_offtopic_activity=0`,
  so review-bomb reviews aren't silently hidden the way Steam's default view hides them).
- The quality filter is a Logistic Regression trained on structural features
  (length, playtime, duplicate text, posting-time clustering) **and** the review
  text itself (TF-IDF) -- see the project notebook, Section 5, for how it was
  built and validated (ROC-AUC 0.847 on held-out test data).
- Scoring is scoped to **English-language reviews only** -- both by requesting
  `language=english` from Steam and by dropping any review containing non-Latin
  script, since a review that can't be manually verified isn't defensible.
  For games with a large non-English review base, this score reflects
  English-speaking reviewers specifically, not the full community.
- For games outside the notebook's original 4-game training set (which is
  every game you search here), the "what's a normal day" baseline is computed
  from the batch of reviews just pulled, not a separately verified pre-bombing
  period. If a game is in the middle of an unbroken, ongoing review-bomb with
  no quiet days in the pulled sample, the filter has no "quiet" period to
  contrast against and may under-flag. This is a known trade-off, not a bug --
  see the project documentation for the full discussion.
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
        model = load_model()
        result = score_game(appid, game_name or str(appid), model)

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
            f"**{result['n_flagged']}** ({result['pct_flagged']}%) as likely low-quality or bomb-driven."
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
