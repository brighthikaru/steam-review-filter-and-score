# Steam Review Filter and Score

Capstone project for my Data Science & AI course (Institute of Data). Built to actually be used, not just graded.

## The problem

Steam's aggregate review score treats every review equally — a one-word "meh," a five-minute-playtime rage post, and a coordinated review-bomb over something that has nothing to do with the game all count the same as a genuine, considered review. Steam does some internal filtering, but it never shows you a "genuine sentiment" number. This project tries to build one.

## What it does

Pulls a game's recent English-language reviews live from Steam's own API, runs them through a trained model that flags likely low-quality or review-bomb-driven reviews, and shows the sentiment score before vs. after filtering — side by side, so you can see exactly what changed and why.

Try it: **[steam-review-filter-and-score.streamlit.app](https://steam-review-filter-and-score.streamlit.app/)**

(First load can take 30-60s if the app's been idle — free tier, it sleeps.)

## Dataset

~16,000 reviews across 4 games, pulled directly from Steam's public `appreviews` API:

| Game | Role |
|---|---|
| Helldivers 2 | Baseline (normal sentiment) |
| War Thunder | Baseline (normal sentiment) |
| Team Fortress 2 | Baseline (normal sentiment) |
| Slay the Spire 2 | Review-bomb case study (May 2026 controversy) |

Scoped to English-only — I can't verify a review's sentiment in a language I can't read, so it wouldn't be defensible to include it. Steam's own `language` tag turned out to be reviewer-selected, not content-checked, so I added a second, content-based filter on top of it.

## Models

**Quality / review-bomb detection** — is this review noise?

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest (unsupervised) | 0.042 | 0.040 | 0.041 | 0.361 |
| Logistic Regression, structural only | 0.154 | 0.793 | 0.258 | 0.683 |
| **Logistic Regression, structural + text (TF-IDF)** | **0.282** | 0.697 | **0.402** | **0.847** |

**Sentiment scoring** — classical ML vs. deep learning:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| LSTM (deep learning) | 0.922 | 0.904 | 0.913 | 0.899 |
| TF-IDF + Logistic Regression (classical ML) | 0.942 | 0.877 | 0.908 | 0.928 |

The deployed app only uses the quality-filter model — Steam already gives us real ground-truth sentiment via each review's `voted_up` field, so there's no need to predict sentiment on top of that. That also keeps the app free of a TensorFlow dependency.

## Result

Filtering barely moves the score on games that aren't being bombed (that's the point — it shouldn't). On Slay the Spire 2, the model flagged a meaningfully higher share of reviews as bomb-driven, and the sample-level score moved from 85.0% to 89.7% positive. Steam's own live all-time score for that game sits at 60.4% ("Mixed") — lower than either of our numbers, because our sample is a narrow ~2-week window around the bomb, not the full ongoing history. Stated plainly in the writeup, not glossed over.

## Running it locally

```
pip install -r requirements.txt
streamlit run app.py
```

Needs the trained model artifacts in `models/` — regenerate them with:

```
python src/train_models.py
```

## Repo layout

```
app.py                  # Streamlit app
src/                     # collection, feature engineering, training, live scoring
notebooks/               # full analysis notebook + write-up
models/                  # trained model artifacts (joblib)
presentation/            # HTML slide deck with embedded live demo
data/raw/                # raw review CSVs
```

## Known limitations

- English-only scope — the score reflects English-speaking reviewers specifically, not a game's full community.
- For games outside the original 4-game training set (i.e. anything you search live), the "what's normal" baseline is computed on the fly from whatever's just been pulled, not a separately verified quiet period. If a game is mid-bomb with no quiet days in the sample, the filter has nothing to contrast against and may under-flag.
- Live lookups cap at 400 reviews per game for response time — a recent-activity snapshot, not the full review history.

## Future work

- A second bomb case study to confirm the pattern generalizes
- Per-language models beyond English
- Calibrated thresholds against a real precision/recall target
- Move from live on-demand scoring to a background/batch pipeline with caching, so full review-history scoring can run asynchronously instead of being capped at 400 per lookup

---

Sean (Hikaru) — Institute of Data, Data Science & AI Capstone
