# Steam Review Filter and Score

Capstone project for my Data Science & AI course (Institute of Data). Built to actually be used, not just graded.

## The problem

Steam's aggregate review score treats every review equally — a one-word "good game," a one-line meme, and a considered, on-topic critique all count the same toward a game's public score. That makes it hard to tell whether a score reflects genuine community sentiment, or is diluted by noise that says nothing about the game at all. Steam does some internal filtering, but it never shows you a "real sentiment" number. This project builds one.

## What it does

Pulls a game's recent English-language reviews live from Steam's own API, runs them through a trained model that flags likely low-effort/junk reviews, and shows the sentiment score before vs. after filtering — side by side, so you can see exactly what changed. It also runs a second model that predicts sentiment straight from the review text (no access to Steam's own thumbs up/down), shown next to Steam's real vote as a built-in accuracy check.

Try it: **[steam-review-filter-and-score.streamlit.app](https://steam-review-filter-and-score.streamlit.app/)**

(First load can take 30-60s if the app's been idle — free tier, it sleeps.)

## Dataset

~48,000 reviews across 6 games, pulled directly from Steam's public `appreviews` API — chosen to span genuinely different genres, not just different games:

| Game | Genre | Reviews |
|---|---|---|
| Helldivers 2 | Co-op shooter | 8,000 |
| Team Fortress 2 | Multiplayer FPS | 8,000 |
| Slay the Spire 2 | Deck-building roguelike | 8,000 |
| Forza Horizon 5 | Racing | 8,000 |
| Resident Evil Requiem | Survival horror / adventure | 8,000 |
| Tekken 8 | Fighting | 8,000 |

Scoped to English-only — I can't verify a review's sentiment in a language I can't read, so it wouldn't be defensible to include it. Steam's own `language` tag turned out to be reviewer-selected, not content-checked (~1.2% of "english"-tagged reviews were actually in another language), so I added a second, content-based filter on top of it.

## Models

Both models are trained on review **text alone** — no playtime, review count, or timing metadata is fed into either one at inference time. That's deliberate: it means the same model works on a review pulled from anywhere, and there's nothing per-game to recalibrate for a brand-new game the app has never seen. (Playtime, review length, and duplicate text are used only to *construct* the training label below — never as a model input.)

**Quality filter** — is this review low-effort/junk? Five model families tested:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.799 | 0.620 | 0.698 | 0.911 |
| TF-IDF + Random Forest | 0.716 | 0.972 | 0.825 | 0.952 |
| TF-IDF + Linear SVM | 0.848 | 0.906 | 0.876 | 0.956 |
| TF-IDF + Logistic Regression | 0.821 | 0.934 | 0.874 | 0.960 |
| Stacking (LogReg + NB + RF) | 0.860 | 0.905 | 0.882 | 0.962 |
| **CNN (deep learning)** | **0.925** | **0.958** | **0.941** | **0.986** |

The CNN wins clearly and consistently — precision holds in a tight 0.90-0.95 band across all six genres, not just on average. It's the model deployed in the app.

**Sentiment scoring** — positive or negative, from text alone:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| CNN (deep learning) | 0.958 | 0.893 | 0.925 | 0.934 |
| LSTM (deep learning) | 0.960 | 0.884 | 0.920 | 0.924 |
| TF-IDF + Logistic Regression | 0.963 | 0.894 | 0.927 | 0.941 |
| **Stacking (LogReg + NB + RF)** | **0.937** | **0.958** | **0.947** | **0.942** |

The result flips here — deep learning's extra capacity doesn't pay off for sentiment the way it does for the quality filter. **Stacking is deployed**, winning on every metric including agreement with Steam's real vote (83-96% vs. Logistic Regression's 83-94%, depending on the game), with per-request inference time (~24ms on a realistic 400-review batch) negligible next to the multi-second Steam API call the app already makes per lookup.

## Result

Filtering out low-effort reviews **lowers** the positive percentage for every single game — a consistent, non-obvious finding. Low-effort reviews skew more positive than substantive ones (a quick "10/10 gg" throwaway is far more common than a quick throwaway complaint), so removing them reveals a more critical, more considered version of a game's sentiment. Helldivers 2 shows the largest shift, crossing a full Steam category from "Mostly Positive" to "Mixed" once low-effort praise is removed. Full numbers and per-game breakdown in the notebook.

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
src/                    # collection, feature engineering, training, live scoring
notebooks/              # full analysis notebook + write-up
models/                 # trained model artifacts (CNN + joblib)
presentation/           # HTML slide deck with embedded live demo
data/raw/               # raw review CSVs
```

## Known limitations

- English-only scope — the score reflects English-speaking reviewers specifically, not a game's full community.
- The "low-effort" label is a proxy (very short, OR low playtime, OR duplicate text) — not verified ground truth of "uselessness." A review can be short but insightful, or long but empty.
- The CNN quality filter and the Stacking sentiment model are both more accurate but less interpretable than a single classical model — the notebook's word-importance analyses use plain Logistic Regression to explain the signal, not either deployed model's own reasoning.
- Live lookups cap at 400 recent reviews per game for response time — a recent-activity snapshot, not the full review history.

## Future work

- Validate the quality filter against an even broader genre spread
- Per-language models beyond English
- Calibrate the flagging threshold against a real precision/recall target
- A pretrained transformer (e.g. DistilBERT) for sentiment, to catch sarcasm and mixed-signal reviews — not adopted yet, CPU inference cost is a poor fit for live latency at this scale
- Caching and rate-limit handling for high-traffic games

---

Sean C — Institute of Data, Data Science & AI Capstone
