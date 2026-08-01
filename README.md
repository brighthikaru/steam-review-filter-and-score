# Steam Review Filter and Score

Capstone project for my Data Science & AI course (Institute of Data). Built to actually be used, not just graded.

## The problem

Steam's aggregate review score treats every review equally — a one-word "good game," a one-line meme, and a considered, on-topic critique all count the same toward a game's public score. That makes it hard to tell whether a score reflects genuine community sentiment, or is diluted by noise that says nothing about the game at all. Steam does some internal filtering, but it never shows you a "real sentiment" number. This project builds one.

## What it does

Pulls a game's recent English-language reviews live from Steam's own API, runs them through a trained model that flags likely low-effort/junk reviews, and shows the sentiment score before vs. after filtering — side by side, so you can see exactly what changed. It also runs a second model that predicts sentiment straight from the review text (no access to Steam's own thumbs up/down), shown next to Steam's real vote as a built-in accuracy check.

Try it: **[steam-review-filter-and-score.streamlit.app](https://steam-review-filter-and-score.streamlit.app/)**

(First load can take 30-60s if the app's been idle — free tier, it sleeps.)

## Dataset

140,000 reviews across 7 games, pulled directly from Steam's public `appreviews` API — chosen to span genuinely different genres, not just different games:

| Game | Genre | Sampled | Total English reviews | % sampled |
|---|---|---|---|---|
| Helldivers 2 | Co-op action shooter | 20,000 | 816,065 | 2.5% |
| Team Fortress 2 | Multiplayer FPS | 20,000 | 739,332 | 2.7% |
| Cyberpunk 2077 | Open-world RPG | 20,000 | 411,481 | 4.9% |
| Resident Evil Requiem | Survival horror / adventure | 20,000 | 79,019 | 25.3% |
| Slay the Spire 2 | Deck-building roguelike | 20,000 | 72,061 | 27.8% |
| Forza Horizon 5 | Racing | 20,000 | 97,866 | 20.4% |
| Tekken 8 | Fighting | 20,000 | 43,200 | 46.3% |

20,000 per game is a practical cap, not an exhaustive pull — it keeps training time manageable and keeps every game contributing equally to what the models learn as "normal" text. For high-volume games this is a thin recent slice (under 5% of history); for lower-volume games it's closer to their whole recent history — worth knowing when interpreting per-game results.

Scoped to English-only — I can't verify a review's sentiment in a language I can't read, so it wouldn't be defensible to include it. Steam's own `language` tag turned out to be reviewer-selected, not content-checked (~1.2% of "english"-tagged reviews were actually in another language), so I added a second, content-based filter on top of it.

## Models

Both models are trained on review **text alone** — no playtime, review count, or timing metadata is fed into either one at inference time. That's deliberate: it means the same model works on a review pulled from anywhere, and there's nothing per-game to recalibrate for a brand-new game the app has never seen. (Playtime, review length, and duplicate text are used only to *construct* the training label below — never as a model input.)

**Quality filter** — is this review low-effort/junk? Six model families tested:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.790 | 0.604 | 0.685 | 0.911 |
| TF-IDF + Random Forest | 0.705 | 0.971 | 0.817 | 0.954 |
| TF-IDF + Linear SVM | 0.845 | 0.891 | 0.867 | 0.956 |
| TF-IDF + Logistic Regression | 0.827 | 0.931 | 0.876 | 0.962 |
| Stacking (LogReg + NB + RF) | 0.867 | 0.900 | 0.883 | 0.963 |
| **CNN (deep learning)** | **0.930** | **0.957** | **0.943** | **0.987** |

The CNN wins clearly and consistently — precision holds in a tight 0.90-0.96 band across all seven genres (including Cyberpunk 2077's RPG-style long-form reviews, which fit right into that band), not just on average. It's the model deployed in the app.

**Sentiment scoring** — positive or negative, from text alone:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| LSTM (deep learning) | 0.958 | 0.915 | 0.936 | 0.950 |
| CNN (deep learning) | 0.956 | 0.922 | 0.939 | 0.955 |
| TF-IDF + Logistic Regression | 0.961 | 0.914 | 0.937 | 0.959 |
| **Stacking (LogReg + NB + RF)** | **0.942** | **0.954** | **0.948** | **0.959** |

**Stacking is deployed.** It wins on F1 and recall (0.948/0.954 vs. Logistic Regression's 0.937/0.914) and on agreement with Steam's real vote (91.8% overall vs. Logistic Regression's 90.4%), though it ties Logistic Regression exactly on ROC-AUC (0.959 both) — worth stating plainly rather than only reporting the metric that favours the deployed model. Per-request inference time (~65ms on a realistic 1,000-review batch) stays negligible next to the multi-second Steam API call the app already makes per lookup.

**Classification threshold.** Both models classify at the default 0.5 probability cutoff — not tuned against a specific precision/recall target, since there's no verified "low-effort" ground truth to tune against (the label itself is a proxy, see Dataset above). In practice this shows up as an asymmetry: the quality filter's precision (0.90-0.96) consistently outpaces its recall, so on the rare miss it leans toward flagging a genuine review rather than letting junk through — the safer direction for a filter feeding a public-facing score, but worth stating rather than leaving implicit. Calibrating this against an explicit target is listed under Future work.

## Result

Filtering out low-effort reviews **lowers** the positive percentage for every single game — a consistent, non-obvious finding across all seven titles. Low-effort reviews skew more positive than substantive ones (a quick "10/10 gg" throwaway is far more common than a quick throwaway complaint), so removing them reveals a more critical, more considered version of a game's sentiment. The clearest category shifts: Tekken 8 drops from "Mixed" to "Mostly Negative" (47.6% → 39.2%), and Forza Horizon 5 drops from "Very Positive" to "Mostly Positive" (88.7% → 79.7%), once low-effort praise is removed. Full numbers and per-game breakdown in the notebook.

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
- Live lookups cap at 1,000 recent reviews per game for response time — a recent-activity snapshot, not the full review history.

## Future work

- Validate the quality filter against an even broader genre spread
- Per-language models beyond English
- Calibrate the flagging threshold against a real precision/recall target
- A pretrained transformer (e.g. DistilBERT) for sentiment, to catch sarcasm and mixed-signal reviews — not adopted yet, CPU inference cost is a poor fit for live latency at this scale
- Caching and rate-limit handling for high-traffic games

---

Sean C — Institute of Data, Data Science & AI Capstone
