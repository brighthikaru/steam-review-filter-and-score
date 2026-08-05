# Steam Review Filter and Score

Capstone project for my Data Science & AI course (Institute of Data). Built to actually be used, not just graded.

## The problem

A one-word "good game" or "bad" tells you someone's vote, but not why — no gameplay mechanics, no bugs, no pacing, nothing that actually helps another player judge whether the game is for them. Steam's review list mixes these thin, low-information reviews in with the ones that actually explain something, at roughly equal visibility, which makes it harder to find the reviews that would actually inform a buying decision. This project filters out the reviews that don't say much and surfaces the ones that do, so a player can read real detail instead of digging for it — with the before/after sentiment score as a secondary check on how much difference that filtering makes.

## What it does

Pulls a game's recent English-language reviews live from Steam's own API, runs them through a trained model that flags likely low-effort/junk reviews (thin on detail, barely played, or duplicated), and surfaces the longest substantive reviews on each side of the vote — the ones with enough detail to actually help a buying decision. A third component generates a plain-English "Players liked: ...", "Players disliked: ..." summary from those kept reviews, so there's a readable takeaway alongside the raw quotes. It also shows the sentiment score before vs. after filtering, led by Steam's own category label (e.g. "Mostly Positive") rather than the raw percentage — that's the number a player actually reads on Steam, so it's the headline here too, with the percentage kept underneath as supporting detail, and an explicit callout when filtering shifts the game into a different category entirely. A second model predicts sentiment straight from the review text (no access to Steam's own thumbs up/down), shown next to Steam's real vote as a built-in accuracy check.

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

20,000 per game is a practical cap, not an exhaustive pull, and three separate reasons drove that choice. First, balance: capping every game at the same number keeps each one contributing equally to what the models learn as "normal" text, so no single title's writing style dominates. Second, recency: reviews are pulled newest-first on purpose — a game can change meaningfully after launch (patches, balance changes, a monetisation controversy), so a review written last month is more representative of the game *today* than one from three years ago. Third, resources: both training time and the live app's per-request latency scale with review count, so the cap keeps the notebook re-runnable and the app responsive.

The trade-off is the flip side of the recency choice: for high-volume games this is a thin *recent* slice (under 5% of history), not a random sample across a game's full lifetime, so it can genuinely diverge from Steam's all-time score if sentiment has shifted recently — see Helldivers 2 in the notebook's EDA section for a real example of this. For lower-volume games the 20,000 cap is closer to their whole recent history, so this effect is much smaller. Worth knowing when interpreting per-game results.

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

**Summarization** — a third component, beyond the two required models for the quality-filter/sentiment comparison above: generates a plain-English "Players liked: ..." / "Players disliked: ..." summary from the kept reviews, using [`Falconsai/text_summarization`](https://huggingface.co/Falconsai/text_summarization) — a `t5-small` checkpoint (60.5M params, 242MB) fine-tuned specifically for summarization. Getting here took five attempts, not one. Four other pretrained summarizers were tried and rejected first: generic `t5-small` (safe on memory, but copy-pasted raw fragments rather than condensing them), `bart-large-cnn` (~400M params, genuinely coherent output, but OOM-crashed a live deployment on its first real request), `distilbart-cnn-12-6` (a dead end before it even loaded — only ships legacy PyTorch weights, no TF-native option), and `flan-t5-base`/`flan-t5-small` (the base model hallucinated a detail — "graphics" — present in none of the source reviews; the small model produced vague, one-sided platitudes that ignored genuinely negative reviews in the input). Given that pattern, the deployed version briefly switched to an extractive TF-IDF phrase-list instead, which couldn't hallucinate but also couldn't produce real prose. Falconsai/text_summarization was tried next specifically because it shares generic t5-small's already-proven-safe memory footprint, but is fine-tuned for the actual task rather than being a general multi-task model — local testing showed coherent, hallucination-free output across multiple runs.

That still left one problem: summarizing several reviews in a single pass reliably dropped at least one review from the output entirely, the same failure mode that sank the original t5-small, just less severe — the model would lock onto whichever single review read as most "quotable" and ignore the rest. The fix was **method, not model**: each kept review is summarized individually (much closer to the single-document task this model was actually fine-tuned on), and the short results are joined together — so no single review can be dropped, since every one gets its own generation pass. Full before/after test output for both approaches is in `test_summarizer_memory.py`.

**Classification threshold.** Both quality-filter and sentiment models classify at the default 0.5 probability cutoff — not tuned against a specific precision/recall target, since there's no verified "low-effort" ground truth to tune against (the label itself is a proxy, see Dataset above). In practice this shows up as an asymmetry: the quality filter's precision (0.90-0.96) consistently outpaces its recall, so on the rare miss it leans toward flagging a genuine review rather than letting junk through — the safer direction for a filter feeding a public-facing score, but worth stating rather than leaving implicit. Calibrating this against an explicit target is listed under Future work.

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
- The "Players liked/disliked" summary is generated text (Falconsai/text_summarization), not extracted phrases — it reads as fluent prose, but that also means it *can* in principle paraphrase or compress in a way that shifts emphasis, even though local testing across multiple runs showed no outright hallucination (no detail appearing that wasn't in the source review).
- Summarization only runs on the curated ~6-review sample (the 3 longest kept reviews per side, see "What players are actually saying"), not the full pulled batch — a sentiment that's common across the wider 1,000-review pull but happens not to appear in those specific 6 reviews won't be reflected. Each review is summarized independently and the results joined, so the output reads as several short sentences back-to-back rather than one fully unified paragraph — a deliberate trade-off (see Summarization above) to guarantee every review is represented, at some cost to flow.

## Future work

- Validate the quality filter against an even broader genre spread
- Per-language models beyond English
- Calibrate the flagging threshold against a real precision/recall target
- A pretrained transformer for sentiment specifically (e.g. DistilBERT), to catch sarcasm and mixed-signal reviews — not adopted yet, CPU inference cost on a 1,000-review batch is a poor fit for live latency at this scale
- A summary that unifies across reviews into one flowing paragraph, rather than joining independently-generated per-review summaries (see Summarization above) — the current per-review approach was chosen deliberately to guarantee no review gets dropped, but a model that can synthesize across multiple documents in one coherent pass without losing that guarantee would read more naturally.
- Filter and summarize across the **full pulled batch** (up to 1,000 reviews), not just the curated ~6-review sample shown today — this was the original goal driving this feature (surfacing what players broadly say, not just a handful of the longest reviews), and is the most direct way to close that gap. Would need a scalable approach (e.g. a batched summarization pass across all kept reviews, not just the sample) since running a generative model on 1,000 reviews per request isn't feasible at interactive speed on this hosting tier.
- Caching and rate-limit handling for high-traffic games

---

Sean C — Institute of Data, Data Science & AI Capstone
