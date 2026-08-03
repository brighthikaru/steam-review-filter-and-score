"""
Combined memory test -- loads everything the real app would load
(quality CNN + sentiment Stacking model + a TF-based t5-small
summarizer), then runs one realistic scoring + summarization pass on
a handful of sample reviews, and reports memory at each stage. This is
the number that actually matters for the "will this fit on Streamlit
Community Cloud's free tier" question -- not just TensorFlow + t5-small
in isolation.
"""

import os
import sys
import time

import psutil

proc = psutil.Process(os.getpid())


def mb():
    return proc.memory_info().rss / 1024 / 1024


print(f"baseline: {mb():.0f} MB")

import tensorflow as tf  # noqa: E402

print(f"after tensorflow import: {mb():.0f} MB")

sys.path.append("src")
from live_scoring import QualityModel, SentimentModel  # noqa: E402

quality_model = QualityModel()
sentiment_model = SentimentModel()
print(f"after quality CNN + sentiment Stacking load: {mb():.0f} MB")

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM  # noqa: E402

t0 = time.time()
tok = AutoTokenizer.from_pretrained("t5-small")
summarizer = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")
print(f"after t5-small load: {mb():.0f} MB  ({time.time() - t0:.1f}s)")

# Simulate a realistic request: score ~30 fake reviews (stand-ins for a
# real pulled batch) with both existing models, then summarize a small
# sample the way the app actually would (only the curated kept-review
# sample, not the full pull).
import pandas as pd  # noqa: E402

sample_reviews = pd.Series(
    [
        "This game has amazing gameplay mechanics but the pacing drags in act 2, still worth it.",
        "Terrible bugs on launch, crashes every hour, avoid until patched thoroughly please.",
        "Solid roguelike deck-builder with great replay value and balanced difficulty curve overall.",
        "good",
        "bad",
        "10/10",
    ]
    * 5
)

t1 = time.time()
quality_model.predict(sample_reviews)
sentiment_model.predict(sample_reviews)
print(f"after scoring {len(sample_reviews)} reviews with both models: {mb():.0f} MB  ({time.time() - t1:.1f}s)")

t2 = time.time()
combined_text = "summarize: " + " ".join(sample_reviews.head(6).tolist())
inputs = tok(combined_text, return_tensors="tf", max_length=512, truncation=True)
out = summarizer.generate(**inputs, max_new_tokens=60)
summary = tok.decode(out[0], skip_special_tokens=True)
print(f"summarization time: {time.time() - t2:.2f}s")
print("SUMMARY:", summary)

print(f"\nFINAL COMBINED MEMORY: {mb():.0f} MB")
