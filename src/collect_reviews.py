"""
collect_reviews.py

Pulls Steam user reviews via the public appreviews endpoint and saves them
to CSV, one file per game.

WHY this file exists:
    The notebook should not contain raw scraping/pagination logic mixed in
    with analysis code. Keeping data collection in its own module means:
      1. You can re-run a pull without re-running your whole notebook.
      2. The notebook stays readable ("load the CSV" vs. 40 lines of
         pagination code).
      3. This file is unit-testable and reusable -- a good thing to show
         in a portfolio repo.

HOW it works (read this before you run it):
    Steam's appreviews API returns reviews in pages of up to 100, and gives
    you a "cursor" string to fetch the next page. There is no "give me
    everything" call -- you keep asking for the next page until the cursor
    stops changing / no more reviews come back.

    Two important, non-obvious parameter choices baked into this script:

    - `filter_offtopic_activity=0`
        By default Steam silently removes reviews it's flagged as
        off-topic/review-bomb activity from API results. This project
        deliberately asks for them back: real-world negativity (e.g. a
        monetisation controversy) is treated as genuine sentiment for a
        language-based filter to read, not noise to exclude before the
        model even sees it.

    - `language` and `purchase_type=all`
        Leaving `purchase_type` unset silently narrows the result set
        (confirmed by hand -- querying without an explicit value
        returns a much smaller `total_reviews` count than the real
        total). Always pass it explicitly or your totals will be wrong
        and you won't notice.

        `language` is deliberately set to `"english"`, not `"all"`. This
        project's scope is English-language reviews only -- verified live
        that this parameter genuinely filters server-side rather than
        just labelling results, so it's a real, clean restriction rather
        than a downstream filter on an already-collected mixed-language
        sample (which would have unevenly shrunk each game/slice instead
        of giving properly-sized English-only pulls).

    Every game is pulled the same simple way: the most recent
    `TARGET_PER_GAME` English reviews, no special date windows. This
    keeps collection identical across every game, with no game-specific
    event tied to any part of the pipeline. Any review-bombing or
    monetisation-backlash context for a given game (e.g. Tekken 8's
    ongoing monetisation controversy) is noted in the project
    documentation as real-world context for why a game's sentiment may
    look mixed -- the pipeline itself judges quality and sentiment from
    review text alone, not from when a review was posted.

    A NOTE ON HIGH-VOLUME GAMES: Steam's cursor pagination for the
    "recent" filter only reliably reaches back so far for very
    high-volume games -- don't expect to page deep into a game's full
    history this way. Since every pull here targets only the most
    recent `TARGET_PER_GAME` reviews, this doesn't affect this project,
    but it's worth knowing if `TARGET_PER_GAME` is ever raised
    significantly higher.

USAGE:
    python collect_reviews.py                # pull all games below
    python collect_reviews.py 2868840         # pull only this appid

    Adjust GAMES and TARGET_PER_GAME below to change scope. Output lands
    in ./data/raw/<appid>_<name>.csv
"""

import csv
import os
import time
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# appid: the numeric Steam app ID (verified live against the Steam API).
# genre: noted for the project write-up (EDA/dataset diversity), not used
#   by any pipeline code.
# All games are pulled the same way -- most recent TARGET_PER_GAME
# English reviews, no special date windows (see the module docstring
# above for why).
GAMES = [
    {
        "appid": 553850,
        "name": "Helldivers 2",
        "genre": "Co-op shooter",
        # Verified live 2026-07-18: 629,255 total reviews, "Very Positive".
    },
    {
        "appid": 440,
        "name": "Team Fortress 2",
        "genre": "Multiplayer FPS",
        # Verified live 2026-07-18: 1,242,227 total reviews, "Very Positive".
    },
    {
        "appid": 2868840,
        "name": "Slay the Spire 2",
        "genre": "Deck-building roguelike",
        # Verified live 2026-07-22: 191,251 total reviews, "Mixed" overall
        # following a 2026-05-05 controversy over a consultant credit.
        # Kept in the dataset for its genuinely mixed, sometimes-heated
        # review text -- exactly the kind of real-world noise the
        # language-based quality filter needs to be able to handle.
    },
    {
        "appid": 1551360,
        "name": "Forza Horizon 5",
        "genre": "Racing",
        # Verified live 2026-07-29: 97,851 total reviews, "Very Positive".
    },
    {
        "appid": 3764200,
        "name": "Resident Evil Requiem",
        "genre": "Survival horror / adventure",
        # Verified live 2026-07-29: 78,953 total reviews, "Overwhelmingly
        # Positive". Released Feb 2026, so review volume is naturally
        # more contained than the older AAA titles in this set.
    },
    {
        "appid": 1778820,
        "name": "Tekken 8",
        "genre": "Fighting",
        # Verified live 2026-07-29: 43,183 total reviews, review_score 5
        # ("Mixed") -- real, ongoing negativity tied to a monetisation
        # controversy (battle pass / in-game store backlash). Included
        # deliberately for its mixed real-world sentiment; any
        # monetisation-driven negativity here is noted as real-world
        # context in the project documentation, not something the
        # pipeline tries to detect.
    },
    {
        "appid": 1091500,
        "name": "Cyberpunk 2077",
        "genre": "Open-world RPG",
        # Included for its open-world RPG genre -- long-form,
        # narrative-focused reviews with a very different writing style
        # than the rest of the dataset. Verified live: 411,481 total
        # English reviews, "Very Positive" -- comfortably supports 20,000.
    },
]

# How many reviews to aim for per game. Kept equal across every game so
# no single title dominates what the language-based quality/sentiment
# models learn as "normal" text. 20,000 per game gives both models
# (particularly the CNN and Stacking ensemble) enough training rows to
# keep improving without any one title's writing style dominating the
# pooled dataset.
TARGET_PER_GAME = 20000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Steam does not publish a hard rate limit for this endpoint, but community
# scrapers report reliable behavior around ~10 requests/sec. We stay well
# under that to avoid getting throttled or IP-blocked mid-pull.
SECONDS_BETWEEN_REQUESTS = 0.3

CSV_FIELDS = [
    "appid", "game_name", "slice", "recommendationid", "steamid",
    "num_games_owned", "num_reviews", "playtime_forever",
    "playtime_last_two_weeks", "playtime_at_review", "language", "review",
    "timestamp_created", "timestamp_updated", "voted_up", "votes_up",
    "votes_funny", "weighted_vote_score", "comment_count", "steam_purchase",
    "received_for_free", "written_during_early_access",
]


def _to_epoch(date_str):
    """Convert 'YYYY-MM-DD' to a Unix timestamp (UTC, midnight)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_page(appid, cursor="*", num_per_page=100):
    """
    Fetch one page of reviews for a given appid.

    Returns the parsed JSON response, or None if the request failed.
    `filter="recent"` is required here (not the default "all") because we
    need results sorted by creation time and a cursor that eventually runs
    out -- the "all" filter is designed to always find *something* to
    return and does not terminate the same way.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        "json": 1,
        "filter": "recent",
        "language": "english",  # English-only: Hikaru can't read/verify
                                 # other languages, and a review he can't
                                 # personally check the sentiment of isn't
                                 # defensible in a presentation Q&A. This
                                 # was "all" earlier in the project --
                                 # verified live that this param genuinely
                                 # filters server-side (not just labels
                                 # the response) before switching.
        "purchase_type": "all",
        "num_per_page": num_per_page,
        "cursor": cursor,
        "filter_offtopic_activity": 0,  # include review-bomb reviews
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [warn] request failed: {e}")
        return None


def pull_reviews(appid, game_name, target_count, window=None):
    """
    Pull up to `target_count` reviews for a game, optionally restricted to
    a (start_date, end_date) window. Stops early if:
      - we hit target_count,
      - the cursor stops advancing (no more pages), or
      - (when a window is set) reviews are now older than the window start,
        since "recent" filter returns reviews newest-first.
    """
    collected = []
    cursor = "*"
    window_start_epoch = _to_epoch(window[0]) if window else None
    window_end_epoch = _to_epoch(window[1]) + 86400 if window else None  # inclusive of end day

    while len(collected) < target_count:
        data = fetch_page(appid, cursor=cursor)
        if data is None or data.get("success") != 1:
            print("  [warn] stopping: bad response")
            break

        reviews = data.get("reviews", [])
        if not reviews:
            break  # no more pages

        for r in reviews:
            ts = r["timestamp_created"]

            if window:
                # Skip anything outside the window, but keep paging --
                # relevant reviews could still be further back.
                if ts > window_end_epoch:
                    continue
                if ts < window_start_epoch:
                    # Since results are newest-first, once we're older
                    # than the window start we will never see the window
                    # again. Safe to stop.
                    break
            collected.append(r)
            if len(collected) >= target_count:
                break
        else:
            # inner loop completed without hitting the "too old, stop" break
            new_cursor = data.get("cursor")
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor
            time.sleep(SECONDS_BETWEEN_REQUESTS)
            continue

        # inner loop hit a `break` (either target reached or past window)
        break

    return collected


def review_to_row(r, appid, game_name):
    """
    Flattens one raw API review object (nested `author` dict and all)
    into the flat dict shape used everywhere else in this project --
    CSV rows here, and DataFrame rows in the live demo app
    (src/live_scoring.py), so both paths produce identical columns from
    the same raw API response.
    """
    author = r.get("author", {})
    return {
        "appid": appid,
        "game_name": game_name,
        "slice": r.get("_slice", "baseline"),
        "recommendationid": r.get("recommendationid"),
        "steamid": author.get("steamid"),
        "num_games_owned": author.get("num_games_owned"),
        "num_reviews": author.get("num_reviews"),
        "playtime_forever": author.get("playtime_forever"),
        "playtime_last_two_weeks": author.get("playtime_last_two_weeks"),
        "playtime_at_review": author.get("playtime_at_review"),
        "language": r.get("language"),
        "review": r.get("review", "").replace("\n", " ").strip(),
        "timestamp_created": r.get("timestamp_created"),
        "timestamp_updated": r.get("timestamp_updated"),
        "voted_up": r.get("voted_up"),
        "votes_up": r.get("votes_up"),
        "votes_funny": r.get("votes_funny"),
        "weighted_vote_score": r.get("weighted_vote_score"),
        "comment_count": r.get("comment_count"),
        "steam_purchase": r.get("steam_purchase"),
        "received_for_free": r.get("received_for_free"),
        "written_during_early_access": r.get("written_during_early_access"),
    }


def save_csv(rows, appid, game_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = game_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"{appid}_{safe_name}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(review_to_row(r, appid, game_name))
    print(f"  saved {len(rows)} reviews -> {path}")


def main():
    import sys

    # Optional: pass one or more appids on the command line to only pull
    # those games, e.g. `python collect_reviews.py 2868840`. Useful when
    # you've already got good data for some games and just need to redo
    # or add one, instead of re-pulling everything.
    requested_appids = {int(a) for a in sys.argv[1:]} if len(sys.argv) > 1 else None
    games_to_pull = (
        [g for g in GAMES if g["appid"] in requested_appids]
        if requested_appids else GAMES
    )

    for game in games_to_pull:
        appid = game["appid"]
        name = game["name"]

        print(f"Pulling reviews for {name} (appid {appid})...")

        # Every game: most recent TARGET_PER_GAME English reviews, no
        # date window. `_slice` is a fixed "baseline" tag on every row --
        # kept in the schema for compatibility with review_to_row()/
        # save_csv(), which both expect the column to exist.
        all_rows = pull_reviews(appid, name, TARGET_PER_GAME)
        for r in all_rows:
            r["_slice"] = "baseline"

        print(f"  got {len(all_rows)} reviews")
        save_csv(all_rows, appid, name)
        time.sleep(1)  # be polite between games

    print("Done.")


if __name__ == "__main__":
    main()
