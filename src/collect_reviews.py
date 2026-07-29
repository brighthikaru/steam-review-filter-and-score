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
        By default Steam SILENTLY REMOVES review-bomb reviews from API
        results (their own anti-abuse filtering). Since this whole project
        is about detecting review bombs, we have to explicitly ask for
        them back with this flag. Without it, you'd pull data for a known
        review-bombed game and see nothing unusual -- not because your
        model failed, but because the reviews were never in your dataset.

    - `language` and `purchase_type=all`
        Leaving `purchase_type` unset silently narrows the result set
        (confirmed by hand: querying War Thunder without an explicit
        value returned a `total_reviews` count of 831 instead of the
        real ~768,000). Always pass it explicitly or your totals will be
        wrong and you won't notice.

        `language` is deliberately set to `"english"`, not `"all"`. This
        project's scope is English-language reviews only -- verified live
        that this parameter genuinely filters server-side rather than
        just labelling results, so it's a real, clean restriction rather
        than a downstream filter on an already-collected mixed-language
        sample (which would have unevenly shrunk each game/slice instead
        of giving properly-sized English-only pulls).

    For games with a known review-bomb event, we pull two separate
    slices instead of one blind pull:
      1. A "bomb window" slice -- reviews created during the known
         incident dates, which is where the review-bomb signal actually
         lives.
      2. A "baseline" slice -- reviews from a specific *pre-bomb* date
         range, so your model has clean non-bombed examples to compare
         against. This is deliberately an explicit past window, not
         "whatever's most recent" -- pulling the newest reviews as
         "baseline" would silently mix in post-bomb-era discussion/
         backlash and undermine the comparison.

    IMPORTANT LESSON LEARNED (2026-07-22): Steam's cursor pagination for
    the "recent"/"updated" filters only reliably reaches back so far for
    high-volume games -- in testing, pulling a bomb window from over a
    year ago (War Thunder May 2023, Helldivers 2 May 2024) returned 0
    results even after the pull otherwise completed normally. This is a
    known limitation of the API itself (community-reported), not a bug
    in this script's logic (the window-filtering logic was unit-tested
    and is correct). Practical takeaway: only target bomb windows that
    are recent (weeks to a few months old, not years), and expect that
    older documented incidents for high-volume games may simply be
    unreachable through this endpoint. Helldivers 2, War Thunder, and TF2
    are kept below as "baseline-only" normal-sentiment comparison games
    (no bomb_window) since their historical incidents are too old to
    reach; Slay the Spire 2 is the live bomb-window case study instead,
    since its incident is only ~2.5 months old as of this writing.

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
# bomb_window: (start, end) as "YYYY-MM-DD" -- dates of a documented
#   review-bomb incident. None if there isn't a reachable one (see note
#   above about how old is "too old" for this API).
# baseline_window: (start, end) as "YYYY-MM-DD" -- a clean pre-bomb (or
#   otherwise "normal") date range to sample as comparison data. None
#   means "just grab the most recent reviews" (fine when there's no
#   bomb_window to avoid overlapping with).
GAMES = [
    {
        "appid": 553850,
        "name": "Helldivers 2",
        # Verified live 2026-07-18: 629,255 total reviews, "Very Positive"
        # overall. Known bomb (May 2024, Sony PSN linking backlash) is too
        # old to reach via cursor pagination -- confirmed by testing on
        # 2026-07-22 (0 results). Kept as a baseline-only comparison game.
        "bomb_window": None,
        "baseline_window": None,
    },
    {
        "appid": 236390,
        "name": "War Thunder",
        # Verified live 2026-07-18: 767,980 total reviews, "Mostly Positive"
        # overall. Known bomb (May 2023, economy changes) unreachable --
        # confirmed by testing on 2026-07-22 (0 results). Baseline-only.
        "bomb_window": None,
        "baseline_window": None,
    },
    {
        "appid": 440,
        "name": "Team Fortress 2",
        # Verified live 2026-07-18: 1,242,227 total reviews, "Very
        # Positive" overall. Known bomb (mid-2024, bot/cheater protest)
        # is both old and this is the highest-volume game in the set --
        # cursor depth is even less likely to reach it. Baseline-only.
        "bomb_window": None,
        "baseline_window": None,
    },
    {
        "appid": 2868840,
        "name": "Slay the Spire 2",
        # Verified live 2026-07-22: 191,251 total reviews, "Mixed" overall
        # (down from "Overwhelmingly Positive"). Review-bombed starting
        # 2026-05-05 after Anita Sarkeesian's credit as an inclusivity
        # consultant spread on X; rating fell to 61% positive within two
        # days. Released 2026-03-05 (early access), so there's a genuine
        # ~2-month pre-bomb baseline window to sample. This is the
        # primary review-bomb case study -- recent enough that cursor
        # pagination can actually reach both windows.
        "bomb_window": ("2026-05-05", "2026-05-15"),
        "baseline_window": ("2026-03-05", "2026-05-04"),
    },
]

# How many reviews to aim for per game, split between the bomb window (if
# any) and a normal baseline sample. Keep this modest -- pulling every
# review for a game with hundreds of thousands of them is unnecessary data
# volume for this project and will just slow down every re-run.
TARGET_PER_GAME = 4000
BOMB_WINDOW_TARGET = 1500  # only used for games that have a bomb_window

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
        bomb_window = game.get("bomb_window")
        baseline_window = game.get("baseline_window")

        print(f"Pulling reviews for {name} (appid {appid})...")

        all_rows = []

        if bomb_window:
            print(f"  fetching bomb-window sample {bomb_window} ...")
            bomb_rows = pull_reviews(appid, name, BOMB_WINDOW_TARGET, window=bomb_window)
            print(f"  got {len(bomb_rows)} bomb-window reviews")
            for r in bomb_rows:
                r["_slice"] = "bomb_window"
            all_rows.extend(bomb_rows)

            baseline_target = TARGET_PER_GAME - len(bomb_rows)
            print(f"  fetching baseline sample {baseline_window} (target {baseline_target}) ...")
            baseline_rows = pull_reviews(appid, name, baseline_target, window=baseline_window)
            for r in baseline_rows:
                r["_slice"] = "baseline"
            all_rows.extend(baseline_rows)
        else:
            # No known/reachable bomb window -- just a normal comparison
            # sample. baseline_window may still be set if you want a
            # specific date range rather than "most recent".
            all_rows = pull_reviews(appid, name, TARGET_PER_GAME, window=baseline_window)
            for r in all_rows:
                r["_slice"] = "baseline"

        save_csv(all_rows, appid, name)
        time.sleep(1)  # be polite between games

    print("Done.")


if __name__ == "__main__":
    main()
