"""
Run once locally to generate data/movies.parquet from IMDb raw files.
Requires in imdb_data/:
  - title.basics.tsv.gz
  - title.ratings.tsv.gz

Language data is fetched from Wikidata via a few paginated queries
(no per-movie lookups — avoids rate limiting).

Usage: python scripts/build_dataset.py
"""
import time
from pathlib import Path
import requests
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "imdb_data"
OUTPUT   = Path(__file__).parent.parent / "data" / "movies.parquet"
OUTPUT.parent.mkdir(exist_ok=True)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "movie-suggester-build/1.0 (educational project)",
    "Accept": "application/sparql-results+json",
}
PAGE_SIZE = 50_000

# ── Fetch all IMDb→language mappings from Wikidata in pages ──────────────────

def fetch_all_languages() -> dict[str, str]:
    """
    Single-pass paginated query: get every film in Wikidata that has
    both an IMDb ID and an original language ISO code.
    Far more efficient than per-movie lookups.
    """
    lang_map: dict[str, str] = {}
    offset = 0

    while True:
        query = f"""
SELECT ?imdbId ?langCode WHERE {{
  ?film wdt:P345 ?imdbId ;
        wdt:P364 ?lang .
  ?lang wdt:P218 ?langCode .
  FILTER(STRSTARTS(STR(?imdbId), "tt"))
}}
LIMIT {PAGE_SIZE}
OFFSET {offset}
"""
        print(f"  Fetching offset {offset} ...")
        rows = None
        for attempt in range(5):
            try:
                r = requests.get(
                    WIKIDATA_SPARQL,
                    params={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=120,
                )
                if r.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s before retry {attempt+1}/5 ...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                rows = r.json()["results"]["bindings"]
                break
            except Exception as e:
                print(f"  Error (attempt {attempt+1}): {e}")
                time.sleep(30)
        if rows is None:
            print("  All retries failed — stopping pagination.")
            break

        for row in rows:
            imdb_id  = row["imdbId"]["value"]
            lang_code = row["langCode"]["value"]
            if imdb_id not in lang_map:  # keep first (Wikidata may list multiple)
                lang_map[imdb_id] = lang_code

        print(f"    Got {len(rows)} rows — total mapped: {len(lang_map):,}")
        if len(rows) < PAGE_SIZE:
            break   # last page
        offset += PAGE_SIZE
        print(f"  Waiting 30s before next page ...")
        time.sleep(30)

    return lang_map


# ── Load basics ───────────────────────────────────────────────────────────────

print("Reading title.basics ...")
basics = pd.read_csv(
    DATA_DIR / "title.basics.tsv.gz",
    sep="\t", na_values="\\N",
    usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres"],
    dtype=str, compression="gzip",
)
basics = basics[basics["titleType"] == "movie"].copy()
basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
basics.dropna(subset=["startYear", "genres"], inplace=True)
basics["startYear"] = basics["startYear"].astype(int)

print("Reading title.ratings ...")
ratings = pd.read_csv(
    DATA_DIR / "title.ratings.tsv.gz",
    sep="\t", na_values="\\N",
    dtype={"tconst": str, "averageRating": float, "numVotes": int},
    compression="gzip",
)

print("Merging ...")
df = basics.merge(ratings, on="tconst", how="inner")
df = df[df["numVotes"] >= 100].reset_index(drop=True)

# ── Fetch languages ───────────────────────────────────────────────────────────

print(f"\nFetching language data from Wikidata ...")
lang_map = fetch_all_languages()
print(f"Total language entries fetched: {len(lang_map):,}")

df["language"] = df["tconst"].map(lang_map).fillna("en")

# ── Save ──────────────────────────────────────────────────────────────────────

df.to_parquet(OUTPUT, index=False, compression="snappy")
size = OUTPUT.stat().st_size / 1e6

print(f"\nDone -> {OUTPUT}  ({size:.1f} MB, {len(df):,} movies)")
print("\nTop languages:")
print(df["language"].value_counts().head(20).to_string())
