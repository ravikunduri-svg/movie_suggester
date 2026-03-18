"""
Run once locally to generate data/movies.parquet from IMDb raw files.
Usage: python scripts/build_dataset.py
"""
from pathlib import Path
import pandas as pd

DATA_DIR   = Path(__file__).parent.parent / "imdb_data"
OUTPUT     = Path(__file__).parent.parent / "data" / "movies.parquet"
OUTPUT.parent.mkdir(exist_ok=True)

print("Reading title.basics …")
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

print("Reading title.ratings …")
ratings = pd.read_csv(
    DATA_DIR / "title.ratings.tsv.gz",
    sep="\t", na_values="\\N",
    dtype={"tconst": str, "averageRating": float, "numVotes": int},
    compression="gzip",
)

print("Merging …")
df = basics.merge(ratings, on="tconst", how="inner")

# Keep only movies with ≥100 votes to cut file size
df = df[df["numVotes"] >= 100].reset_index(drop=True)

df.to_parquet(OUTPUT, index=False, compression="snappy")
size = OUTPUT.stat().st_size / 1e6
print(f"Done -> {OUTPUT}  ({size:.1f} MB, {len(df):,} movies)")
