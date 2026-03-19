"""
Build OTT catalog via JustWatch GraphQL API (no auth, no API key).
Queries each provider for India (IN), extracts IMDb IDs, writes
data/ott_catalog.parquet.

Usage:  python scripts/build_ott_dataset.py
Runtime: ~5-15 min (rate-limited to 1 req/sec).
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

GRAPHQL_URL = "https://apis.justwatch.com/graphql"
COUNTRY     = "IN"
LANGUAGE    = "en"
PAGE_SIZE   = 100

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Origin": "https://www.justwatch.com",
    "Referer": "https://www.justwatch.com/",
}

OUTPUT = Path(__file__).parent.parent / "data" / "ott_catalog.parquet"

TARGET_PROVIDERS = {
    "Netflix":          "netflix",
    "Prime Video":      "amazonprimevideo",
    "JioHotstar":       "jiohotstar",   # Disney+ Hotstar merged into JioHotstar
    "SonyLIV":          "sonyliv",
    "Zee5":             "zee5",
    "Aha":              "aha",
    "MX Player":        "mxplayer",
    "Lionsgate Play":   "lionsgateplay",
}

# GraphQL query with cursor-based pagination
_QUERY = """
query GetPopularTitles(
  $country: Country!
  $language: Language!
  $first: Int!
  $after: String
  $filter: TitleFilter!
  $offerFilter: OfferFilter!
) {
  popularTitles(
    country: $country
    filter: $filter
    first: $first
    after: $after
    sortBy: POPULAR
    sortRandomSeed: 0
  ) {
    pageInfo { endCursor hasNextPage }
    edges {
      node {
        objectType
        content(country: $country, language: $language) {
          title
          originalReleaseYear
          externalIds { imdbId }
        }
        offers(country: $country, platform: WEB, filter: $offerFilter) {
          package { technicalName }
        }
      }
    }
  }
}
"""


def get_providers() -> dict[str, str]:
    """Return {clearName: technicalName} for all providers in IN."""
    payload = {
        "operationName": "GetProviders",
        "variables": {"platform": "WEB", "country": COUNTRY},
        "query": """
          query GetProviders($platform: Platform!, $country: Country!) {
            packages(platform: $platform, country: $country) {
              clearName technicalName
            }
          }
        """,
    }
    r = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=15)
    r.raise_for_status()
    pkgs = r.json()["data"]["packages"]
    return {p["clearName"]: p["technicalName"] for p in pkgs}


def fetch_provider_movies(provider_id: str) -> list[str]:
    """Return list of IMDb IDs for movies on this provider in IN."""
    imdb_ids: list[str] = []
    cursor: str | None = None
    page = 0

    while True:
        payload = {
            "operationName": "GetPopularTitles",
            "variables": {
                "country": COUNTRY,
                "language": LANGUAGE,
                "first": PAGE_SIZE,
                "after": cursor,
                "filter": {
                    "objectTypes": ["MOVIE"],
                    "packages": [provider_id],
                },
                "offerFilter": {
                    "packages": [provider_id],
                },
            },
            "query": _QUERY,
        }

        try:
            r = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"    Error on page {page + 1}: {exc}. Stopping.")
            break

        popular = data.get("data", {}).get("popularTitles", {})
        edges = popular.get("edges", [])
        page_info = popular.get("pageInfo", {})

        for edge in edges:
            node = edge.get("node", {})
            # Confirm the item actually has an offer from this provider
            offers = node.get("offers", [])
            has_offer = any(
                o.get("package", {}).get("technicalName") == provider_id
                for o in offers
            )
            if not has_offer:
                continue
            imdb_id = (
                node.get("content", {})
                    .get("externalIds", {})
                    .get("imdbId")
            )
            if imdb_id:
                imdb_ids.append(imdb_id)

        page += 1
        print(f"    page {page:3d}  items: {len(edges):3d}  with IMDb ID so far: {len(imdb_ids):,}")

        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        time.sleep(1)

    return imdb_ids


def main() -> None:
    # Verify provider availability
    print("Fetching provider list for IN...")
    try:
        available = get_providers()
    except Exception as exc:
        print(f"Failed to fetch providers: {exc}")
        sys.exit(1)

    available_ids = set(available.values())
    print(f"Found {len(available_ids)} providers in IN region.\n")

    for name, pid in TARGET_PROVIDERS.items():
        status = "OK" if pid in available_ids else "NOT FOUND"
        print(f"  {name:20s} ({pid})  {status}")

    print()

    tconst_to_platforms: dict[str, list[str]] = {}

    for platform_name, provider_id in TARGET_PROVIDERS.items():
        if provider_id not in available_ids:
            print(f"Skipping {platform_name} - provider '{provider_id}' not available in IN.")
            continue

        print(f"Fetching {platform_name} ({provider_id})...")
        ids = fetch_provider_movies(provider_id)
        for imdb_id in ids:
            tconst_to_platforms.setdefault(imdb_id, []).append(platform_name)
        print(f"  {platform_name} done - {len(ids):,} movies with IMDb IDs.\n")

    if not tconst_to_platforms:
        print("No data collected. Check network access or provider IDs.")
        sys.exit(1)

    df = pd.DataFrame([
        {"tconst": k, "platforms": ", ".join(sorted(set(v)))}
        for k, v in tconst_to_platforms.items()
    ])

    OUTPUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression="snappy")

    print(f"\nDone -> {OUTPUT}  ({len(df):,} movies with OTT data)")
    print("\nPlatform breakdown:")
    print(df["platforms"].str.split(", ").explode().value_counts().to_string())


if __name__ == "__main__":
    main()
