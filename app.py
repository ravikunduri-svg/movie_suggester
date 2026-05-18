"""
Movie Suggester — powered by IMDb Non-Commercial Datasets
Run: streamlit run app.py
"""

import csv
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from groq import Groq
import pandas as pd
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILE     = Path(__file__).parent / "data" / "movies.parquet"
FEEDBACK_FILE = Path(__file__).parent / "data" / "feedback.csv"

GENRES = [
    "Any", "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
    "Music", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western",
]

MOOD_GENRES = {
    "😂 Laugh":        ["Comedy", "Animation"],
    "😨 Thrill":       ["Thriller", "Horror", "Mystery"],
    "🥺 Feel Things":  ["Drama", "Romance"],
    "🚀 Adventure":    ["Action", "Adventure", "Sci-Fi"],
    "🧠 Think":        ["Documentary", "Biography", "History"],
    "✨ Wonder":       ["Fantasy", "Animation", "Family"],
    "💥 Intensity":    ["Crime", "War", "Action"],
}

QUIZ_ERA = {
    "🕰️ Classic (before 1990)": (1900, 1989),
    "📼 90s–00s":               (1990, 2009),
    "📱 Modern (2010–now)":     (2010, 2025),
    "🎲 Any era":               (1900, 2025),
}

QUIZ_POPULARITY = {
    "🌍 Blockbusters only": 100_000,
    "👥 Well-known":         10_000,
    "🔍 Hidden gems":           500,
}

QUIZ_QUALITY = {
    "🏆 Only the best (8.0+)": 8.0,
    "👍 Good is enough (7.0+)": 7.0,
    "🎲 Surprise me (6.0+)":    6.0,
}

LANGUAGES = {
    "Any":       None,
    "English":   "en",
    "French":    "fr",
    "Spanish":   "es",
    "Italian":   "it",
    "Japanese":  "ja",
    "German":    "de",
    "Hindi":     "hi",
    "Russian":   "ru",
    "Tamil":     "ta",
    "Korean":    "ko",
    "Telugu":    "te",
    "Malayalam": "ml",
}

# Reverse lookup for human-readable display in results table
LANG_CODE_TO_NAME = {code: name for name, code in LANGUAGES.items() if code}

OTT_PLATFORMS = [
    "Netflix", "Prime Video", "JioHotstar",
    "SonyLIV", "Zee5", "Aha", "MX Player", "Lionsgate Play",
]

OTT_FILE = Path(__file__).parent / "data" / "ott_catalog.parquet"

SOUTH_LANG_OPTIONS = ["Tamil", "Telugu", "Malayalam", "Kannada"]
SOUTH_LANG_CODES   = {"Tamil": "ta", "Telugu": "te", "Malayalam": "ml", "Kannada": "kn"}

_GROQ_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

PAGE_SIZE = 20

# ── JustWatch helpers (no API key required) ───────────────────────────────────

_JW_URL = "https://apis.justwatch.com/graphql"
_JW_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Accept":       "*/*",
    "Origin":       "https://www.justwatch.com",
    "Referer":      "https://www.justwatch.com/",
}

_JW_SEARCH_QUERY = """
query SearchOTT(
  $country: Country!
  $language: Language!
  $first: Int!
  $searchQuery: String!
  $offerFilter: OfferFilter!
) {
  popularTitles(
    country: $country
    language: $language
    first: $first
    sortBy: POPULAR
    sortRandomSeed: 0
    filter: { objectTypes: [MOVIE], searchQuery: $searchQuery }
  ) {
    edges {
      node {
        content(country: $country, language: $language) {
          title
          originalReleaseYear
          externalIds { imdbId }
        }
        offers(country: $country, platform: WEB, filter: $offerFilter) {
          package { clearName }
          monetizationType
        }
      }
    }
  }
}
"""

_JW_OFFER_FILTER = {"monetizationTypes": ["FLATRATE", "FREE", "ADS"]}

def _jw_post(query: str, variables: dict) -> dict | None:
    try:
        r = requests.post(
            _JW_URL,
            json={"variables": variables, "query": query},
            headers=_JW_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _jw_edges_to_rows(edges: list, df_imdb: pd.DataFrame) -> list[dict]:
    """Convert JustWatch edges to display rows, enriching from local IMDb data."""
    rows = []
    for edge in edges:
        node    = edge.get("node", {})
        content = node.get("content", {})
        imdb_id = content.get("externalIds", {}).get("imdbId")
        title   = content.get("title", "—")
        year    = content.get("originalReleaseYear") or "—"

        offers  = node.get("offers", [])
        plat_names = sorted({o["package"]["clearName"] for o in offers if o.get("package")})
        plat_str   = ", ".join(plat_names) if plat_names else "—"

        # Enrich with IMDb rating/language if we have a match
        rating, lang_disp = "—", "—"
        if imdb_id and not df_imdb.empty:
            match = df_imdb[df_imdb["tconst"] == imdb_id]
            if not match.empty:
                m = match.iloc[0]
                rating    = round(m["averageRating"], 1) if pd.notna(m.get("averageRating")) else "—"
                lang_code = m.get("language", "")
                lang_disp = LANG_CODE_TO_NAME.get(lang_code, lang_code.upper() if lang_code else "—")

        rows.append({
            "Title":       title,
            "Year":        year,
            "Language":    lang_disp,
            "Rating ⭐":   rating,
            "OTT (India)": plat_str,
            "IMDb":        f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "—",
        })
    return rows

@st.cache_data(ttl=10800, show_spinner=False)
def search_jw(query: str) -> list:
    data = _jw_post(_JW_SEARCH_QUERY, {
        "country":     "IN",
        "language":    "en",
        "first":       8,
        "searchQuery": query,
        "offerFilter": _JW_OFFER_FILTER,
    })
    return (data or {}).get("data", {}).get("popularTitles", {}).get("edges", [])

@st.cache_data(ttl=10800, show_spinner=False)
def fetch_south_ott_confirmed(
    lang_codes_tuple: tuple,
    min_year: int,
    max_results: int,
) -> list[dict]:
    """
    IMDb-first approach:
      1. Pull recent South Indian movies from local IMDb parquet (language-accurate).
      2. Search each title on JustWatch India to confirm OTT availability.
      3. Return only movies with at least one flatrate/free/ads platform.
    Cached 3 hours. First load is slow (~1s per candidate); subsequent loads instant.
    """
    lang_set = set(lang_codes_tuple)
    mask = (
        df_all["language"].isin(lang_set) &
        (df_all["startYear"] >= min_year) &
        (df_all["numVotes"] >= 500)
    )
    candidates = (
        df_all[mask]
        .sort_values(["startYear", "numVotes"], ascending=[False, False])
        .head(max_results * 5)   # over-fetch so we still hit max_results after JW misses
    )

    rows: list[dict] = []
    for _, movie in candidates.iterrows():
        if len(rows) >= max_results:
            break
        edges = search_jw(movie["primaryTitle"])
        if not edges:
            continue
        for edge in edges[:5]:
            node    = edge.get("node", {})
            content = node.get("content", {})
            jw_imdb = (content.get("externalIds") or {}).get("imdbId")
            if jw_imdb != movie["tconst"]:
                continue                      # wrong movie — keep scanning
            offers = node.get("offers", [])
            plats  = sorted({o["package"]["clearName"] for o in offers if o.get("package")})
            if not plats:
                break                         # right movie but not on OTT right now
            lang_code = movie.get("language", "")
            lang_disp = LANG_CODE_TO_NAME.get(lang_code, lang_code.upper() if lang_code else "—")
            rows.append({
                "Title":       movie["primaryTitle"],
                "Year":        int(movie["startYear"]),
                "Language":    lang_disp,
                "Rating ⭐":   round(movie["averageRating"], 1) if pd.notna(movie.get("averageRating")) else "—",
                "OTT (India)": ", ".join(plats),
                "IMDb":        f"https://www.imdb.com/title/{movie['tconst']}/",
            })
            break
    return rows

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Suggester",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_FILE)

@st.cache_data(show_spinner=False)
def load_ott_data() -> pd.DataFrame | None:
    if not OTT_FILE.exists():
        return None
    return pd.read_parquet(OTT_FILE)

with st.spinner("Loading movie database…"):
    df_all = load_data()

df_ott = load_ott_data()

# ── Session state ─────────────────────────────────────────────────────────────

for k, v in {"results": None, "page": 1, "fy_results": None, "fy_page": 1,
             "ott_results": None, "ott_page": 1,
             "ott_lookup_results": None,
             "jw_lookup_results": None,
             "south_releases": None,
             "cast_result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_search, tab_foryou, tab_ott, tab_south, tab_cast, tab_feedback = st.tabs(
    ["🎬 Movies", "✨ For You", "📺 Streaming", "🌟 New on OTT", "🎭 Cast AI", "💬 Feedback"]
)

# ── Cast prediction ───────────────────────────────────────────────────────────

def get_cast_prediction(story: str, genre: str, language: str, era: str) -> dict:
    """Call Gemini to predict cast. Returns parsed JSON dict."""
    hints = []
    if genre != "Any":
        hints.append(f"Genre: {genre}")
    if language != "Any":
        hints.append(f"Language/Region: {language}")
    if era != "Any":
        hints.append(f"Era: {era}")
    hint_text = "\n".join(hints) if hints else "No additional hints."

    prompt = f"""You are a world-class film casting director.
Given the story below, suggest an ideal cast for a film adaptation.
{hint_text}

Story:
{story}

Respond with ONLY valid JSON in this exact structure:
{{
  "suggested_title": "A short working title",
  "genre": "Genre(s) of this film",
  "tone": "2-4 word tone descriptor",
  "director": {{
    "name": "Director Name",
    "reason": "One sentence why"
  }},
  "cast": [
    {{"role": "Lead Actor", "actor": "Name", "reason": "One sentence why"}},
    {{"role": "Lead Actress", "actor": "Name", "reason": "One sentence why"}},
    {{"role": "Supporting 1", "actor": "Name", "reason": "One sentence why"}},
    {{"role": "Supporting 2", "actor": "Name", "reason": "One sentence why"}},
    {{"role": "Supporting 3", "actor": "Name", "reason": "One sentence why"}}
  ],
  "overall_reasoning": "2-3 sentence summary of casting vision"
}}"""

    client = Groq(api_key=_GROQ_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# ── Shared results renderer ───────────────────────────────────────────────────

def render_results(df: pd.DataFrame | None, page_key: str = "page", extra_cols: list | None = None) -> None:
    if df is None:
        return
    if df.empty:
        st.warning("No movies found. Try adjusting your filters.")
        return

    page_df = df.head(st.session_state[page_key] * PAGE_SIZE)
    st.markdown(f"Showing **{len(page_df):,}** of **{len(df):,}** results")

    rows = [
        {
            "Title":     m["primaryTitle"],
            "Year":      int(m["startYear"]),
            "Rating ⭐": round(m["averageRating"], 1),
            "Votes":     int(m["numVotes"]),
            "Genres":    m["genres"].replace(",", ", "),
            "Language":  LANG_CODE_TO_NAME.get(m.get("language"), m.get("language") or "—"),
            **( {"Platforms": m.get("platforms") or "—"} if extra_cols and "platforms" in extra_cols else {} ),
            "IMDb":      f"https://www.imdb.com/title/{m['tconst']}/",
        }
        for _, m in page_df.iterrows()
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rating ⭐": st.column_config.NumberColumn(format="%.1f"),
            "Votes":     st.column_config.NumberColumn(format="%,d"),
            "IMDb":      st.column_config.LinkColumn(display_text="View on IMDb"),
        },
    )

    if len(page_df) < len(df):
        if st.button("Load 20 more", key=f"load_more_{page_key}"):
            st.session_state[page_key] += 1
            st.rerun()
    else:
        st.caption("You've reached the end of the results.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MOVIE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.title("🎬 Movie Suggester")
    st.caption(f"Powered by IMDb Non-Commercial Datasets · {len(df_all):,} movies indexed")

    with st.sidebar:
        st.header("Search Filters")

        genre = st.selectbox("Genre", GENRES)

        language = st.selectbox("Language", list(LANGUAGES.keys()))

        min_rating = st.select_slider(
            "Minimum Rating",
            options=[5.0, 6.0, 7.0, 7.5, 8.0, 8.5],
            value=7.0,
            format_func=lambda x: f"⭐ {x}",
        )

        min_votes = st.select_slider(
            "Minimum Votes",
            options=[100, 500, 1_000, 5_000, 10_000, 50_000],
            value=1_000,
            format_func=lambda x: f"{x:,}",
        )

        col1, col2 = st.columns(2)
        with col1:
            year_from = st.number_input("From Year", min_value=1900, max_value=2025, value=2000, step=1)
        with col2:
            year_to = st.number_input("To Year", min_value=1900, max_value=2025, value=2025, step=1)

        st.divider()
        search_btn = st.button("🔍 Search", use_container_width=True, type="primary")

    if search_btn:
        if year_to < year_from:
            st.sidebar.error("'To Year' must be >= 'From Year'.")
        else:
            mask = (
                (df_all["averageRating"] >= min_rating) &
                (df_all["numVotes"]      >= min_votes)  &
                (df_all["startYear"]     >= int(year_from)) &
                (df_all["startYear"]     <= int(year_to))
            )
            if genre != "Any":
                mask &= df_all["genres"].str.contains(genre, na=False)
            lang_code = LANGUAGES[language]
            if lang_code and "language" in df_all.columns:
                mask &= df_all["language"] == lang_code

            st.session_state.results = df_all[mask].sort_values("averageRating", ascending=False)
            st.session_state.page = 1

    if st.session_state.results is None:
        st.info("Set your filters in the sidebar and click **Search** to find movies.")
    else:
        render_results(st.session_state.results, page_key="page")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FOR YOU
# ═══════════════════════════════════════════════════════════════════════════════

with tab_foryou:
    st.title("✨ For You")
    st.caption("Tell us your mood and we'll find your perfect movie.")

    # ── Mood selector ──────────────────────────────────────────────────────────

    st.subheader("How are you feeling right now?")
    mood = st.radio(
        "mood",
        options=list(MOOD_GENRES.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Personality quiz ───────────────────────────────────────────────────────

    st.subheader("Three quick questions")

    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        era_choice = st.radio("When from?", list(QUIZ_ERA.keys()))
    with col_q2:
        pop_choice = st.radio("How mainstream?", list(QUIZ_POPULARITY.keys()))
    with col_q3:
        qual_choice = st.radio("Quality bar?", list(QUIZ_QUALITY.keys()))

    st.divider()

    find_btn = st.button("✨ Find My Movies", type="primary", use_container_width=True)

    if find_btn:
        genres              = MOOD_GENRES[mood]
        year_from_fy, year_to_fy = QUIZ_ERA[era_choice]
        min_votes_fy        = QUIZ_POPULARITY[pop_choice]
        min_rating_fy       = QUIZ_QUALITY[qual_choice]

        def _build_mask(rating: float) -> pd.Series:
            return (
                (df_all["averageRating"] >= rating) &
                (df_all["numVotes"]      >= min_votes_fy) &
                (df_all["startYear"]     >= year_from_fy) &
                (df_all["startYear"]     <= year_to_fy) &
                (df_all["genres"].str.contains("|".join(genres), na=False))
            )

        filtered = df_all[_build_mask(min_rating_fy)].sort_values("averageRating", ascending=False)

        # Auto-relax rating threshold once if no results
        if filtered.empty and min_rating_fy > 6.0:
            relaxed = min_rating_fy - 1.0
            filtered = df_all[_build_mask(relaxed)].sort_values("averageRating", ascending=False)
            if not filtered.empty:
                st.info(f"No exact matches at {min_rating_fy}+ — showing {relaxed}+ instead.")

        st.session_state.fy_results = filtered
        st.session_state.fy_page = 1

    render_results(st.session_state.fy_results, page_key="fy_page")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STREAMING NOW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ott:
    st.title("📺 Streaming Now")
    st.caption("Find movies available on your favourite OTT platforms.")

    # ── Movie Lookup ────────────────────────────────────────────────────────────
    st.subheader("🔎 Find a Movie on OTT")
    lk_col, btn_col = st.columns([4, 1])
    with lk_col:
        lookup_title = st.text_input(
            "Movie name", placeholder="e.g. Interstellar", label_visibility="collapsed",
            key="ott_lookup_title"
        )
    with btn_col:
        lookup_btn = st.button("Find", type="primary", use_container_width=True, key="ott_lookup_btn")

    if lookup_btn:
        if not lookup_title.strip():
            st.warning("Enter a movie name to search.")
        else:
            hits = df_all[
                df_all["primaryTitle"].str.contains(lookup_title.strip(), case=False, na=False)
            ].merge(df_ott[["tconst", "platforms"]], on="tconst", how="left") \
             .sort_values("numVotes", ascending=False) \
             .head(10)
            st.session_state.ott_lookup_results = hits

    if st.session_state.ott_lookup_results is not None:
        hits = st.session_state.ott_lookup_results
        if hits.empty:
            st.info("No movies found matching that title.")
        else:
            for _, row in hits.iterrows():
                platforms = row.get("platforms")
                if pd.isna(platforms) or not platforms:
                    badge = "❌ Not on any tracked platform"
                else:
                    badge = "✅ " + str(platforms)
                st.markdown(
                    f"**{row['primaryTitle']}** ({int(row['startYear'])}) "
                    f"⭐ {row['averageRating']} &nbsp;·&nbsp; {badge}"
                )

    # ── Live JustWatch search (no API key needed) ────────────────────────────
    st.markdown("**🌐 Search Online — Live OTT Data**")
    st.caption("Search JustWatch India for current streaming availability — no API key needed.")
    jw_col, jw_btn_col = st.columns([4, 1])
    with jw_col:
        jw_query = st.text_input(
            "Search JustWatch", placeholder="e.g. Pushpa 2, Kalki 2898-AD, RRR",
            label_visibility="collapsed", key="jw_query_input",
        )
    with jw_btn_col:
        jw_search_btn = st.button(
            "Search", key="jw_search_btn", type="secondary", use_container_width=True
        )

    if jw_search_btn:
        if not jw_query.strip():
            st.warning("Enter a movie title to search.")
        else:
            with st.spinner("Searching JustWatch India…"):
                edges = search_jw(jw_query.strip())
            if not edges:
                st.session_state.jw_lookup_results = []
            else:
                st.session_state.jw_lookup_results = _jw_edges_to_rows(edges, df_all)

    jw_res = st.session_state.jw_lookup_results
    if jw_res is not None:
        if not jw_res:
            st.info("No results found on JustWatch India for that title.")
        else:
            col_cfg = {"Rating ⭐": st.column_config.NumberColumn(format="%.1f")}
            # Only add IMDb link column if entries have real URLs
            if any(r["IMDb"].startswith("http") for r in jw_res):
                col_cfg["IMDb"] = st.column_config.LinkColumn(display_text="IMDb")
            st.dataframe(
                jw_res, use_container_width=True, hide_index=True, column_config=col_cfg,
            )
            st.caption("Live from JustWatch India · Flatrate / Free / Ad-supported only · Cached 3 hours")

    st.divider()
    # ── Platform-browse section ──────────────────────────────────────────────────

    if df_ott is None:
        st.warning(
            "OTT dataset not built yet. "
            "Run `python scripts/build_ott_dataset.py` to generate it."
        )
    else:
        # ── Filters (inline, 2-column layout) ──────────────────────────────
        col_plat, col_filters = st.columns([2, 3])

        with col_plat:
            st.subheader("Platforms")
            selected_platforms = [
                p for p in OTT_PLATFORMS
                if st.checkbox(p, value=True, key=f"ott_{p}")
            ]

        with col_filters:
            st.subheader("Filters")
            cf1, cf2 = st.columns(2)
            with cf1:
                ott_genre    = st.selectbox("Genre", GENRES, key="ott_genre")
                ott_language = st.selectbox("Language", list(LANGUAGES.keys()), key="ott_lang")
            with cf2:
                ott_rating = st.select_slider(
                    "Min Rating", options=[5.0, 6.0, 7.0, 7.5, 8.0, 8.5],
                    value=7.0, format_func=lambda x: f"⭐ {x}", key="ott_rating"
                )
                ott_votes = st.select_slider(
                    "Min Votes", options=[100, 500, 1_000, 5_000, 10_000, 50_000],
                    value=1_000, format_func=lambda x: f"{x:,}", key="ott_votes"
                )
            cc1, cc2 = st.columns(2)
            with cc1:
                ott_year_from = st.number_input("From Year", 1900, 2025, 2000, key="ott_yf")
            with cc2:
                ott_year_to   = st.number_input("To Year", 1900, 2025, 2025, key="ott_yt")

        ott_search_btn = st.button("🔍 Search Streaming", type="primary", use_container_width=True)

        if ott_search_btn:
            if not selected_platforms:
                st.warning("Select at least one platform.")
            elif ott_year_to < ott_year_from:
                st.error("'To Year' must be >= 'From Year'.")
            else:
                pat = "|".join(selected_platforms)
                ott_mask = df_ott["platforms"].str.contains(pat, na=False)
                matched_tconsts = df_ott[ott_mask]["tconst"]

                mask = (
                    (df_all["tconst"].isin(matched_tconsts)) &
                    (df_all["averageRating"] >= ott_rating) &
                    (df_all["numVotes"]      >= ott_votes) &
                    (df_all["startYear"]     >= int(ott_year_from)) &
                    (df_all["startYear"]     <= int(ott_year_to))
                )
                if ott_genre != "Any":
                    mask &= df_all["genres"].str.contains(ott_genre, na=False)
                ott_lang_code = LANGUAGES[ott_language]
                if ott_lang_code and "language" in df_all.columns:
                    mask &= df_all["language"] == ott_lang_code

                result = df_all[mask].merge(
                    df_ott[["tconst", "platforms"]], on="tconst", how="left"
                ).sort_values("averageRating", ascending=False)

                st.session_state.ott_results = result
                st.session_state.ott_page = 1

        if st.session_state.ott_results is None:
            st.info("Select platforms and filters, then click **Search Streaming**.")
        else:
            render_results(st.session_state.ott_results, page_key="ott_page",
                           extra_cols=["platforms"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NEW ON OTT (South Movies)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_south:
    st.title("🌟 New on OTT — South Movies")
    st.caption(
        "Recent South Indian movies confirmed on streaming in India · "
        "IMDb for language accuracy · JustWatch for live platform data · No API key required"
    )

    s_col1, s_col2, s_col3 = st.columns([3, 2, 2])
    with s_col1:
        selected_south_langs = st.multiselect(
            "Languages",
            options=SOUTH_LANG_OPTIONS,
            default=SOUTH_LANG_OPTIONS,
            key="south_lang_select",
        )
    with s_col2:
        south_min_year = st.select_slider(
            "Released since",
            options=[2020, 2021, 2022, 2023, 2024],
            value=2023,
            format_func=lambda y: str(y),
            key="south_min_year",
        )
    with s_col3:
        south_max_results = st.select_slider(
            "Max results",
            options=[10, 20, 30, 40],
            value=20,
            key="south_max_results",
        )

    south_fetch_btn = st.button(
        "🔄 Load South OTT Movies", type="primary", use_container_width=True, key="south_fetch_btn"
    )

    if south_fetch_btn:
        if not selected_south_langs:
            st.warning("Select at least one language.")
        else:
            lang_codes = tuple(sorted(SOUTH_LANG_CODES[l] for l in selected_south_langs))
            with st.spinner(
                "Searching JustWatch India for each candidate… "
                "first load ~30s · results cached 3 hours"
            ):
                south_rows = fetch_south_ott_confirmed(lang_codes, south_min_year, south_max_results)
            st.session_state.south_releases = south_rows

    south_res = st.session_state.south_releases
    if south_res is None:
        st.info(
            "Click **Load South OTT Movies** to find recent Tamil, Telugu, "
            "Malayalam, and Kannada movies currently streaming in India."
        )
    elif not south_res:
        st.warning(
            "No confirmed OTT results for the selected filters. "
            "Try an earlier 'Released since' year or add more languages."
        )
    else:
        st.markdown(f"**{len(south_res)} South Indian movies** currently on OTT in India")
        col_cfg_south = {"Rating ⭐": st.column_config.NumberColumn(format="%.1f")}
        if any(r["IMDb"].startswith("http") for r in south_res):
            col_cfg_south["IMDb"] = st.column_config.LinkColumn(display_text="IMDb")
        st.dataframe(
            south_res,
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg_south,
        )
        st.caption(
            "Source: IMDb (language · rating) + JustWatch India (platform availability) · "
            "Flatrate / Free / Ad-supported only · Cached 3 hours"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CAST AI
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cast:
    st.title("🎭 Cast Predictor")
    st.caption("Describe your story — AI will suggest the perfect cast.")

    if not _GROQ_KEY:
        st.error(
            "Groq API key not found. "
            "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or set it as an env var."
        )
    else:
        # ── Inputs ─────────────────────────────────────────────────────────
        story = st.text_area(
            "Your story *",
            height=180,
            placeholder=(
                "e.g. A grieving father in rural India seeks revenge after his "
                "daughter is kidnapped by a local crime syndicate..."
            ),
            key="cast_story",
        )

        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            cast_genre = st.selectbox("Genre hint", GENRES, key="cast_genre")
        with hc2:
            cast_language = st.selectbox("Language / Region", list(LANGUAGES.keys()), key="cast_lang")
        with hc3:
            cast_era = st.selectbox(
                "Era", ["Any", "Classic (pre-1990)", "90s–2000s", "Modern (2010+)"],
                key="cast_era",
            )

        predict_btn = st.button(
            "🎬 Predict Cast", type="primary", use_container_width=True
        )

        if predict_btn:
            if not story.strip():
                st.warning("Please enter a story description.")
            else:
                with st.spinner("Casting in progress…"):
                    try:
                        result = get_cast_prediction(
                            story.strip(), cast_genre, cast_language, cast_era
                        )
                        st.session_state.cast_result = result
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.session_state.cast_result = None

        # ── Results ────────────────────────────────────────────────────────
        res = st.session_state.cast_result
        if res is None:
            st.info("Enter your story and click **Predict Cast**.")
        else:
            st.markdown(
                f"### {res.get('suggested_title', 'Untitled')}"
                f"  &nbsp; `{res.get('genre', '')}` &nbsp; · &nbsp; _{res.get('tone', '')}_"
            )
            st.divider()

            st.subheader("Suggested Cast")
            cast_rows = [
                {"Role": c["role"], "Actor / Actress": c["actor"], "Why": c["reason"]}
                for c in res.get("cast", [])
            ]
            st.dataframe(cast_rows, use_container_width=True, hide_index=True)

            director = res.get("director", {})
            st.markdown(
                f"**Director:** {director.get('name', '—')}  \n"
                f"_{director.get('reason', '')}_"
            )
            st.divider()

            st.markdown(f"**Casting vision:** {res.get('overall_reasoning', '')}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

with tab_feedback:
    st.header("💬 Feedback & Feature Requests")
    st.markdown(
        "Got an idea? Something not working? Tell me what to build next — "
        "I read every submission."
    )

    with st.form("feedback_form", clear_on_submit=True):
        name    = st.text_input("Your name (optional)")
        email   = st.text_input("Email (optional — only if you want a reply)")
        topic   = st.selectbox(
            "What's this about?",
            ["Feature request", "Bug report", "UI/UX suggestion", "Other"],
        )
        message = st.text_area("Your message *", height=120, placeholder="Tell me what you'd like to see...")
        submit  = st.form_submit_button("Send Feedback", type="primary", use_container_width=True)

    if submit:
        if not message.strip():
            st.error("Message cannot be empty.")
        else:
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "name":      name.strip(),
                "email":     email.strip(),
                "topic":     topic,
                "message":   message.strip(),
            }
            FEEDBACK_FILE.parent.mkdir(exist_ok=True)
            write_header = not FEEDBACK_FILE.exists()
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            st.success("Thanks! Your feedback has been recorded.")
