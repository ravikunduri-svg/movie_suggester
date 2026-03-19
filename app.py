"""
Movie Suggester — powered by IMDb Non-Commercial Datasets
Run: streamlit run app.py
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
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

_ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))

PAGE_SIZE = 20

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
             "ott_results": None, "ott_page": 1, "cast_result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_search, tab_foryou, tab_ott, tab_cast, tab_feedback = st.tabs(
    ["🎬 Movies", "✨ For You", "📺 Streaming", "🎭 Cast AI", "💬 Feedback"]
)

# ── Cast prediction ───────────────────────────────────────────────────────────

def get_cast_prediction(story: str, genre: str, language: str, era: str) -> dict:
    """Call Claude to predict cast. Returns parsed JSON dict."""
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

    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

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
# TAB 4 — CAST AI
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cast:
    st.title("🎭 Cast Predictor")
    st.caption("Describe your story — AI will suggest the perfect cast.")

    if not _ANTHROPIC_KEY:
        st.error(
            "Anthropic API key not found. "
            "Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` or set it as an env var."
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
# TAB 5 — FEEDBACK
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
