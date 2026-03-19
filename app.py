"""
Movie Suggester — powered by IMDb Non-Commercial Datasets
Run: streamlit run app.py
"""

import csv
from datetime import datetime
from pathlib import Path

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

LANGUAGES = {
    "Any":        None,
    # Indian languages
    "Hindi":      "hi",
    "Tamil":      "ta",
    "Telugu":     "te",
    "Malayalam":  "ml",
    "Kannada":    "kn",
    "Bengali":    "bn",
    "Marathi":    "mr",
    "Punjabi":    "pa",
    # International
    "English":    "en",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Italian":    "it",
    "Portuguese": "pt",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Chinese":    "zh",
    "Arabic":     "ar",
    "Turkish":    "tr",
    "Russian":    "ru",
    "Persian":    "fa",
}

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

with st.spinner("Loading movie database…"):
    df_all = load_data()

# ── Session state ─────────────────────────────────────────────────────────────

for k, v in {"results": None, "page": 1}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_search, tab_feedback = st.tabs(["🎬 Movies", "💬 Feedback"])

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
            if lang_code:
                mask &= df_all["language"] == lang_code

            st.session_state.results = df_all[mask].sort_values("averageRating", ascending=False)
            st.session_state.page = 1

    if st.session_state.results is None:
        st.info("Set your filters in the sidebar and click **Search** to find movies.")
    else:
        results: pd.DataFrame = st.session_state.results

        if results.empty:
            st.warning("No movies found. Try adjusting your filters.")
        else:
            page_df = results.head(st.session_state.page * PAGE_SIZE)
            st.markdown(f"Showing **{len(page_df):,}** of **{len(results):,}** results")

            rows = [
                {
                    "Title":     m["primaryTitle"],
                    "Year":      int(m["startYear"]),
                    "Rating ⭐": round(m["averageRating"], 1),
                    "Votes":     int(m["numVotes"]),
                    "Genres":    m["genres"].replace(",", ", "),
                    "Language":  m.get("language", "—"),
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

            if len(page_df) < len(results):
                if st.button("Load 20 more"):
                    st.session_state.page += 1
                    st.rerun()
            else:
                st.caption("You've reached the end of the results.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEEDBACK
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
