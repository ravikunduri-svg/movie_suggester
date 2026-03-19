"""
Movie Suggester — TMDB-powered terminal app
Usage: python movie_suggester.py
"""

import os
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import requests
    import questionary
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import box
    from dotenv import load_dotenv, set_key
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run:  pip install requests questionary rich python-dotenv")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
TMDB_BASE   = "https://api.themoviedb.org/3"
ENV_FILE    = Path(".env")
PAGE_SIZE   = 10

GENRES = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "Thriller": 53, "War": 10752, "Western": 37, "Any": None,
}

LANGUAGES = {
    # Global
    "Any": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Turkish": "tr",
    "Dutch": "nl",
    "Thai": "th",
    "Indonesian": "id",
    "Chinese": "zh",
    # East Asian
    "Japanese": "ja",
    "Korean": "ko",
    # South Asian
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
    "Punjabi": "pa",
}

console = Console()


# ── API helpers ───────────────────────────────────────────────────────────────

def tmdb_get(path: str, api_key: str, params: dict = None) -> dict | None:
    url = f"{TMDB_BASE}{path}"
    p = {"api_key": api_key, **(params or {})}
    try:
        r = requests.get(url, params=p, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            console.print("[bold red]Invalid API key.[/bold red]")
        else:
            console.print(f"[red]HTTP error: {e}[/red]")
        return None
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Network error: {e}[/red]")
        return None


def validate_api_key(key: str) -> bool:
    result = tmdb_get("/configuration", key)
    return result is not None


def get_watch_providers(movie_id: int, api_key: str, region: str = "US") -> list[str]:
    data = tmdb_get(f"/movie/{movie_id}/watch/providers", api_key)
    if not data:
        return []
    results = data.get("results", {}).get(region, {})
    providers: list[str] = []
    for ptype in ("flatrate", "free", "ads", "rent", "buy"):
        for p in results.get(ptype, []):
            name = p.get("provider_name", "")
            if name and name not in providers:
                providers.append(name)
    return providers


def discover_movies(
    api_key: str,
    genre_id: int | None,
    language: str | None,
    min_rating: float,
    year_from: int,
    year_to: int,
    page: int,
) -> tuple[list[dict], int]:
    params: dict = {
        "sort_by": "vote_average.desc",
        "vote_count.gte": 50,
        "vote_average.gte": min_rating,
        "primary_release_date.gte": f"{year_from}-01-01",
        "primary_release_date.lte": f"{year_to}-12-31",
        "page": page,
    }
    if genre_id:
        params["with_genres"] = genre_id
    if language:
        params["with_original_language"] = language

    data = tmdb_get("/discover/movie", api_key, params)
    if not data:
        return [], 0
    return data.get("results", []), data.get("total_results", 0)


# ── API key management ────────────────────────────────────────────────────────

def load_or_request_api_key() -> str:
    load_dotenv(ENV_FILE)
    key = os.getenv("TMDB_API_KEY", "").strip()

    if key:
        console.print("[dim]Using saved TMDB API key from .env[/dim]")
        return key

    console.print(Panel(
        "[bold cyan]No TMDB API key found.[/bold cyan]\n"
        "Get a free key at [link=https://www.themoviedb.org/settings/api]themoviedb.org/settings/api[/link]\n"
        "Your key will be saved to [bold].env[/bold] for future runs.",
        title="Setup", border_style="cyan"
    ))

    while True:
        key = Prompt.ask("Paste your TMDB API key").strip()
        if not key:
            console.print("[yellow]Key cannot be empty.[/yellow]")
            continue
        console.print("[dim]Validating key…[/dim]")
        if validate_api_key(key):
            ENV_FILE.touch(exist_ok=True)
            set_key(str(ENV_FILE), "TMDB_API_KEY", key)
            console.print("[green]✓ Key valid and saved.[/green]")
            return key
        console.print("[red]Key rejected by TMDB. Try again.[/red]")


# ── UI helpers ────────────────────────────────────────────────────────────────

def ask_filters() -> dict:
    console.print(Panel("[bold]Configure your search[/bold]", border_style="blue"))

    genre_name = questionary.select(
        "Genre:",
        choices=list(GENRES.keys()),
        default="Any",
    ).ask()
    if genre_name is None:
        sys.exit(0)

    lang_name = questionary.select(
        "Original language:",
        choices=list(LANGUAGES.keys()),
        default="Any",
    ).ask()
    if lang_name is None:
        sys.exit(0)

    min_rating_str = questionary.select(
        "Minimum rating:",
        choices=["5.0", "6.0", "7.0", "7.5", "8.0", "8.5"],
        default="7.0",
    ).ask()
    if min_rating_str is None:
        sys.exit(0)

    year_from_str = questionary.text(
        "Release year from:",
        default="2000",
        validate=lambda v: v.isdigit() and 1900 <= int(v) <= 2100 or "Enter a valid year",
    ).ask()
    if year_from_str is None:
        sys.exit(0)

    year_to_str = questionary.text(
        "Release year to:",
        default="2025",
        validate=lambda v: v.isdigit() and 1900 <= int(v) <= 2100 or "Enter a valid year",
    ).ask()
    if year_to_str is None:
        sys.exit(0)

    region = questionary.text(
        "Your region (ISO 3166-1, e.g. US, IN, GB):",
        default="US",
        validate=lambda v: len(v.strip()) == 2 or "Enter a 2-letter country code",
    ).ask()
    if region is None:
        sys.exit(0)

    return {
        "genre_id":   GENRES[genre_name],
        "language":   LANGUAGES[lang_name],
        "min_rating": float(min_rating_str),
        "year_from":  int(year_from_str),
        "year_to":    int(year_to_str),
        "region":     region.upper().strip(),
    }


def render_table(movies: list[dict], providers_map: dict[int, list[str]], offset: int) -> None:
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
        title="[bold cyan]Movie Suggestions[/bold cyan]",
    )
    table.add_column("#",           style="dim",        width=4,  no_wrap=True)
    table.add_column("Title",       style="bold white", min_width=24)
    table.add_column("Year",        justify="center",   width=6)
    table.add_column("Rating",      justify="center",   width=8)
    table.add_column("Votes",       justify="right",    width=8)
    table.add_column("Language",    justify="center",   width=10)
    table.add_column("OTT Platforms", style="green",   min_width=20)

    for i, m in enumerate(movies, start=offset + 1):
        year     = (m.get("release_date") or "")[:4] or "—"
        rating   = f"{m.get('vote_average', 0):.1f}"
        votes    = f"{m.get('vote_count', 0):,}"
        lang     = (m.get("original_language") or "—").upper()
        mid      = m["id"]
        platforms = providers_map.get(mid, [])
        ott_text = ", ".join(platforms[:5]) if platforms else "[dim]Not available[/dim]"
        if len(platforms) > 5:
            ott_text += f" +{len(platforms)-5}"

        table.add_row(str(i), m.get("title", "Unknown"), year, rating, votes, lang, ott_text)

    console.print(table)


# ── main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    console.print(Panel(
        "[bold cyan]🎬 Movie Suggester[/bold cyan]\n"
        "[dim]Powered by The Movie Database (TMDB)[/dim]",
        border_style="cyan",
    ))

    api_key = load_or_request_api_key()
    filters = ask_filters()

    page       = 1
    offset     = 0
    total      = None
    all_movies : list[dict] = []

    while True:
        # ── fetch a TMDB page (20 results) and serve PAGE_SIZE at a time ──
        if not all_movies:
            with console.status("[cyan]Fetching movies…[/cyan]"):
                raw, total = discover_movies(
                    api_key=api_key,
                    genre_id=filters["genre_id"],
                    language=filters["language"],
                    min_rating=filters["min_rating"],
                    year_from=filters["year_from"],
                    year_to=filters["year_to"],
                    page=page,
                )
            if not raw:
                console.print("[yellow]No movies found. Try adjusting your filters.[/yellow]")
                break
            all_movies = raw

        batch     = all_movies[:PAGE_SIZE]
        all_movies = all_movies[PAGE_SIZE:]

        # ── fetch OTT providers for this batch ──
        providers_map: dict[int, list[str]] = {}
        with console.status("[cyan]Fetching OTT availability…[/cyan]"):
            for m in batch:
                providers_map[m["id"]] = get_watch_providers(
                    m["id"], api_key, filters["region"]
                )

        render_table(batch, providers_map, offset)
        offset += len(batch)

        console.print(
            f"[dim]Showing {offset} of ~{total} results "
            f"(filtered, ≥{filters['min_rating']} rating, ≥50 votes)[/dim]"
        )

        # ── decide what's next ──
        choices = []
        has_more_local = bool(all_movies)
        has_more_remote = page * 20 < (total or 0)

        if has_more_local or has_more_remote:
            choices.append("Next 10 movies")
        choices += ["New search", "Quit"]

        action = questionary.select("What next?", choices=choices).ask()
        if action is None or action == "Quit":
            break
        if action == "New search":
            filters    = ask_filters()
            page       = 1
            offset     = 0
            total      = None
            all_movies = []
        elif action == "Next 10 movies":
            if not all_movies:          # local buffer empty → fetch next TMDB page
                page += 1

    console.print("[bold green]Thanks for using Movie Suggester. Enjoy your film! 🍿[/bold green]")


if __name__ == "__main__":
    run()
