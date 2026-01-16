"""Load and join Spotify API cache data with streaming history."""

import ast
from pathlib import Path
from typing import Optional

import pandas as pd


def load_track_cache(cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load track metadata from the API cache.

    Args:
        cache_dir: Path to spotify_cache folder. Defaults to project root.

    Returns:
        DataFrame with track metadata indexed by track_id.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "spotify_cache"

    track_file = cache_dir / "trackLibrary.csv"
    if not track_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(track_file)

    # Ensure track_id is string for joining
    if "track_id" in df.columns:
        df["track_id"] = df["track_id"].astype(str)

    return df


def load_artist_cache(cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load artist metadata from the API cache.

    Args:
        cache_dir: Path to spotify_cache folder. Defaults to project root.

    Returns:
        DataFrame with artist metadata indexed by artist_id.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "spotify_cache"

    artist_file = cache_dir / "artistLibrary.csv"
    if not artist_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(artist_file)

    # Ensure artist_id is string for joining
    if "artist_id" in df.columns:
        df["artist_id"] = df["artist_id"].astype(str)

    # Parse artist_genres from string representation to list
    if "artist_genres" in df.columns:
        df["artist_genres"] = df["artist_genres"].apply(parse_list_string)

    return df


def load_album_cache(cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load album metadata from the API cache.

    Args:
        cache_dir: Path to spotify_cache folder. Defaults to project root.

    Returns:
        DataFrame with album metadata indexed by album_id.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "spotify_cache"

    album_file = cache_dir / "albumLibrary.csv"
    if not album_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(album_file)

    # Ensure album_id is string for joining
    if "album_id" in df.columns:
        df["album_id"] = df["album_id"].astype(str)

    return df


def parse_list_string(value) -> list:
    """
    Parse a string representation of a list into an actual list.

    Handles formats like: "['rock', 'indie']" or "[]"

    Args:
        value: String representation of list, or already a list, or NaN.

    Returns:
        List of strings (empty if parsing fails or input is empty/NaN).
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value or value == "[]":
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        return []


def extract_track_id(uri: str) -> Optional[str]:
    """
    Extract track ID from Spotify URI.

    Args:
        uri: Spotify URI like "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"

    Returns:
        Track ID string, or None if invalid format.
    """
    if pd.isna(uri) or not isinstance(uri, str):
        return None

    parts = uri.split(":")
    if len(parts) == 3 and parts[0] == "spotify" and parts[1] == "track":
        return parts[2]
    return None


def enrich_streaming_data(
    df: pd.DataFrame,
    track_cache: Optional[pd.DataFrame] = None,
    artist_cache: Optional[pd.DataFrame] = None,
    album_cache: Optional[pd.DataFrame] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Enrich streaming history with metadata from API cache.

    Joins on track_id to add: track_popularity, genres, release_date, explicit,
    duration_ms, album_popularity, artist_popularity, artist_followers.

    Args:
        df: Streaming history DataFrame (must have track_uri column).
        track_cache: Track metadata DataFrame (loaded if not provided).
        artist_cache: Artist metadata DataFrame (loaded if not provided).
        album_cache: Album metadata DataFrame (loaded if not provided).
        cache_dir: Path to cache directory.

    Returns:
        Enriched DataFrame with additional metadata columns.
    """
    df = df.copy()

    # Extract track_id from URI
    if "track_uri" in df.columns:
        df["track_id"] = df["track_uri"].apply(extract_track_id)
    elif "track_id" not in df.columns:
        # Can't enrich without track identifier
        return df

    # Load caches if not provided
    if track_cache is None:
        track_cache = load_track_cache(cache_dir)
    if artist_cache is None:
        artist_cache = load_artist_cache(cache_dir)
    if album_cache is None:
        album_cache = load_album_cache(cache_dir)

    # Join track metadata
    if not track_cache.empty and "track_id" in track_cache.columns:
        # Select columns to join from track cache
        track_cols = ["track_id"]
        optional_cols = [
            "track_popularity", "duration_ms", "explicit",
            "album_id", "artist_ids", "album_release_date"
        ]
        for col in optional_cols:
            if col in track_cache.columns:
                track_cols.append(col)

        track_subset = track_cache[track_cols].drop_duplicates(subset=["track_id"])
        df = df.merge(track_subset, on="track_id", how="left")

    # Parse artist_ids and explode to get primary artist for genre lookup
    if "artist_ids" in df.columns:
        df["primary_artist_id"] = df["artist_ids"].apply(_extract_primary_artist_id)

    # Join artist metadata (using primary artist)
    if not artist_cache.empty and "artist_id" in artist_cache.columns and "primary_artist_id" in df.columns:
        artist_cols = ["artist_id"]
        optional_cols = ["artist_genres", "artist_popularity", "artist_followers"]
        for col in optional_cols:
            if col in artist_cache.columns:
                artist_cols.append(col)

        artist_subset = artist_cache[artist_cols].drop_duplicates(subset=["artist_id"])
        df = df.merge(
            artist_subset,
            left_on="primary_artist_id",
            right_on="artist_id",
            how="left",
            suffixes=("", "_cache")
        )
        # Drop duplicate artist_id column if created
        if "artist_id_cache" in df.columns:
            df = df.drop(columns=["artist_id_cache"])
        if "artist_id" in df.columns and "primary_artist_id" in df.columns:
            # Keep artist_id only if it was added by merge
            if df["artist_id"].equals(df["primary_artist_id"]):
                df = df.drop(columns=["artist_id"])

    # Join album metadata
    if not album_cache.empty and "album_id" in album_cache.columns and "album_id" in df.columns:
        album_cols = ["album_id"]
        optional_cols = ["label", "album_popularity", "release_date"]
        for col in optional_cols:
            if col in album_cache.columns:
                album_cols.append(col)

        album_subset = album_cache[album_cols].drop_duplicates(subset=["album_id"])
        df = df.merge(album_subset, on="album_id", how="left", suffixes=("", "_album"))

    # Parse release dates
    if "album_release_date" in df.columns:
        df["release_year"] = df["album_release_date"].apply(_extract_year)
        df["release_decade"] = df["release_year"].apply(
            lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) else None
        )

    return df


def _extract_primary_artist_id(artist_ids_str) -> Optional[str]:
    """Extract the first artist ID from a string list of artist IDs."""
    if pd.isna(artist_ids_str):
        return None
    if isinstance(artist_ids_str, list):
        return artist_ids_str[0] if artist_ids_str else None
    if not isinstance(artist_ids_str, str):
        return None

    # Parse string like "['id1', 'id2']" or just "id1"
    parsed = parse_list_string(artist_ids_str)
    if parsed:
        return parsed[0]

    # Maybe it's just a plain ID string
    artist_ids_str = artist_ids_str.strip()
    if artist_ids_str and not artist_ids_str.startswith("["):
        return artist_ids_str

    return None


def _extract_year(date_str) -> Optional[int]:
    """Extract year from a date string (YYYY, YYYY-MM, or YYYY-MM-DD)."""
    if pd.isna(date_str):
        return None
    if not isinstance(date_str, str):
        return None

    date_str = date_str.strip()
    if not date_str:
        return None

    # Handle various date formats
    try:
        # Try to get the year part (first 4 characters should be year)
        year_str = date_str[:4]
        year = int(year_str)
        if 1900 <= year <= 2100:
            return year
    except (ValueError, IndexError):
        pass

    return None


def get_enrichment_stats(df: pd.DataFrame) -> dict:
    """
    Get statistics about enrichment coverage.

    Args:
        df: Enriched streaming history DataFrame.

    Returns:
        Dictionary with enrichment statistics.
    """
    total = len(df)

    stats = {
        "total_streams": total,
        "with_track_id": df["track_id"].notna().sum() if "track_id" in df.columns else 0,
        "with_popularity": df["track_popularity"].notna().sum() if "track_popularity" in df.columns else 0,
        "with_genres": 0,
        "with_release_year": df["release_year"].notna().sum() if "release_year" in df.columns else 0,
        "with_explicit": df["explicit"].notna().sum() if "explicit" in df.columns else 0,
    }

    # Count streams with at least one genre
    if "artist_genres" in df.columns:
        stats["with_genres"] = df["artist_genres"].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        ).sum()

    # Calculate percentages
    if total > 0:
        for key in list(stats.keys()):
            if key != "total_streams":
                stats[f"{key}_pct"] = stats[key] / total * 100

    return stats
