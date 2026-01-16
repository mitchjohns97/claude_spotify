"""Statistical analysis functions for Spotify streaming history."""

import pandas as pd


def top_artists(df: pd.DataFrame, limit: int = 20, by: str = "time") -> pd.DataFrame:
    """
    Get top artists by listening time or play count.

    Args:
        df: Streaming history DataFrame.
        limit: Number of artists to return.
        by: Sort by 'time' (total hours) or 'count' (play count).

    Returns:
        DataFrame with artist stats.
    """
    grouped = df.groupby("artist_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
        unique_tracks=("track_name", "nunique"),
    ).reset_index()

    sort_col = "total_hours" if by == "time" else "play_count"
    grouped = grouped.sort_values(sort_col, ascending=False).head(limit)

    return grouped.reset_index(drop=True)


def top_tracks(df: pd.DataFrame, limit: int = 20, by: str = "time") -> pd.DataFrame:
    """
    Get top tracks by listening time or play count.

    Args:
        df: Streaming history DataFrame.
        limit: Number of tracks to return.
        by: Sort by 'time' (total hours) or 'count' (play count).

    Returns:
        DataFrame with track stats.
    """
    grouped = df.groupby(["track_name", "artist_name"]).agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    sort_col = "total_hours" if by == "time" else "play_count"
    grouped = grouped.sort_values(sort_col, ascending=False).head(limit)

    return grouped.reset_index(drop=True)


def top_albums(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """
    Get top albums by listening time.

    Args:
        df: Streaming history DataFrame.
        limit: Number of albums to return.

    Returns:
        DataFrame with album stats.
    """
    grouped = df.groupby(["album_name", "artist_name"]).agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
        unique_tracks=("track_name", "nunique"),
    ).reset_index()

    grouped = grouped.sort_values("total_hours", ascending=False).head(limit)

    return grouped.reset_index(drop=True)


def listening_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Get listening hours and play counts by year."""
    grouped = df.groupby("year").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
        unique_tracks=("track_name", "nunique"),
        unique_artists=("artist_name", "nunique"),
    ).reset_index()

    return grouped.sort_values("year")


def listening_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Get listening hours by year-month."""
    df = df.copy()
    df["year_month"] = df["timestamp"].dt.to_period("M")

    grouped = df.groupby("year_month").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped["year_month"] = grouped["year_month"].astype(str)

    return grouped


def listening_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Get listening patterns by hour of day."""
    grouped = df.groupby("hour").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    return grouped.sort_values("hour")


def listening_by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Get listening patterns by day of week."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    grouped = df.groupby("day_of_week").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped["day_order"] = grouped["day_of_week"].map({d: i for i, d in enumerate(day_order)})
    grouped = grouped.sort_values("day_order").drop(columns=["day_order"])

    return grouped


def listening_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get listening data for hour x day-of-week heatmap."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    grouped = df.groupby(["day_of_week", "hour"]).agg(
        total_hours=("hours_played", "sum"),
    ).reset_index()

    # Create pivot table
    pivot = grouped.pivot(index="day_of_week", columns="hour", values="total_hours")
    pivot = pivot.fillna(0)

    # Reorder days
    pivot = pivot.reindex(day_order)

    return pivot


def platform_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Get listening stats by platform."""
    if "platform" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("platform").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped = grouped.sort_values("total_hours", ascending=False)
    grouped["percentage"] = grouped["total_hours"] / grouped["total_hours"].sum() * 100

    return grouped


def skip_analysis(df: pd.DataFrame, limit: int = 20) -> dict:
    """Analyze skip behavior."""
    if "skipped" not in df.columns:
        return {"artists": pd.DataFrame(), "tracks": pd.DataFrame()}

    # Filter to skipped tracks
    skipped = df[df["skipped"] == True]

    # Most skipped artists
    skipped_artists = skipped.groupby("artist_name").size().reset_index(name="skip_count")
    skipped_artists = skipped_artists.sort_values("skip_count", ascending=False).head(limit)

    # Calculate skip rate by artist
    total_by_artist = df.groupby("artist_name").size().reset_index(name="total_plays")
    skipped_artists = skipped_artists.merge(total_by_artist, on="artist_name")
    skipped_artists["skip_rate"] = skipped_artists["skip_count"] / skipped_artists["total_plays"] * 100

    # Most skipped tracks
    skipped_tracks = skipped.groupby(["track_name", "artist_name"]).size().reset_index(name="skip_count")
    skipped_tracks = skipped_tracks.sort_values("skip_count", ascending=False).head(limit)

    return {
        "artists": skipped_artists,
        "tracks": skipped_tracks,
    }


def listening_streaks(df: pd.DataFrame) -> dict:
    """Calculate listening streaks (consecutive days with activity)."""
    if "date" not in df.columns:
        return {"longest_streak": 0, "current_streak": 0}

    # Get unique listening dates
    dates = pd.Series(df["date"].unique()).sort_values()
    dates = pd.to_datetime(dates)

    if len(dates) == 0:
        return {"longest_streak": 0, "current_streak": 0}

    # Calculate gaps between consecutive dates
    date_diffs = dates.diff().dt.days

    # Find streaks
    streaks = []
    current_streak = 1

    for diff in date_diffs[1:]:
        if diff == 1:
            current_streak += 1
        else:
            streaks.append(current_streak)
            current_streak = 1

    streaks.append(current_streak)

    longest_streak = max(streaks) if streaks else 0

    # Current streak (from most recent date)
    if len(dates) > 0:
        last_date = dates.iloc[-1]
        today = pd.Timestamp.now().normalize()
        days_since_last = (today - last_date).days

        if days_since_last <= 1:
            current = 1
            for diff in reversed(date_diffs[1:].tolist()):
                if diff == 1:
                    current += 1
                else:
                    break
        else:
            current = 0
    else:
        current = 0

    return {
        "longest_streak": longest_streak,
        "current_streak": current,
        "total_listening_days": len(dates),
    }


def yearly_top_artists(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Get top artists for each year."""
    results = []

    for year in df["year"].unique():
        year_df = df[df["year"] == year]
        top = top_artists(year_df, limit=top_n, by="time")
        top["year"] = year
        results.append(top)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Generate comprehensive summary statistics."""
    stats = {
        "total_streams": len(df),
        "total_hours": df["hours_played"].sum(),
        "total_days": df["hours_played"].sum() / 24,
        "unique_tracks": df["track_name"].nunique(),
        "unique_artists": df["artist_name"].nunique(),
        "unique_albums": df["album_name"].nunique(),
    }

    if "timestamp" in df.columns and len(df) > 0:
        stats["first_stream"] = df["timestamp"].min()
        stats["last_stream"] = df["timestamp"].max()
        stats["years_span"] = (stats["last_stream"] - stats["first_stream"]).days / 365.25

    streaks = listening_streaks(df)
    stats.update(streaks)

    # Average stats
    if stats["years_span"] > 0:
        stats["avg_hours_per_year"] = stats["total_hours"] / stats["years_span"]
        stats["avg_hours_per_day"] = stats["total_hours"] / stats["total_listening_days"] if stats["total_listening_days"] > 0 else 0

    return stats


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """
    Filter DataFrame to a specific date range.

    Args:
        df: Streaming history DataFrame.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).

    Returns:
        Filtered DataFrame.
    """
    if "timestamp" not in df.columns:
        return df

    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    return df[mask].copy()


def artist_deep_dive(df: pd.DataFrame, artist_name: str) -> dict:
    """
    Get detailed statistics for a specific artist.

    Args:
        df: Streaming history DataFrame.
        artist_name: Name of the artist to analyze.

    Returns:
        Dictionary containing artist statistics.
    """
    artist_df = df[df["artist_name"] == artist_name]

    if artist_df.empty:
        return {
            "artist_name": artist_name,
            "total_hours": 0,
            "play_count": 0,
            "first_listen": None,
            "last_listen": None,
            "top_tracks": pd.DataFrame(),
            "top_albums": pd.DataFrame(),
        }

    # Basic stats
    total_hours = artist_df["hours_played"].sum()
    play_count = len(artist_df)
    first_listen = artist_df["timestamp"].min()
    last_listen = artist_df["timestamp"].max()

    # Top tracks for this artist
    top_tracks_df = artist_df.groupby("track_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()
    top_tracks_df = top_tracks_df.sort_values("total_hours", ascending=False).head(10)
    top_tracks_df = top_tracks_df.reset_index(drop=True)

    # Top albums for this artist
    top_albums_df = artist_df.groupby("album_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("album_name", "count"),
        unique_tracks=("track_name", "nunique"),
    ).reset_index()
    top_albums_df = top_albums_df.sort_values("total_hours", ascending=False).head(5)
    top_albums_df = top_albums_df.reset_index(drop=True)

    return {
        "artist_name": artist_name,
        "total_hours": total_hours,
        "play_count": play_count,
        "first_listen": first_listen,
        "last_listen": last_listen,
        "top_tracks": top_tracks_df,
        "top_albums": top_albums_df,
    }


def artist_timeline(df: pd.DataFrame, artist_name: str) -> pd.DataFrame:
    """
    Get monthly listening timeline for a specific artist.

    Args:
        df: Streaming history DataFrame.
        artist_name: Name of the artist.

    Returns:
        DataFrame with monthly listening hours for the artist.
    """
    artist_df = df[df["artist_name"] == artist_name].copy()

    if artist_df.empty:
        return pd.DataFrame(columns=["year_month", "total_hours", "play_count"])

    artist_df["year_month"] = artist_df["timestamp"].dt.to_period("M")

    grouped = artist_df.groupby("year_month").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped["year_month"] = grouped["year_month"].astype(str)

    return grouped


def song_deep_dive(df: pd.DataFrame, track_name: str, artist_name: str) -> dict:
    """
    Get detailed statistics for a specific song.

    Args:
        df: Streaming history DataFrame.
        track_name: Name of the track to analyze.
        artist_name: Name of the artist.

    Returns:
        Dictionary containing song statistics.
    """
    song_df = df[(df["track_name"] == track_name) & (df["artist_name"] == artist_name)]

    if song_df.empty:
        return {
            "track_name": track_name,
            "artist_name": artist_name,
            "total_hours": 0,
            "play_count": 0,
            "first_listen": None,
            "last_listen": None,
            "albums": pd.DataFrame(),
        }

    # Basic stats
    total_hours = song_df["hours_played"].sum()
    play_count = len(song_df)
    first_listen = song_df["timestamp"].min()
    last_listen = song_df["timestamp"].max()

    # Albums containing this track
    albums_df = song_df.groupby("album_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()
    albums_df = albums_df.sort_values("total_hours", ascending=False)
    albums_df = albums_df.reset_index(drop=True)

    return {
        "track_name": track_name,
        "artist_name": artist_name,
        "total_hours": total_hours,
        "play_count": play_count,
        "first_listen": first_listen,
        "last_listen": last_listen,
        "albums": albums_df,
    }


def song_timeline(df: pd.DataFrame, track_name: str, artist_name: str) -> pd.DataFrame:
    """
    Get monthly listening timeline for a specific song.

    Args:
        df: Streaming history DataFrame.
        track_name: Name of the track.
        artist_name: Name of the artist.

    Returns:
        DataFrame with monthly listening hours for the song.
    """
    song_df = df[(df["track_name"] == track_name) & (df["artist_name"] == artist_name)].copy()

    if song_df.empty:
        return pd.DataFrame(columns=["year_month", "total_hours", "play_count"])

    song_df["year_month"] = song_df["timestamp"].dt.to_period("M")

    grouped = song_df.groupby("year_month").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped["year_month"] = grouped["year_month"].astype(str)

    return grouped


def album_deep_dive(df: pd.DataFrame, album_name: str, artist_name: str) -> dict:
    """
    Get detailed statistics for a specific album.

    Args:
        df: Streaming history DataFrame.
        album_name: Name of the album to analyze.
        artist_name: Name of the artist.

    Returns:
        Dictionary containing album statistics.
    """
    album_df = df[(df["album_name"] == album_name) & (df["artist_name"] == artist_name)]

    if album_df.empty:
        return {
            "album_name": album_name,
            "artist_name": artist_name,
            "total_hours": 0,
            "play_count": 0,
            "first_listen": None,
            "last_listen": None,
            "unique_tracks": 0,
            "top_tracks": pd.DataFrame(),
        }

    # Basic stats
    total_hours = album_df["hours_played"].sum()
    play_count = len(album_df)
    first_listen = album_df["timestamp"].min()
    last_listen = album_df["timestamp"].max()
    unique_tracks = album_df["track_name"].nunique()

    # Top tracks from this album
    top_tracks_df = album_df.groupby("track_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()
    top_tracks_df = top_tracks_df.sort_values("total_hours", ascending=False).head(10)
    top_tracks_df = top_tracks_df.reset_index(drop=True)

    return {
        "album_name": album_name,
        "artist_name": artist_name,
        "total_hours": total_hours,
        "play_count": play_count,
        "first_listen": first_listen,
        "last_listen": last_listen,
        "unique_tracks": unique_tracks,
        "top_tracks": top_tracks_df,
    }


def album_timeline(df: pd.DataFrame, album_name: str, artist_name: str) -> pd.DataFrame:
    """
    Get monthly listening timeline for a specific album.

    Args:
        df: Streaming history DataFrame.
        album_name: Name of the album.
        artist_name: Name of the artist.

    Returns:
        DataFrame with monthly listening hours for the album.
    """
    album_df = df[(df["album_name"] == album_name) & (df["artist_name"] == artist_name)].copy()

    if album_df.empty:
        return pd.DataFrame(columns=["year_month", "total_hours", "play_count"])

    album_df["year_month"] = album_df["timestamp"].dt.to_period("M")

    grouped = album_df.groupby("year_month").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped["year_month"] = grouped["year_month"].astype(str)

    return grouped


# =============================================================================
# Genre Analysis Functions
# =============================================================================

def top_genres(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """
    Get top genres by listening time.

    Requires 'artist_genres' column from enrichment.

    Args:
        df: Enriched streaming history DataFrame.
        limit: Number of genres to return.

    Returns:
        DataFrame with genre, total_hours, play_count, unique_artists.
    """
    if "artist_genres" not in df.columns:
        return pd.DataFrame(columns=["genre", "total_hours", "play_count", "unique_artists"])

    # Explode genres to get one row per genre
    df_exploded = df.explode("artist_genres")
    df_exploded = df_exploded[df_exploded["artist_genres"].notna()]
    df_exploded = df_exploded[df_exploded["artist_genres"] != ""]

    if df_exploded.empty:
        return pd.DataFrame(columns=["genre", "total_hours", "play_count", "unique_artists"])

    grouped = df_exploded.groupby("artist_genres").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
        unique_artists=("artist_name", "nunique"),
    ).reset_index()

    grouped = grouped.rename(columns={"artist_genres": "genre"})
    grouped = grouped.sort_values("total_hours", ascending=False).head(limit)

    return grouped.reset_index(drop=True)


def genre_by_year(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Get genre trends over time (top genres per year).

    Args:
        df: Enriched streaming history DataFrame.
        limit: Number of top genres to track.

    Returns:
        DataFrame with year, genre, total_hours (pivotable for charts).
    """
    if "artist_genres" not in df.columns or "year" not in df.columns:
        return pd.DataFrame(columns=["year", "genre", "total_hours"])

    # Get overall top genres to focus on
    overall_top = top_genres(df, limit=limit)
    if overall_top.empty:
        return pd.DataFrame(columns=["year", "genre", "total_hours"])

    top_genre_list = overall_top["genre"].tolist()

    # Explode genres
    df_exploded = df.explode("artist_genres")
    df_exploded = df_exploded[df_exploded["artist_genres"].isin(top_genre_list)]

    if df_exploded.empty:
        return pd.DataFrame(columns=["year", "genre", "total_hours"])

    grouped = df_exploded.groupby(["year", "artist_genres"]).agg(
        total_hours=("hours_played", "sum"),
    ).reset_index()

    grouped = grouped.rename(columns={"artist_genres": "genre"})

    return grouped.sort_values(["year", "total_hours"], ascending=[True, False])


def popularity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get distribution of track popularity scores.

    Args:
        df: Enriched streaming history DataFrame.

    Returns:
        DataFrame with popularity_bucket, count, total_hours.
    """
    if "track_popularity" not in df.columns:
        return pd.DataFrame(columns=["popularity_bucket", "count", "total_hours"])

    df_with_pop = df[df["track_popularity"].notna()].copy()
    if df_with_pop.empty:
        return pd.DataFrame(columns=["popularity_bucket", "count", "total_hours"])

    # Create buckets: 0-9, 10-19, ..., 90-100
    df_with_pop["popularity_bucket"] = (df_with_pop["track_popularity"] // 10 * 10).astype(int)
    df_with_pop["popularity_bucket"] = df_with_pop["popularity_bucket"].apply(
        lambda x: f"{x}-{min(x+9, 100)}"
    )

    grouped = df_with_pop.groupby("popularity_bucket").agg(
        count=("track_name", "count"),
        total_hours=("hours_played", "sum"),
    ).reset_index()

    # Sort by bucket numerically
    grouped["sort_key"] = grouped["popularity_bucket"].str.extract(r"(\d+)").astype(int)
    grouped = grouped.sort_values("sort_key").drop(columns=["sort_key"])

    return grouped.reset_index(drop=True)


def discovery_score(df: pd.DataFrame, threshold: int = 30) -> dict:
    """
    Calculate discovery score - percentage of listening to low-popularity tracks.

    Args:
        df: Enriched streaming history DataFrame.
        threshold: Popularity threshold for "discovery" (default 30).

    Returns:
        Dictionary with discovery metrics.
    """
    if "track_popularity" not in df.columns:
        return {
            "discovery_score": None,
            "discovery_hours": 0,
            "total_hours_with_data": 0,
            "discovery_count": 0,
            "total_with_data": 0,
        }

    df_with_pop = df[df["track_popularity"].notna()]
    if df_with_pop.empty:
        return {
            "discovery_score": None,
            "discovery_hours": 0,
            "total_hours_with_data": 0,
            "discovery_count": 0,
            "total_with_data": 0,
        }

    discovery_mask = df_with_pop["track_popularity"] < threshold
    discovery_hours = df_with_pop.loc[discovery_mask, "hours_played"].sum()
    total_hours = df_with_pop["hours_played"].sum()
    discovery_count = discovery_mask.sum()
    total_count = len(df_with_pop)

    return {
        "discovery_score": (discovery_hours / total_hours * 100) if total_hours > 0 else 0,
        "discovery_hours": discovery_hours,
        "total_hours_with_data": total_hours,
        "discovery_count": discovery_count,
        "total_with_data": total_count,
    }


def mainstream_vs_indie(df: pd.DataFrame, threshold: int = 50) -> dict:
    """
    Compare listening to mainstream vs indie tracks.

    Args:
        df: Enriched streaming history DataFrame.
        threshold: Popularity threshold separating mainstream from indie.

    Returns:
        Dictionary with mainstream vs indie metrics.
    """
    if "track_popularity" not in df.columns:
        return {
            "mainstream_hours": 0,
            "indie_hours": 0,
            "mainstream_pct": None,
            "indie_pct": None,
        }

    df_with_pop = df[df["track_popularity"].notna()]
    if df_with_pop.empty:
        return {
            "mainstream_hours": 0,
            "indie_hours": 0,
            "mainstream_pct": None,
            "indie_pct": None,
        }

    mainstream_mask = df_with_pop["track_popularity"] >= threshold
    mainstream_hours = df_with_pop.loc[mainstream_mask, "hours_played"].sum()
    indie_hours = df_with_pop.loc[~mainstream_mask, "hours_played"].sum()
    total = mainstream_hours + indie_hours

    return {
        "mainstream_hours": mainstream_hours,
        "indie_hours": indie_hours,
        "mainstream_pct": (mainstream_hours / total * 100) if total > 0 else 0,
        "indie_pct": (indie_hours / total * 100) if total > 0 else 0,
    }


def most_obscure_favorites(df: pd.DataFrame, min_plays: int = 5, limit: int = 20) -> pd.DataFrame:
    """
    Find tracks with low popularity but high play count (hidden gems).

    Args:
        df: Enriched streaming history DataFrame.
        min_plays: Minimum play count to consider.
        limit: Number of tracks to return.

    Returns:
        DataFrame with track_name, artist_name, track_popularity, play_count, total_hours.
    """
    if "track_popularity" not in df.columns:
        return pd.DataFrame(columns=["track_name", "artist_name", "track_popularity", "play_count", "total_hours"])

    # Group by track
    grouped = df.groupby(["track_name", "artist_name"]).agg(
        play_count=("track_name", "count"),
        total_hours=("hours_played", "sum"),
        track_popularity=("track_popularity", "first"),
    ).reset_index()

    # Filter to tracks with data and minimum plays
    grouped = grouped[grouped["track_popularity"].notna()]
    grouped = grouped[grouped["play_count"] >= min_plays]

    # Sort by lowest popularity first
    grouped = grouped.sort_values(["track_popularity", "play_count"], ascending=[True, False])

    return grouped.head(limit).reset_index(drop=True)


def hipster_score(df: pd.DataFrame) -> dict:
    """
    Calculate "hipster score" - how much you listen to unpopular music.

    Based on weighted average of track popularity, inverted.

    Args:
        df: Enriched streaming history DataFrame.

    Returns:
        Dictionary with hipster metrics.
    """
    if "track_popularity" not in df.columns:
        return {
            "hipster_score": None,
            "avg_popularity": None,
            "median_popularity": None,
        }

    df_with_pop = df[df["track_popularity"].notna()]
    if df_with_pop.empty:
        return {
            "hipster_score": None,
            "avg_popularity": None,
            "median_popularity": None,
        }

    # Weighted average popularity (weighted by hours played)
    weighted_avg = (
        (df_with_pop["track_popularity"] * df_with_pop["hours_played"]).sum()
        / df_with_pop["hours_played"].sum()
    )

    # Hipster score: 100 - weighted_avg (higher = more hipster)
    hipster = 100 - weighted_avg

    return {
        "hipster_score": hipster,
        "avg_popularity": weighted_avg,
        "median_popularity": df_with_pop["track_popularity"].median(),
    }


def genre_deep_dive(df: pd.DataFrame, genre: str) -> dict:
    """
    Get detailed statistics for a specific genre.

    Args:
        df: Enriched streaming history DataFrame.
        genre: Genre to analyze.

    Returns:
        Dictionary with genre statistics.
    """
    if "artist_genres" not in df.columns:
        return {
            "genre": genre,
            "total_hours": 0,
            "play_count": 0,
            "unique_artists": 0,
            "unique_tracks": 0,
            "top_artists": pd.DataFrame(),
            "top_tracks": pd.DataFrame(),
        }

    # Filter to streams containing this genre
    genre_mask = df["artist_genres"].apply(
        lambda x: isinstance(x, list) and genre in x
    )
    genre_df = df[genre_mask]

    if genre_df.empty:
        return {
            "genre": genre,
            "total_hours": 0,
            "play_count": 0,
            "unique_artists": 0,
            "unique_tracks": 0,
            "top_artists": pd.DataFrame(),
            "top_tracks": pd.DataFrame(),
        }

    # Top artists in this genre
    top_artists_df = genre_df.groupby("artist_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()
    top_artists_df = top_artists_df.sort_values("total_hours", ascending=False).head(10)

    # Top tracks in this genre
    top_tracks_df = genre_df.groupby(["track_name", "artist_name"]).agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()
    top_tracks_df = top_tracks_df.sort_values("total_hours", ascending=False).head(10)

    return {
        "genre": genre,
        "total_hours": genre_df["hours_played"].sum(),
        "play_count": len(genre_df),
        "unique_artists": genre_df["artist_name"].nunique(),
        "unique_tracks": genre_df["track_name"].nunique(),
        "top_artists": top_artists_df.reset_index(drop=True),
        "top_tracks": top_tracks_df.reset_index(drop=True),
    }


def explicit_content_stats(df: pd.DataFrame) -> dict:
    """
    Get explicit vs clean content breakdown.

    Args:
        df: Enriched streaming history DataFrame.

    Returns:
        Dictionary with explicit/clean statistics.
    """
    if "explicit" not in df.columns:
        return {
            "explicit_hours": 0,
            "clean_hours": 0,
            "explicit_pct": None,
            "explicit_count": 0,
            "clean_count": 0,
        }

    df_with_explicit = df[df["explicit"].notna()]
    if df_with_explicit.empty:
        return {
            "explicit_hours": 0,
            "clean_hours": 0,
            "explicit_pct": None,
            "explicit_count": 0,
            "clean_count": 0,
        }

    explicit_mask = df_with_explicit["explicit"] == True
    explicit_hours = df_with_explicit.loc[explicit_mask, "hours_played"].sum()
    clean_hours = df_with_explicit.loc[~explicit_mask, "hours_played"].sum()
    total = explicit_hours + clean_hours

    return {
        "explicit_hours": explicit_hours,
        "clean_hours": clean_hours,
        "explicit_pct": (explicit_hours / total * 100) if total > 0 else 0,
        "explicit_count": explicit_mask.sum(),
        "clean_count": (~explicit_mask).sum(),
    }


def decade_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get listening breakdown by release decade.

    Args:
        df: Enriched streaming history DataFrame.

    Returns:
        DataFrame with decade, total_hours, play_count, unique_tracks.
    """
    if "release_decade" not in df.columns:
        return pd.DataFrame(columns=["decade", "total_hours", "play_count", "unique_tracks"])

    df_with_decade = df[df["release_decade"].notna()]
    if df_with_decade.empty:
        return pd.DataFrame(columns=["decade", "total_hours", "play_count", "unique_tracks"])

    grouped = df_with_decade.groupby("release_decade").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
        unique_tracks=("track_name", "nunique"),
    ).reset_index()

    grouped = grouped.rename(columns={"release_decade": "decade"})

    # Sort by decade
    grouped["sort_key"] = grouped["decade"].str.extract(r"(\d+)").astype(int)
    grouped = grouped.sort_values("sort_key").drop(columns=["sort_key"])

    return grouped.reset_index(drop=True)


def release_recency_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze how "current" your listening is (new releases vs catalog).

    Args:
        df: Enriched streaming history DataFrame.

    Returns:
        Dictionary with recency metrics.
    """
    if "release_year" not in df.columns or "year" not in df.columns:
        return {
            "avg_release_age": None,
            "current_year_pct": None,
            "last_5_years_pct": None,
            "catalog_pct": None,
        }

    df_with_release = df[df["release_year"].notna()].copy()
    if df_with_release.empty:
        return {
            "avg_release_age": None,
            "current_year_pct": None,
            "last_5_years_pct": None,
            "catalog_pct": None,
        }

    # Calculate age at time of listen
    df_with_release["release_age"] = df_with_release["year"] - df_with_release["release_year"]

    # Weighted average age
    total_hours = df_with_release["hours_played"].sum()
    weighted_age = (
        (df_with_release["release_age"] * df_with_release["hours_played"]).sum()
        / total_hours
    ) if total_hours > 0 else 0

    # Breakdown by age category
    current_year = df_with_release.loc[
        df_with_release["release_age"] <= 0, "hours_played"
    ].sum()
    last_5_years = df_with_release.loc[
        df_with_release["release_age"] <= 5, "hours_played"
    ].sum()
    catalog = df_with_release.loc[
        df_with_release["release_age"] > 5, "hours_played"
    ].sum()

    return {
        "avg_release_age": weighted_age,
        "current_year_pct": (current_year / total_hours * 100) if total_hours > 0 else 0,
        "last_5_years_pct": (last_5_years / total_hours * 100) if total_hours > 0 else 0,
        "catalog_pct": (catalog / total_hours * 100) if total_hours > 0 else 0,
    }


def artist_loyalty(df: pd.DataFrame, min_listens: int = 10) -> pd.DataFrame:
    """
    Analyze artist loyalty over time (how long you've been listening to each artist).

    Args:
        df: Streaming history DataFrame.
        min_listens: Minimum listens to include artist.

    Returns:
        DataFrame with artist loyalty metrics.
    """
    grouped = df.groupby("artist_name").agg(
        first_listen=("timestamp", "min"),
        last_listen=("timestamp", "max"),
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped = grouped[grouped["play_count"] >= min_listens]

    if grouped.empty:
        return pd.DataFrame(columns=["artist_name", "loyalty_days", "total_hours", "play_count"])

    # Calculate loyalty span in days
    grouped["loyalty_days"] = (
        grouped["last_listen"] - grouped["first_listen"]
    ).dt.days

    grouped = grouped.sort_values("loyalty_days", ascending=False)

    return grouped[["artist_name", "loyalty_days", "total_hours", "play_count", "first_listen", "last_listen"]].reset_index(drop=True)


def year_over_year_comparison(df: pd.DataFrame, year1: int, year2: int) -> dict:
    """
    Compare listening stats between two years.

    Args:
        df: Streaming history DataFrame.
        year1: First year to compare.
        year2: Second year to compare.

    Returns:
        Dictionary with stats for each year, deltas, and artist changes.
    """
    y1_df = df[df["year"] == year1]
    y2_df = df[df["year"] == year2]

    def get_year_stats(year_df):
        if year_df.empty:
            return None
        top_artist_hours = year_df.groupby("artist_name")["hours_played"].sum()
        top_track_hours = year_df.groupby(["track_name", "artist_name"])["hours_played"].sum()
        return {
            "total_hours": year_df["hours_played"].sum(),
            "total_streams": len(year_df),
            "unique_artists": year_df["artist_name"].nunique(),
            "unique_tracks": year_df["track_name"].nunique(),
            "unique_albums": year_df["album_name"].nunique(),
            "top_artist": top_artist_hours.idxmax() if not top_artist_hours.empty else None,
            "top_artist_hours": top_artist_hours.max() if not top_artist_hours.empty else 0,
            "top_track": top_track_hours.idxmax() if not top_track_hours.empty else None,
            "top_track_hours": top_track_hours.max() if not top_track_hours.empty else 0,
        }

    y1_stats = get_year_stats(y1_df)
    y2_stats = get_year_stats(y2_df)

    # Calculate deltas
    deltas = {}
    if y1_stats and y2_stats:
        for key in ["total_hours", "total_streams", "unique_artists", "unique_tracks", "unique_albums"]:
            old_val = y1_stats[key]
            new_val = y2_stats[key]
            if old_val > 0:
                deltas[key] = ((new_val - old_val) / old_val) * 100
            else:
                deltas[key] = None

    # Find artists gained/lost
    y1_artists = set(y1_df["artist_name"].unique()) if not y1_df.empty else set()
    y2_artists = set(y2_df["artist_name"].unique()) if not y2_df.empty else set()

    # Get top new artists (most listened in year2 that weren't in year1)
    new_artist_names = y2_artists - y1_artists
    if new_artist_names and not y2_df.empty:
        new_artists_df = y2_df[y2_df["artist_name"].isin(new_artist_names)]
        top_new_artists = new_artists_df.groupby("artist_name")["hours_played"].sum()
        top_new_artists = top_new_artists.nlargest(5).reset_index()
        top_new_artists.columns = ["artist_name", "hours"]
    else:
        top_new_artists = pd.DataFrame(columns=["artist_name", "hours"])

    # Get dropped artists (most listened in year1 that aren't in year2)
    dropped_artist_names = y1_artists - y2_artists
    if dropped_artist_names and not y1_df.empty:
        dropped_artists_df = y1_df[y1_df["artist_name"].isin(dropped_artist_names)]
        top_dropped_artists = dropped_artists_df.groupby("artist_name")["hours_played"].sum()
        top_dropped_artists = top_dropped_artists.nlargest(5).reset_index()
        top_dropped_artists.columns = ["artist_name", "hours"]
    else:
        top_dropped_artists = pd.DataFrame(columns=["artist_name", "hours"])

    return {
        "year1": year1,
        "year2": year2,
        "year1_stats": y1_stats,
        "year2_stats": y2_stats,
        "deltas": deltas,
        "new_artists_count": len(y2_artists - y1_artists),
        "dropped_artists_count": len(y1_artists - y2_artists),
        "consistent_artists_count": len(y1_artists & y2_artists),
        "top_new_artists": top_new_artists,
        "top_dropped_artists": top_dropped_artists,
    }


def seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze seasonal listening patterns by month.

    Args:
        df: Streaming history DataFrame.

    Returns:
        DataFrame with month, total_hours, play_count (averaged across years).
    """
    if "month" not in df.columns:
        return pd.DataFrame(columns=["month", "total_hours", "play_count"])

    # Get unique years of data
    years = df["year"].nunique()
    if years == 0:
        return pd.DataFrame(columns=["month", "total_hours", "play_count"])

    grouped = df.groupby("month").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    # Average per year
    grouped["avg_hours"] = grouped["total_hours"] / years
    grouped["avg_plays"] = grouped["play_count"] / years

    # Add month names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    grouped["month_name"] = grouped["month"].map(month_names)

    return grouped.sort_values("month")


def album_completion_rate(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """
    Calculate album completion rates (tracks played / total tracks).

    Requires enriched data with 'album_id' to join with album cache.

    Args:
        df: Streaming history DataFrame (preferably enriched).
        limit: Number of albums to return.

    Returns:
        DataFrame with album completion statistics.
    """
    if "album_id" not in df.columns:
        return pd.DataFrame()

    # Load album cache for total_tracks
    from src.enrichment import load_album_cache
    album_cache = load_album_cache()

    # Get unique tracks played per album
    played = df.groupby(["album_name", "artist_name", "album_id"]).agg(
        tracks_played=("track_name", "nunique"),
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    # Join with album cache to get total_tracks
    if not album_cache.empty and "total_tracks" in album_cache.columns:
        played = played.merge(
            album_cache[["album_id", "total_tracks"]],
            on="album_id",
            how="left"
        )
        # Calculate completion rate
        played["completion_rate"] = (
            played["tracks_played"] / played["total_tracks"] * 100
        ).clip(0, 100)
    else:
        played["total_tracks"] = None
        played["completion_rate"] = None

    # Sort by hours listened
    played = played.sort_values("total_hours", ascending=False).head(limit)

    return played.reset_index(drop=True)


def album_completion_summary(df: pd.DataFrame) -> dict:
    """
    Get summary stats for album completion.

    Args:
        df: Streaming history DataFrame (preferably enriched).

    Returns:
        Dictionary with completion summary statistics.
    """
    completion = album_completion_rate(df, limit=1000)

    if completion.empty or "completion_rate" not in completion.columns:
        return {
            "avg_completion": None,
            "full_albums": 0,
            "partial_albums": 0,
            "single_track_albums": 0,
            "total_albums": 0,
        }

    valid = completion[completion["completion_rate"].notna()]

    if valid.empty:
        return {
            "avg_completion": None,
            "full_albums": 0,
            "partial_albums": 0,
            "single_track_albums": 0,
            "total_albums": 0,
        }

    return {
        "avg_completion": valid["completion_rate"].mean(),
        "full_albums": (valid["completion_rate"] >= 100).sum(),
        "partial_albums": ((valid["completion_rate"] > 0) & (valid["completion_rate"] < 100)).sum(),
        "single_track_albums": (valid["tracks_played"] == 1).sum(),
        "total_albums": len(valid),
    }
