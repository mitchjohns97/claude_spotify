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
