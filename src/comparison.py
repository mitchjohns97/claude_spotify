"""Cross-person comparison functions for Spotify streaming analysis."""

from typing import Optional
import pandas as pd
import numpy as np


def compare_top_artists(df: pd.DataFrame, people: Optional[list[str]] = None, limit: int = 20) -> pd.DataFrame:
    """
    Compare top artists across multiple people.

    Args:
        df: Streaming history DataFrame with 'person' column.
        people: List of people to compare (None for all).
        limit: Number of top artists per person.

    Returns:
        DataFrame with artist_name and hours for each person.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    if people is None:
        people = df["person"].unique().tolist()

    # Get top artists for each person
    result_dfs = []
    for person in people:
        person_df = df[df["person"] == person]
        top = person_df.groupby("artist_name").agg(
            total_hours=("hours_played", "sum"),
        ).reset_index()
        top = top.nlargest(limit, "total_hours")
        top = top.rename(columns={"total_hours": person})
        result_dfs.append(top.set_index("artist_name"))

    if not result_dfs:
        return pd.DataFrame()

    # Merge all
    combined = result_dfs[0]
    for other_df in result_dfs[1:]:
        combined = combined.join(other_df, how="outer")

    combined = combined.fillna(0)
    combined = combined.reset_index()

    return combined


def shared_artists(df: pd.DataFrame, person1: str, person2: str, min_hours: float = 0.5) -> pd.DataFrame:
    """
    Find artists that both people listen to.

    Args:
        df: Streaming history DataFrame with 'person' column.
        person1: First person's name.
        person2: Second person's name.
        min_hours: Minimum hours to count as "listening to" an artist.

    Returns:
        DataFrame with shared artists and hours for each person.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    # Get artists for person1
    p1_df = df[df["person"] == person1]
    p1_artists = p1_df.groupby("artist_name").agg(
        hours_p1=("hours_played", "sum"),
    ).reset_index()
    p1_artists = p1_artists[p1_artists["hours_p1"] >= min_hours]

    # Get artists for person2
    p2_df = df[df["person"] == person2]
    p2_artists = p2_df.groupby("artist_name").agg(
        hours_p2=("hours_played", "sum"),
    ).reset_index()
    p2_artists = p2_artists[p2_artists["hours_p2"] >= min_hours]

    # Find intersection
    shared = p1_artists.merge(p2_artists, on="artist_name", how="inner")
    shared = shared.rename(columns={
        "hours_p1": f"{person1}_hours",
        "hours_p2": f"{person2}_hours",
    })

    # Add combined hours and sort
    shared["combined_hours"] = shared[f"{person1}_hours"] + shared[f"{person2}_hours"]
    shared = shared.sort_values("combined_hours", ascending=False)

    return shared.reset_index(drop=True)


def unique_artists(df: pd.DataFrame, person: str, others: Optional[list[str]] = None, min_hours: float = 1.0) -> pd.DataFrame:
    """
    Find artists that only this person listens to.

    Args:
        df: Streaming history DataFrame with 'person' column.
        person: Person to analyze.
        others: List of other people to compare against (None for all others).
        min_hours: Minimum hours to count as "listening to" an artist.

    Returns:
        DataFrame with unique artists and hours.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    all_people = df["person"].unique().tolist()
    if others is None:
        others = [p for p in all_people if p != person]

    # Get artists for target person
    person_df = df[df["person"] == person]
    person_artists = person_df.groupby("artist_name").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()
    person_artists = person_artists[person_artists["total_hours"] >= min_hours]

    # Get artists that others listen to
    others_df = df[df["person"].isin(others)]
    others_artists = others_df.groupby("artist_name").agg(
        others_hours=("hours_played", "sum"),
    ).reset_index()
    others_artists = others_artists[others_artists["others_hours"] >= min_hours]

    # Find unique to person
    unique = person_artists[~person_artists["artist_name"].isin(others_artists["artist_name"])]
    unique = unique.sort_values("total_hours", ascending=False)

    return unique.reset_index(drop=True)


def taste_similarity_score(df: pd.DataFrame, person1: str, person2: str) -> dict:
    """
    Calculate taste similarity between two people (0-100).

    Uses Jaccard similarity on artist sets, weighted by hours.

    Args:
        df: Streaming history DataFrame with 'person' column.
        person1: First person's name.
        person2: Second person's name.

    Returns:
        Dictionary with similarity metrics.
    """
    if "person" not in df.columns:
        return {"similarity_score": None, "shared_artists": 0, "p1_artists": 0, "p2_artists": 0}

    # Get artists for each person with hours
    p1_df = df[df["person"] == person1]
    p1_artists = set(p1_df.groupby("artist_name").filter(
        lambda x: x["hours_played"].sum() >= 0.5
    )["artist_name"].unique())

    p2_df = df[df["person"] == person2]
    p2_artists = set(p2_df.groupby("artist_name").filter(
        lambda x: x["hours_played"].sum() >= 0.5
    )["artist_name"].unique())

    if not p1_artists or not p2_artists:
        return {"similarity_score": 0, "shared_artists": 0, "p1_artists": len(p1_artists), "p2_artists": len(p2_artists)}

    # Jaccard similarity
    intersection = len(p1_artists & p2_artists)
    union = len(p1_artists | p2_artists)
    jaccard = (intersection / union * 100) if union > 0 else 0

    return {
        "similarity_score": jaccard,
        "shared_artists": intersection,
        "p1_artists": len(p1_artists),
        "p2_artists": len(p2_artists),
    }


def genre_comparison(df: pd.DataFrame, people: Optional[list[str]] = None, limit: int = 10) -> pd.DataFrame:
    """
    Compare genre preferences across people.

    Args:
        df: Enriched streaming history DataFrame with 'person' and 'artist_genres' columns.
        people: List of people to compare (None for all).
        limit: Number of top genres to include.

    Returns:
        DataFrame with genre and percentage for each person.
    """
    if "person" not in df.columns or "artist_genres" not in df.columns:
        return pd.DataFrame()

    if people is None:
        people = df["person"].unique().tolist()

    # Collect genre stats for each person
    genre_stats = {}
    for person in people:
        person_df = df[df["person"] == person]

        # Explode genres
        exploded = person_df.explode("artist_genres")
        exploded = exploded[exploded["artist_genres"].notna()]
        exploded = exploded[exploded["artist_genres"] != ""]

        if exploded.empty:
            continue

        grouped = exploded.groupby("artist_genres").agg(
            total_hours=("hours_played", "sum"),
        ).reset_index()

        total = grouped["total_hours"].sum()
        grouped["percentage"] = grouped["total_hours"] / total * 100
        grouped = grouped.nlargest(limit, "total_hours")

        genre_stats[person] = grouped.set_index("artist_genres")["percentage"]

    if not genre_stats:
        return pd.DataFrame()

    # Combine into single DataFrame
    combined = pd.DataFrame(genre_stats)
    combined = combined.fillna(0)
    combined = combined.reset_index()
    combined = combined.rename(columns={"artist_genres": "genre"})

    return combined


def listening_volume_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare total listening volume across people.

    Args:
        df: Streaming history DataFrame with 'person' column.

    Returns:
        DataFrame with person, total_hours, play_count, unique_artists, unique_tracks.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("person").agg(
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
        unique_artists=("artist_name", "nunique"),
        unique_tracks=("track_name", "nunique"),
    ).reset_index()

    grouped = grouped.sort_values("total_hours", ascending=False)

    return grouped.reset_index(drop=True)


def time_pattern_comparison(df: pd.DataFrame) -> dict:
    """
    Compare when each person listens (hour of day patterns).

    Args:
        df: Streaming history DataFrame with 'person' and 'hour' columns.

    Returns:
        Dictionary with DataFrames for hour and day patterns by person.
    """
    if "person" not in df.columns or "hour" not in df.columns:
        return {"by_hour": pd.DataFrame(), "by_day": pd.DataFrame()}

    # By hour
    hour_grouped = df.groupby(["person", "hour"]).agg(
        total_hours=("hours_played", "sum"),
    ).reset_index()

    # Calculate percentage within each person
    person_totals = hour_grouped.groupby("person")["total_hours"].transform("sum")
    hour_grouped["percentage"] = hour_grouped["total_hours"] / person_totals * 100

    # Pivot for easier comparison
    hour_pivot = hour_grouped.pivot(index="hour", columns="person", values="percentage")
    hour_pivot = hour_pivot.fillna(0)

    # By day of week
    if "day_of_week" in df.columns:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_grouped = df.groupby(["person", "day_of_week"]).agg(
            total_hours=("hours_played", "sum"),
        ).reset_index()

        person_totals = day_grouped.groupby("person")["total_hours"].transform("sum")
        day_grouped["percentage"] = day_grouped["total_hours"] / person_totals * 100

        day_pivot = day_grouped.pivot(index="day_of_week", columns="person", values="percentage")
        day_pivot = day_pivot.fillna(0)
        day_pivot = day_pivot.reindex(day_order)
    else:
        day_pivot = pd.DataFrame()

    return {"by_hour": hour_pivot.reset_index(), "by_day": day_pivot.reset_index()}


def artist_overlap_matrix(df: pd.DataFrame, min_hours: float = 1.0) -> pd.DataFrame:
    """
    Create a matrix showing artist overlap between all person pairs.

    Args:
        df: Streaming history DataFrame with 'person' column.
        min_hours: Minimum hours to count as "listening to" an artist.

    Returns:
        DataFrame matrix with overlap counts.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    people = sorted(df["person"].unique().tolist())

    # Get artist sets for each person
    artist_sets = {}
    for person in people:
        person_df = df[df["person"] == person]
        artists = person_df.groupby("artist_name").agg(
            total_hours=("hours_played", "sum"),
        ).reset_index()
        artists = artists[artists["total_hours"] >= min_hours]
        artist_sets[person] = set(artists["artist_name"].tolist())

    # Create overlap matrix
    matrix_data = []
    for p1 in people:
        row = {"person": p1}
        for p2 in people:
            if p1 == p2:
                row[p2] = len(artist_sets[p1])  # Self = total artists
            else:
                overlap = len(artist_sets[p1] & artist_sets[p2])
                row[p2] = overlap
        matrix_data.append(row)

    return pd.DataFrame(matrix_data)


def first_listener_analysis(df: pd.DataFrame, artists: Optional[list[str]] = None, limit: int = 50) -> pd.DataFrame:
    """
    Determine who discovered each artist first.

    Args:
        df: Streaming history DataFrame with 'person' and 'timestamp' columns.
        artists: List of artists to analyze (None for top shared artists).
        limit: Number of artists to analyze if not specified.

    Returns:
        DataFrame with artist_name, first_listener, first_listen_date.
    """
    if "person" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame()

    people = df["person"].unique().tolist()

    # If no artists specified, get most listened artists that multiple people share
    if artists is None:
        # Find artists with >1 listener
        artist_listeners = df.groupby("artist_name")["person"].nunique()
        shared_artists = artist_listeners[artist_listeners > 1].index.tolist()

        # Get top by total hours among shared
        shared_df = df[df["artist_name"].isin(shared_artists)]
        top_shared = shared_df.groupby("artist_name")["hours_played"].sum()
        top_shared = top_shared.nlargest(limit).index.tolist()
        artists = top_shared

    results = []
    for artist in artists:
        artist_df = df[df["artist_name"] == artist]

        # Find first listen per person
        first_listens = artist_df.groupby("person")["timestamp"].min()

        if len(first_listens) == 0:
            continue

        first_person = first_listens.idxmin()
        first_date = first_listens.min()

        # Get all people who listen to this artist
        listeners = first_listens.index.tolist()

        results.append({
            "artist_name": artist,
            "first_listener": first_person,
            "first_listen_date": first_date,
            "total_listeners": len(listeners),
            "listeners": ", ".join(sorted(listeners)),
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("first_listen_date")

    return result_df.reset_index(drop=True)


def recommendation_overlaps(df: pd.DataFrame, source_person: str, target_person: str, limit: int = 20) -> pd.DataFrame:
    """
    Find artists that source_person listens to that target_person might like.

    Based on overlap with target's existing taste.

    Args:
        df: Streaming history DataFrame with 'person' column.
        source_person: Person to get recommendations from.
        target_person: Person to recommend to.
        limit: Number of recommendations.

    Returns:
        DataFrame with recommended artists and reasoning.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    # Get source's artists that target doesn't listen to
    source_df = df[df["person"] == source_person]
    target_df = df[df["person"] == target_person]

    source_artists = source_df.groupby("artist_name").agg(
        source_hours=("hours_played", "sum"),
    ).reset_index()
    source_artists = source_artists[source_artists["source_hours"] >= 1.0]

    target_artists = set(
        target_df.groupby("artist_name")
        .filter(lambda x: x["hours_played"].sum() >= 0.5)["artist_name"]
        .unique()
    )

    # Artists source likes that target doesn't have
    recs = source_artists[~source_artists["artist_name"].isin(target_artists)]
    recs = recs.nlargest(limit, "source_hours")

    if recs.empty:
        return pd.DataFrame()

    # Add context
    recs = recs.rename(columns={"source_hours": f"{source_person}_hours"})

    return recs.reset_index(drop=True)


def get_recommendations_from_person(
    df: pd.DataFrame,
    source_person: str,
    target_person: str,
    limit: int = 20
) -> pd.DataFrame:
    """
    Get artist recommendations for target_person from source_person's library.

    Enhanced version with genre info and context.

    Args:
        df: Streaming history DataFrame with 'person' column.
        source_person: Person to get recommendations from.
        target_person: Person to recommend to.
        limit: Number of recommendations.

    Returns:
        DataFrame with recommended artists and detailed context.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    source_df = df[df["person"] == source_person]
    target_df = df[df["person"] == target_person]

    if source_df.empty or target_df.empty:
        return pd.DataFrame()

    # Source's artists with stats
    agg_dict = {
        "source_hours": ("hours_played", "sum"),
        "source_plays": ("track_name", "count"),
        "first_listen": ("timestamp", "min"),
    }

    # Add genres if available
    if "artist_genres" in source_df.columns:
        agg_dict["genres"] = ("artist_genres", "first")

    source_artists = source_df.groupby("artist_name").agg(**agg_dict).reset_index()
    source_artists = source_artists[source_artists["source_hours"] >= 1.0]

    # Target's known artists (threshold of 0.25 hours = 15 minutes)
    target_known = set(
        target_df.groupby("artist_name")
        .filter(lambda x: x["hours_played"].sum() >= 0.25)["artist_name"]
        .unique()
    )

    # Filter to unknown artists
    recs = source_artists[~source_artists["artist_name"].isin(target_known)]
    recs = recs.nlargest(limit, "source_hours").copy()

    if recs.empty:
        return pd.DataFrame()

    # Add recommendation reason
    def make_reason(row):
        hours = row["source_hours"]
        first = row["first_listen"]
        first_str = first.strftime("%Y-%m-%d") if pd.notna(first) else "unknown"
        return f"{source_person} has {hours:.1f} hours since {first_str}"

    recs["recommendation_reason"] = recs.apply(make_reason, axis=1)

    # Format genres if available
    if "genres" in recs.columns:
        recs["genres"] = recs["genres"].apply(
            lambda x: ", ".join(x[:3]) if isinstance(x, list) and x else "Unknown"
        )
    else:
        recs["genres"] = "Unknown"

    output_cols = ["artist_name", "source_hours", "source_plays", "genres", "recommendation_reason"]
    return recs[output_cols].reset_index(drop=True)


def artist_journey(df: pd.DataFrame, artist_name: str) -> pd.DataFrame:
    """
    Show timeline of when each person discovered an artist.

    Args:
        df: Streaming history DataFrame with 'person' and 'timestamp' columns.
        artist_name: Artist to analyze.

    Returns:
        DataFrame with person, first_listen, total_hours, status.
    """
    if "person" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame()

    artist_df = df[df["artist_name"] == artist_name]
    if artist_df.empty:
        return pd.DataFrame()

    grouped = artist_df.groupby("person").agg(
        first_listen=("timestamp", "min"),
        last_listen=("timestamp", "max"),
        total_hours=("hours_played", "sum"),
        play_count=("track_name", "count"),
    ).reset_index()

    grouped = grouped.sort_values("first_listen")

    # Add discovery order
    grouped["discovery_order"] = range(1, len(grouped) + 1)

    return grouped.reset_index(drop=True)


def find_taste_twin(df: pd.DataFrame, target_person: str) -> pd.DataFrame:
    """
    Find the person most similar to target_person.

    Returns DataFrame sorted by similarity with detailed breakdown.

    Args:
        df: Streaming history DataFrame with 'person' column.
        target_person: Person to find taste twin for.

    Returns:
        DataFrame with similarity scores and details for each other person.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    people = [p for p in df["person"].unique() if p != target_person]

    if not people:
        return pd.DataFrame()

    results = []
    for person in people:
        sim = taste_similarity_score(df, target_person, person)
        shared = shared_artists(df, target_person, person, min_hours=0.5)

        top_shared_list = shared["artist_name"].head(3).tolist() if not shared.empty else []

        results.append({
            "person": person,
            "similarity_score": sim["similarity_score"],
            "shared_artists": sim["shared_artists"],
            "their_unique_artists": sim["p2_artists"] - sim["shared_artists"],
            "your_unique_artists": sim["p1_artists"] - sim["shared_artists"],
            "top_shared": ", ".join(top_shared_list) if top_shared_list else "None",
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("similarity_score", ascending=False)

    return result_df.reset_index(drop=True)


def taste_compatibility_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate full compatibility matrix for all people.

    Returns pivot table with similarity scores.

    Args:
        df: Streaming history DataFrame with 'person' column.

    Returns:
        DataFrame matrix with person pairs and similarity scores.
    """
    if "person" not in df.columns:
        return pd.DataFrame()

    people = sorted(df["person"].unique().tolist())

    if len(people) < 2:
        return pd.DataFrame()

    matrix_data = []
    for p1 in people:
        row = {"person": p1}
        for p2 in people:
            if p1 == p2:
                row[p2] = 100.0  # 100% match with yourself
            else:
                sim = taste_similarity_score(df, p1, p2)
                row[p2] = sim["similarity_score"] if sim["similarity_score"] is not None else 0
        matrix_data.append(row)

    return pd.DataFrame(matrix_data)


def artist_network_data(df: pd.DataFrame, people: list[str], min_listeners: int = 2, limit: int = 30) -> dict:
    """
    Generate network data for artist connections visualization.

    Creates nodes (artists + people) and edges (person-artist connections)
    for visualizing shared artists.

    Args:
        df: Streaming history DataFrame with 'person' column.
        people: List of people to include.
        min_listeners: Minimum number of people who must listen to an artist.
        limit: Maximum number of artists to include.

    Returns:
        dict with 'nodes' (DataFrame) and 'edges' (DataFrame) for network visualization.
    """
    if "person" not in df.columns:
        return {"nodes": pd.DataFrame(), "edges": pd.DataFrame()}

    # Filter to selected people
    filtered_df = df[df["person"].isin(people)]

    # Get artists listened to by multiple people
    artist_listeners = filtered_df.groupby("artist_name").agg(
        listeners=("person", "nunique"),
        total_hours=("hours_played", "sum"),
        people_list=("person", lambda x: list(x.unique()))
    ).reset_index()

    # Filter to shared artists
    shared = artist_listeners[artist_listeners["listeners"] >= min_listeners]
    shared = shared.nlargest(limit, "total_hours")

    if shared.empty:
        return {"nodes": pd.DataFrame(), "edges": pd.DataFrame()}

    # Build nodes: people + artists
    nodes = []

    # Add person nodes
    for person in people:
        person_hours = filtered_df[filtered_df["person"] == person]["hours_played"].sum()
        nodes.append({
            "id": person,
            "type": "person",
            "size": 30,
            "hours": person_hours,
            "label": person
        })

    # Add artist nodes
    for _, row in shared.iterrows():
        nodes.append({
            "id": row["artist_name"],
            "type": "artist",
            "size": min(20, 5 + row["total_hours"] / 10),
            "hours": row["total_hours"],
            "listeners": row["listeners"],
            "label": row["artist_name"][:15] + "..." if len(row["artist_name"]) > 15 else row["artist_name"]
        })

    # Build edges: person -> artist
    edges = []
    for _, row in shared.iterrows():
        for person in row["people_list"]:
            if person in people:
                person_hours = filtered_df[
                    (filtered_df["person"] == person) &
                    (filtered_df["artist_name"] == row["artist_name"])
                ]["hours_played"].sum()
                edges.append({
                    "source": person,
                    "target": row["artist_name"],
                    "weight": person_hours
                })

    return {
        "nodes": pd.DataFrame(nodes),
        "edges": pd.DataFrame(edges),
        "shared_artists": shared
    }


def label_comparison(df: pd.DataFrame, people: Optional[list[str]] = None, limit: int = 20) -> pd.DataFrame:
    """
    Compare listening by record label across people.

    Args:
        df: Enriched streaming history DataFrame with 'person' and 'label' columns.
        people: List of people to compare (None for all).
        limit: Number of top labels.

    Returns:
        DataFrame with label and hours for each person.
    """
    if "person" not in df.columns or "label" not in df.columns:
        return pd.DataFrame()

    if people is None:
        people = df["person"].unique().tolist()

    # Filter to valid labels
    df_with_label = df[df["label"].notna()]
    df_with_label = df_with_label[df_with_label["label"] != ""]

    if df_with_label.empty:
        return pd.DataFrame()

    # Get top labels overall
    top_labels = df_with_label.groupby("label")["hours_played"].sum()
    top_labels = top_labels.nlargest(limit).index.tolist()

    # Get hours per person per label
    result_dfs = []
    for person in people:
        person_df = df_with_label[df_with_label["person"] == person]
        person_df = person_df[person_df["label"].isin(top_labels)]

        grouped = person_df.groupby("label").agg(
            total_hours=("hours_played", "sum"),
        ).reset_index()
        grouped = grouped.rename(columns={"total_hours": person})
        result_dfs.append(grouped.set_index("label"))

    if not result_dfs:
        return pd.DataFrame()

    combined = result_dfs[0]
    for other_df in result_dfs[1:]:
        combined = combined.join(other_df, how="outer")

    combined = combined.fillna(0)
    combined = combined.reset_index()

    # Sort by total
    combined["_total"] = combined[people].sum(axis=1)
    combined = combined.sort_values("_total", ascending=False)
    combined = combined.drop(columns=["_total"])

    return combined.reset_index(drop=True)
