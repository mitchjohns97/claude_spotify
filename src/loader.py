"""Load and combine Spotify streaming history from JSON files."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def find_json_files(data_dir: Path) -> list[Path]:
    """Recursively find all JSON files in the data directory."""
    return list(data_dir.rglob("*.json"))


def get_people_folders(data_dir: Optional[Path] = None) -> list[str]:
    """
    Get list of people (folder names) in the data directory.

    Args:
        data_dir: Path to spotify_data folder. Defaults to project root.

    Returns:
        List of folder names representing people.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "spotify_data"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    # Return subfolders that contain JSON files
    people = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            json_files = list(item.glob("*.json"))
            if json_files:
                people.append(item.name)

    return people


def load_all_people(
    data_dir: Optional[Path] = None,
    enrich: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load streaming history from all people folders with person column.

    Args:
        data_dir: Path to spotify_data folder containing person subfolders.
        enrich: Whether to enrich with API cache data.
        cache_dir: Path to cache directory for enrichment.

    Returns:
        DataFrame with all streaming records and 'person' column.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "spotify_data"
    else:
        data_dir = Path(data_dir)

    people = get_people_folders(data_dir)
    if not people:
        raise FileNotFoundError(f"No people folders found in {data_dir}")

    all_dfs = []
    for person in people:
        person_dir = data_dir / person
        try:
            person_df = load_streaming_history(
                person_dir,
                add_person=False,  # We'll add it manually
                enrich=False,  # Enrich after combining
            )
            person_df["person"] = person
            all_dfs.append(person_df)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load data for {person}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No valid streaming data found for any person")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if enrich:
        from src.enrichment import enrich_streaming_data
        combined = enrich_streaming_data(combined, cache_dir=cache_dir)

    return combined


def load_streaming_history(
    data_dirs: Optional[list[Path] | Path] = None,
    add_person: bool = True,
    enrich: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load all Spotify streaming history JSON files and combine into a DataFrame.

    Args:
        data_dirs: Path(s) to spotify_data folder(s). Can be a single path or list of paths.
                   Defaults to spotify_data/ in project root.
        add_person: Whether to add 'person' column based on parent folder name.
        enrich: Whether to enrich with API cache data.
        cache_dir: Path to cache directory for enrichment.

    Returns:
        DataFrame with all streaming records.
    """
    # Normalize to list
    if data_dirs is None:
        data_dirs = [Path(__file__).parent.parent / "spotify_data"]
    elif isinstance(data_dirs, (str, Path)):
        data_dirs = [Path(data_dirs)]
    else:
        data_dirs = [Path(d) for d in data_dirs]

    # Collect JSON files from all paths, tracking source folder
    json_files_with_source = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"Warning: {data_dir} not found, skipping")
            continue
        found_files = find_json_files(data_dir)
        for f in found_files:
            # Determine person from folder structure
            # If file is directly in data_dir, use data_dir name
            # If file is in a subfolder, use subfolder name
            relative = f.relative_to(data_dir)
            if len(relative.parts) > 1:
                person = relative.parts[0]
            else:
                person = data_dir.name
            json_files_with_source.append((f, person))

    if not json_files_with_source:
        raise FileNotFoundError(f"No JSON files found in any of the specified directories: {data_dirs}")

    all_records = []
    for json_file, person in json_files_with_source:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                if isinstance(records, list):
                    if add_person:
                        for record in records:
                            record["_person"] = person
                    all_records.extend(records)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue

    if not all_records:
        raise ValueError("No valid streaming records found")

    df = pd.DataFrame(all_records)
    df = clean_dataframe(df)

    # Move _person to person column
    if add_person and "_person" in df.columns:
        df["person"] = df["_person"]
        df = df.drop(columns=["_person"])

    if enrich:
        from src.enrichment import enrich_streaming_data
        df = enrich_streaming_data(df, cache_dir=cache_dir)

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the streaming history DataFrame."""
    # Rename columns for easier access
    column_mapping = {
        "ts": "timestamp",
        "ms_played": "ms_played",
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name",
        "spotify_track_uri": "track_uri",
        "platform": "platform",
        "skipped": "skipped",
        "shuffle": "shuffle",
    }

    # Only rename columns that exist
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["date"] = df["timestamp"].dt.date
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day_of_week"] = df["timestamp"].dt.day_name()
        df["hour"] = df["timestamp"].dt.hour

    # Calculate minutes played
    if "ms_played" in df.columns:
        df["minutes_played"] = df["ms_played"] / 60000
        df["hours_played"] = df["ms_played"] / 3600000

    # Clean up null values in key columns
    if "track_name" in df.columns:
        df["track_name"] = df["track_name"].fillna("Unknown Track")
    if "artist_name" in df.columns:
        df["artist_name"] = df["artist_name"].fillna("Unknown Artist")
    if "album_name" in df.columns:
        df["album_name"] = df["album_name"].fillna("Unknown Album")

    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get a summary of the loaded data."""
    summary = {
        "total_streams": len(df),
        "total_hours": df["hours_played"].sum() if "hours_played" in df.columns else 0,
        "unique_tracks": df["track_name"].nunique() if "track_name" in df.columns else 0,
        "unique_artists": df["artist_name"].nunique() if "artist_name" in df.columns else 0,
        "unique_albums": df["album_name"].nunique() if "album_name" in df.columns else 0,
        "date_range": None,
    }

    if "timestamp" in df.columns and len(df) > 0:
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()
        summary["date_range"] = (min_date, max_date)

    return summary
