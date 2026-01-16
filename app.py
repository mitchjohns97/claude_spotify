#!/usr/bin/env python3
"""
Spotify Streaming History Analyzer - Streamlit App

Interactive web app for exploring Spotify listening history.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import date

from src.loader import load_streaming_history, get_data_summary
from src import analysis


st.set_page_config(
    page_title="Spotify Analyzer",
    page_icon="🎵",
    layout="wide",
)


def load_data(data_paths: list[str]) -> pd.DataFrame:
    """Load streaming history from specified paths."""
    paths = [Path(p.strip()) for p in data_paths if p.strip()]
    return load_streaming_history(paths if paths else None)


def format_hours(hours: float) -> str:
    """Format hours into readable string."""
    if hours >= 24:
        days = hours / 24
        return f"{hours:,.1f} hours ({days:,.1f} days)"
    return f"{hours:,.1f} hours"


# Sidebar - Data Source Selection
st.sidebar.title("Spotify Analyzer")
st.sidebar.markdown("---")

default_path = str(Path(__file__).parent / "spotify_data")
data_paths_text = st.sidebar.text_area(
    "Data folder paths (one per line):",
    value=default_path,
    height=100
)

if st.sidebar.button("Reload Data"):
    st.session_state.pop("df", None)
    st.session_state.pop("summary", None)

# Load data
if "df" not in st.session_state:
    try:
        with st.spinner("Loading data..."):
            data_paths = data_paths_text.strip().split("\n")
            st.session_state.df = load_data(data_paths)
            st.session_state.summary = get_data_summary(st.session_state.df)
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = st.session_state.df
summary = st.session_state.summary

# Display data summary in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Loaded Data")
st.sidebar.metric("Total Streams", f"{summary['total_streams']:,}")
st.sidebar.metric("Unique Artists", f"{summary['unique_artists']:,}")
if summary["date_range"]:
    start, end = summary["date_range"]
    st.sidebar.text(f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Artist Deep-Dive", "Song Deep-Dive", "Album Deep-Dive", "Time Range Analysis"]
)


def create_bar_chart(data: pd.Series, x_title: str = "Category", y_title: str = "Hours") -> alt.Chart:
    """Create a bar chart with y-axis starting at 0."""
    chart_df = data.reset_index()
    chart_df.columns = ["category", "value"]
    return alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("category:N", sort=None, title=x_title),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, chart_df["value"].max() * 1.1]), title=y_title)
    ).properties(height=400)


def create_line_chart(data: pd.Series, x_title: str = "Period", y_title: str = "Hours") -> alt.Chart:
    """Create a line chart with y-axis starting at 0."""
    chart_df = data.reset_index()
    chart_df.columns = ["period", "value"]
    return alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X("period:N", sort=None, title=x_title),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, chart_df["value"].max() * 1.1]), title=y_title)
    ).properties(height=400)


# Page: Overview
if page == "Overview":
    st.title("Spotify Listening Overview")

    # Summary stats
    stats = analysis.generate_summary_stats(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hours", f"{stats['total_hours']:,.0f}")
    with col2:
        st.metric("Total Streams", f"{stats['total_streams']:,}")
    with col3:
        st.metric("Unique Artists", f"{stats['unique_artists']:,}")
    with col4:
        st.metric("Unique Tracks", f"{stats['unique_tracks']:,}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Unique Albums", f"{stats['unique_albums']:,}")
    with col6:
        if "years_span" in stats:
            st.metric("Years of Data", f"{stats['years_span']:.1f}")
    with col7:
        st.metric("Listening Days", f"{stats.get('total_listening_days', 0):,}")
    with col8:
        st.metric("Longest Streak", f"{stats.get('longest_streak', 0)} days")

    st.markdown("---")

    # Top 10 Artists Chart
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Artists")
        top_artists = analysis.top_artists(df, limit=10)
        chart_data = top_artists.set_index("artist_name")["total_hours"]
        st.altair_chart(create_bar_chart(chart_data, "Artist", "Hours"), use_container_width=True)

        st.dataframe(
            top_artists[["artist_name", "total_hours", "play_count"]].rename(
                columns={"artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
            ),
            hide_index=True,
        )

    with col_right:
        st.subheader("Listening by Year")
        yearly = analysis.listening_by_year(df)
        chart_data = yearly.set_index("year")["total_hours"]
        st.altair_chart(create_bar_chart(chart_data, "Year", "Hours"), use_container_width=True)

        st.dataframe(
            yearly[["year", "total_hours", "play_count", "unique_artists"]].rename(
                columns={"year": "Year", "total_hours": "Hours", "play_count": "Plays", "unique_artists": "Artists"}
            ),
            hide_index=True,
        )


# Page: Artist Deep-Dive
elif page == "Artist Deep-Dive":
    st.title("Artist Deep-Dive")

    # Get all unique artists sorted by listening time
    all_artists = analysis.top_artists(df, limit=len(df))["artist_name"].tolist()

    selected_artist = st.selectbox(
        "Select an artist:",
        options=all_artists,
        index=all_artists.index("Bright Eyes") if "Bright Eyes" in all_artists else 0,
    )

    if selected_artist:
        artist_data = analysis.artist_deep_dive(df, selected_artist)

        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hours", f"{artist_data['total_hours']:.1f}")
        with col2:
            st.metric("Play Count", f"{artist_data['play_count']:,}")
        with col3:
            if artist_data["first_listen"]:
                st.metric("First Listen", artist_data["first_listen"].strftime("%Y-%m-%d"))
        with col4:
            if artist_data["last_listen"]:
                st.metric("Last Listen", artist_data["last_listen"].strftime("%Y-%m-%d"))

        st.markdown("---")

        # Top tracks and albums
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Top 10 Tracks")
            top_tracks = artist_data["top_tracks"]
            if not top_tracks.empty:
                chart_data = top_tracks.set_index("track_name")["total_hours"]
                st.altair_chart(create_bar_chart(chart_data, "Track", "Hours"), use_container_width=True)

                st.dataframe(
                    top_tracks[["track_name", "total_hours", "play_count"]].rename(
                        columns={"track_name": "Track", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

        with col_right:
            st.subheader("Top 5 Albums")
            top_albums = artist_data["top_albums"]
            if not top_albums.empty:
                chart_data = top_albums.set_index("album_name")["total_hours"]
                st.altair_chart(create_bar_chart(chart_data, "Album", "Hours"), use_container_width=True)

                st.dataframe(
                    top_albums[["album_name", "total_hours", "play_count"]].rename(
                        columns={"album_name": "Album", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

        st.markdown("---")

        # Listening timeline for artist
        st.subheader("Listening Over Time")
        timeline = analysis.artist_timeline(df, selected_artist)
        if not timeline.empty:
            chart_data = timeline.set_index("year_month")["total_hours"]
            st.altair_chart(create_line_chart(chart_data, "Month", "Hours"), use_container_width=True)


# Page: Song Deep-Dive
elif page == "Song Deep-Dive":
    st.title("Song Deep-Dive")

    # Get all unique tracks sorted by listening time
    all_tracks = analysis.top_tracks(df, limit=len(df))

    # Create "Track Name - Artist" labels for dropdown
    track_options = [f"{row['track_name']} - {row['artist_name']}" for _, row in all_tracks.iterrows()]
    track_mapping = {f"{row['track_name']} - {row['artist_name']}": (row['track_name'], row['artist_name'])
                     for _, row in all_tracks.iterrows()}

    selected_track_label = st.selectbox(
        "Select a song:",
        options=track_options,
        index=0,
    )

    if selected_track_label:
        track_name, artist_name = track_mapping[selected_track_label]
        song_data = analysis.song_deep_dive(df, track_name, artist_name)

        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hours", f"{song_data['total_hours']:.2f}")
        with col2:
            st.metric("Play Count", f"{song_data['play_count']:,}")
        with col3:
            if song_data["first_listen"]:
                st.metric("First Listen", song_data["first_listen"].strftime("%Y-%m-%d"))
        with col4:
            if song_data["last_listen"]:
                st.metric("Last Listen", song_data["last_listen"].strftime("%Y-%m-%d"))

        st.markdown("---")

        # Albums containing this track
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Albums Containing This Track")
            albums = song_data["albums"]
            if not albums.empty:
                chart_data = albums.set_index("album_name")["total_hours"]
                st.altair_chart(create_bar_chart(chart_data, "Album", "Hours"), use_container_width=True)

                st.dataframe(
                    albums[["album_name", "total_hours", "play_count"]].rename(
                        columns={"album_name": "Album", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

        with col_right:
            st.subheader("Artist Info")
            st.write(f"**Artist:** {artist_name}")
            # Get artist rank
            all_artists = analysis.top_artists(df, limit=len(df))
            artist_rank = all_artists[all_artists["artist_name"] == artist_name].index
            if len(artist_rank) > 0:
                st.write(f"**Artist Rank:** #{artist_rank[0] + 1}")

            # Get track rank
            track_rank = all_tracks[(all_tracks["track_name"] == track_name) &
                                     (all_tracks["artist_name"] == artist_name)].index
            if len(track_rank) > 0:
                st.write(f"**Track Rank:** #{track_rank[0] + 1}")

        st.markdown("---")

        # Listening timeline for song
        st.subheader("Listening Over Time")
        timeline = analysis.song_timeline(df, track_name, artist_name)
        if not timeline.empty:
            chart_data = timeline.set_index("year_month")["total_hours"]
            st.altair_chart(create_line_chart(chart_data, "Month", "Hours"), use_container_width=True)


# Page: Album Deep-Dive
elif page == "Album Deep-Dive":
    st.title("Album Deep-Dive")

    # Get all unique albums sorted by listening time
    all_albums = analysis.top_albums(df, limit=len(df))

    # Create "Album Name - Artist" labels for dropdown
    album_options = [f"{row['album_name']} - {row['artist_name']}" for _, row in all_albums.iterrows()]
    album_mapping = {f"{row['album_name']} - {row['artist_name']}": (row['album_name'], row['artist_name'])
                     for _, row in all_albums.iterrows()}

    selected_album_label = st.selectbox(
        "Select an album:",
        options=album_options,
        index=0,
    )

    if selected_album_label:
        album_name, artist_name = album_mapping[selected_album_label]
        album_data = analysis.album_deep_dive(df, album_name, artist_name)

        # Basic stats
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Hours", f"{album_data['total_hours']:.2f}")
        with col2:
            st.metric("Play Count", f"{album_data['play_count']:,}")
        with col3:
            st.metric("Unique Tracks", f"{album_data['unique_tracks']}")
        with col4:
            if album_data["first_listen"]:
                st.metric("First Listen", album_data["first_listen"].strftime("%Y-%m-%d"))
        with col5:
            if album_data["last_listen"]:
                st.metric("Last Listen", album_data["last_listen"].strftime("%Y-%m-%d"))

        st.markdown("---")

        # Top tracks and album info
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Top Tracks on This Album")
            top_tracks = album_data["top_tracks"]
            if not top_tracks.empty:
                chart_data = top_tracks.set_index("track_name")["total_hours"]
                st.altair_chart(create_bar_chart(chart_data, "Track", "Hours"), use_container_width=True)

                st.dataframe(
                    top_tracks[["track_name", "total_hours", "play_count"]].rename(
                        columns={"track_name": "Track", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

        with col_right:
            st.subheader("Album Info")
            st.write(f"**Artist:** {artist_name}")
            # Get artist rank
            all_artists = analysis.top_artists(df, limit=len(df))
            artist_rank = all_artists[all_artists["artist_name"] == artist_name].index
            if len(artist_rank) > 0:
                st.write(f"**Artist Rank:** #{artist_rank[0] + 1}")

            # Get album rank
            album_rank = all_albums[(all_albums["album_name"] == album_name) &
                                     (all_albums["artist_name"] == artist_name)].index
            if len(album_rank) > 0:
                st.write(f"**Album Rank:** #{album_rank[0] + 1}")

        st.markdown("---")

        # Listening timeline for album
        st.subheader("Listening Over Time")
        timeline = analysis.album_timeline(df, album_name, artist_name)
        if not timeline.empty:
            chart_data = timeline.set_index("year_month")["total_hours"]
            st.altair_chart(create_line_chart(chart_data, "Month", "Hours"), use_container_width=True)


# Page: Time Range Analysis
elif page == "Time Range Analysis":
    st.title("Time Range Analysis")

    # Date range picker
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date:", value=date(2020, 1, 1), min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End date:", value=date(2021, 12, 31), min_value=min_date, max_value=max_date)

    # Filter data
    filtered_df = analysis.filter_by_date_range(df, start_date, end_date)

    if filtered_df.empty:
        st.warning("No data found for the selected date range.")
        st.stop()

    # Stats for filtered range
    filtered_stats = analysis.generate_summary_stats(filtered_df)
    overall_stats = analysis.generate_summary_stats(df)

    st.markdown("---")
    st.subheader("Selected Range Stats")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Hours",
            f"{filtered_stats['total_hours']:,.0f}",
            delta=f"{filtered_stats['total_hours'] / overall_stats['total_hours'] * 100:.1f}% of total",
        )
    with col2:
        st.metric("Streams", f"{filtered_stats['total_streams']:,}")
    with col3:
        st.metric("Unique Artists", f"{filtered_stats['unique_artists']:,}")
    with col4:
        st.metric("Unique Tracks", f"{filtered_stats['unique_tracks']:,}")

    st.markdown("---")

    # Top 10 for filtered range
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.subheader("Top 10 Artists")
        top_artists = analysis.top_artists(filtered_df, limit=10)
        chart_data = top_artists.set_index("artist_name")["total_hours"]
        st.altair_chart(create_bar_chart(chart_data, "Artist", "Hours"), use_container_width=True)
        st.dataframe(
            top_artists[["artist_name", "total_hours", "play_count"]].rename(
                columns={"artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
            ),
            hide_index=True,
        )

    with col_mid:
        st.subheader("Top 10 Tracks")
        top_tracks = analysis.top_tracks(filtered_df, limit=10)
        st.dataframe(
            top_tracks[["track_name", "artist_name", "total_hours", "play_count"]].rename(
                columns={"track_name": "Track", "artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
            ),
            hide_index=True,
        )

    with col_right:
        st.subheader("Top 10 Albums")
        top_albums = analysis.top_albums(filtered_df, limit=10)
        st.dataframe(
            top_albums[["album_name", "artist_name", "total_hours"]].rename(
                columns={"album_name": "Album", "artist_name": "Artist", "total_hours": "Hours"}
            ),
            hide_index=True,
        )

    # Comparison to overall
    st.markdown("---")
    st.subheader("Comparison to Overall")

    comparison_data = {
        "Metric": ["Hours", "Streams", "Artists", "Tracks", "Albums"],
        "Selected Range": [
            f"{filtered_stats['total_hours']:,.0f}",
            f"{filtered_stats['total_streams']:,}",
            f"{filtered_stats['unique_artists']:,}",
            f"{filtered_stats['unique_tracks']:,}",
            f"{filtered_stats['unique_albums']:,}",
        ],
        "Overall": [
            f"{overall_stats['total_hours']:,.0f}",
            f"{overall_stats['total_streams']:,}",
            f"{overall_stats['unique_artists']:,}",
            f"{overall_stats['unique_tracks']:,}",
            f"{overall_stats['unique_albums']:,}",
        ],
        "% of Total": [
            f"{filtered_stats['total_hours'] / overall_stats['total_hours'] * 100:.1f}%",
            f"{filtered_stats['total_streams'] / overall_stats['total_streams'] * 100:.1f}%",
            f"{filtered_stats['unique_artists'] / overall_stats['unique_artists'] * 100:.1f}%",
            f"{filtered_stats['unique_tracks'] / overall_stats['unique_tracks'] * 100:.1f}%",
            f"{filtered_stats['unique_albums'] / overall_stats['unique_albums'] * 100:.1f}%",
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
