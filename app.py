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

from src.loader import load_streaming_history, get_data_summary, get_people_folders
from src import analysis
from src import comparison
from src.enrichment import enrich_streaming_data, get_enrichment_stats


st.set_page_config(
    page_title="Spotify Analyzer",
    page_icon="🎵",
    layout="wide",
)


def load_data(data_paths: list[str], enrich: bool = True) -> pd.DataFrame:
    """Load streaming history from specified paths."""
    paths = [Path(p.strip()) for p in data_paths if p.strip()]
    df = load_streaming_history(paths if paths else None, add_person=True, enrich=False)

    if enrich:
        df = enrich_streaming_data(df)

    return df


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
        with st.spinner("Loading and enriching data..."):
            data_paths = data_paths_text.strip().split("\n")
            st.session_state.df = load_data(data_paths, enrich=True)
            st.session_state.summary = get_data_summary(st.session_state.df)
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = st.session_state.df
summary = st.session_state.summary

# Global Person Selector
st.sidebar.markdown("---")
st.sidebar.subheader("Filter by Person")

if "person" in df.columns:
    all_people = sorted(df["person"].unique().tolist())
    selected_people = st.sidebar.multiselect(
        "Select people:",
        options=all_people,
        default=all_people,
        help="Filter all pages to selected people"
    )

    # Filter DataFrame
    if selected_people:
        filtered_df = df[df["person"].isin(selected_people)]
    else:
        filtered_df = df
        selected_people = all_people
else:
    filtered_df = df
    selected_people = []
    all_people = []

# Display data summary in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Loaded Data")
st.sidebar.metric("Total Streams", f"{len(filtered_df):,}")
st.sidebar.metric("Unique Artists", f"{filtered_df['artist_name'].nunique():,}")
if summary["date_range"]:
    start, end = summary["date_range"]
    st.sidebar.text(f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

# Show enrichment stats
if "track_popularity" in filtered_df.columns:
    enrich_stats = get_enrichment_stats(filtered_df)
    if enrich_stats["with_popularity_pct"]:
        st.sidebar.text(f"Enriched: {enrich_stats['with_popularity_pct']:.0f}%")

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Genre Analysis",
        "Popularity Insights",
        "Compare Listeners",
        "Music Trends",
        "Spotify Wrapped",
        "Artist Deep-Dive",
        "Song Deep-Dive",
        "Album Deep-Dive",
        "Time Range Analysis",
    ]
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

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Summary stats
    stats = analysis.generate_summary_stats(filtered_df)

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
        top_artists = analysis.top_artists(filtered_df, limit=10)
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
        yearly = analysis.listening_by_year(filtered_df)
        chart_data = yearly.set_index("year")["total_hours"]
        st.altair_chart(create_bar_chart(chart_data, "Year", "Hours"), use_container_width=True)

        st.dataframe(
            yearly[["year", "total_hours", "play_count", "unique_artists"]].rename(
                columns={"year": "Year", "total_hours": "Hours", "play_count": "Plays", "unique_artists": "Artists"}
            ),
            hide_index=True,
        )


# Page: Genre Analysis
elif page == "Genre Analysis":
    st.title("Genre Analysis")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Check for genre data
    if "artist_genres" not in filtered_df.columns:
        st.warning("Genre data not available. Make sure the cache files are present.")
        st.stop()

    # Top Genres
    st.subheader("Top 20 Genres by Listening Time")
    top_genres_df = analysis.top_genres(filtered_df, limit=20)

    if top_genres_df.empty:
        st.warning("No genre data found for the selected people.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            chart_data = top_genres_df.set_index("genre")["total_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Genre", "Hours"), use_container_width=True)

        with col_right:
            st.dataframe(
                top_genres_df[["genre", "total_hours", "play_count", "unique_artists"]].rename(
                    columns={"genre": "Genre", "total_hours": "Hours", "play_count": "Plays", "unique_artists": "Artists"}
                ),
                hide_index=True,
                height=400,
            )

    st.markdown("---")

    # Genre Trends Over Time
    st.subheader("Genre Trends Over Time")
    genre_trends = analysis.genre_by_year(filtered_df, limit=8)

    if not genre_trends.empty:
        trend_chart = alt.Chart(genre_trends).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("total_hours:Q", title="Hours"),
            color=alt.Color("genre:N", title="Genre"),
            tooltip=["year", "genre", "total_hours"]
        ).properties(height=400)
        st.altair_chart(trend_chart, use_container_width=True)
    else:
        st.info("Not enough data to show genre trends.")

    st.markdown("---")

    # Discovery Score & Decade Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Discovery Score")
        discovery = analysis.discovery_score(filtered_df)
        if discovery["discovery_score"] is not None:
            st.metric("Discovery Score", f"{discovery['discovery_score']:.1f}%")
            st.caption("Percentage of listening time spent on tracks with popularity < 30")
            st.write(f"Discovery hours: {discovery['discovery_hours']:.1f}")
            st.write(f"Total analyzed hours: {discovery['total_hours_with_data']:.1f}")
        else:
            st.info("Popularity data not available")

    with col2:
        st.subheader("Listening by Decade")
        decade_df = analysis.decade_analysis(filtered_df)
        if not decade_df.empty:
            chart_data = decade_df.set_index("decade")["total_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Decade", "Hours"), use_container_width=True)
        else:
            st.info("Release date data not available")

    st.markdown("---")

    # Genre Deep Dive
    st.subheader("Genre Deep Dive")
    if not top_genres_df.empty:
        selected_genre = st.selectbox(
            "Select a genre to explore:",
            options=top_genres_df["genre"].tolist(),
            index=0
        )

        genre_data = analysis.genre_deep_dive(filtered_df, selected_genre)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hours", f"{genre_data['total_hours']:.1f}")
        with col2:
            st.metric("Play Count", f"{genre_data['play_count']:,}")
        with col3:
            st.metric("Unique Artists", f"{genre_data['unique_artists']:,}")
        with col4:
            st.metric("Unique Tracks", f"{genre_data['unique_tracks']:,}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.write("**Top Artists in this Genre**")
            if not genre_data["top_artists"].empty:
                st.dataframe(
                    genre_data["top_artists"][["artist_name", "total_hours", "play_count"]].rename(
                        columns={"artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

        with col_right:
            st.write("**Top Tracks in this Genre**")
            if not genre_data["top_tracks"].empty:
                st.dataframe(
                    genre_data["top_tracks"][["track_name", "artist_name", "total_hours"]].rename(
                        columns={"track_name": "Track", "artist_name": "Artist", "total_hours": "Hours"}
                    ),
                    hide_index=True,
                )


# Page: Popularity Insights
elif page == "Popularity Insights":
    st.title("Popularity Insights")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Check for popularity data
    if "track_popularity" not in filtered_df.columns:
        st.warning("Popularity data not available. Make sure the cache files are present.")
        st.stop()

    # Summary metrics
    hipster = analysis.hipster_score(filtered_df)
    mainstream = analysis.mainstream_vs_indie(filtered_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if hipster["hipster_score"] is not None:
            st.metric("Hipster Score", f"{hipster['hipster_score']:.1f}")
            st.caption("100 = all obscure, 0 = all mainstream")
    with col2:
        if hipster["avg_popularity"] is not None:
            st.metric("Avg Track Popularity", f"{hipster['avg_popularity']:.1f}")
    with col3:
        if mainstream["mainstream_pct"] is not None:
            st.metric("Mainstream %", f"{mainstream['mainstream_pct']:.1f}%")
            st.caption("Tracks with popularity >= 50")
    with col4:
        if mainstream["indie_pct"] is not None:
            st.metric("Indie %", f"{mainstream['indie_pct']:.1f}%")
            st.caption("Tracks with popularity < 50")

    st.markdown("---")

    # Popularity Distribution
    st.subheader("Popularity Distribution")
    pop_dist = analysis.popularity_distribution(filtered_df)

    if not pop_dist.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            chart_data = pop_dist.set_index("popularity_bucket")["total_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Popularity Range", "Hours"), use_container_width=True)

        with col_right:
            chart_data = pop_dist.set_index("popularity_bucket")["count"]
            st.altair_chart(create_bar_chart(chart_data, "Popularity Range", "Plays"), use_container_width=True)

    st.markdown("---")

    # Most Obscure Favorites (Hidden Gems)
    st.subheader("Hidden Gems (Low Popularity, High Plays)")
    st.caption("Tracks you love that aren't popular on Spotify")

    min_plays = st.slider("Minimum plays to include:", min_value=3, max_value=50, value=10)
    obscure = analysis.most_obscure_favorites(filtered_df, min_plays=min_plays, limit=20)

    if not obscure.empty:
        st.dataframe(
            obscure[["track_name", "artist_name", "track_popularity", "play_count", "total_hours"]].rename(
                columns={
                    "track_name": "Track",
                    "artist_name": "Artist",
                    "track_popularity": "Popularity",
                    "play_count": "Plays",
                    "total_hours": "Hours"
                }
            ),
            hide_index=True,
        )
    else:
        st.info("No tracks found matching criteria")

    st.markdown("---")

    # Explicit Content Stats
    st.subheader("Explicit Content")
    explicit = analysis.explicit_content_stats(filtered_df)

    if explicit["explicit_pct"] is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Explicit Hours", f"{explicit['explicit_hours']:.1f}")
        with col2:
            st.metric("Clean Hours", f"{explicit['clean_hours']:.1f}")
        with col3:
            st.metric("Explicit %", f"{explicit['explicit_pct']:.1f}%")
    else:
        st.info("Explicit content data not available")


# Page: Compare Listeners
elif page == "Compare Listeners":
    st.title("Compare Listeners")

    if "person" not in df.columns or len(all_people) < 2:
        st.warning("Need data from multiple people to compare. Make sure each person has their own subfolder.")
        st.stop()

    # Person selector for comparison
    st.subheader("Select People to Compare")
    compare_people = st.multiselect(
        "Choose people:",
        options=all_people,
        default=all_people[:min(3, len(all_people))],
    )

    if len(compare_people) < 2:
        st.warning("Select at least 2 people to compare")
        st.stop()

    compare_df = df[df["person"].isin(compare_people)]

    # Volume Comparison
    st.markdown("---")
    st.subheader("Listening Volume Comparison")
    volume = comparison.listening_volume_comparison(compare_df)

    if not volume.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            chart_data = volume.set_index("person")["total_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Person", "Hours"), use_container_width=True)

        with col_right:
            st.dataframe(
                volume.rename(columns={
                    "person": "Person",
                    "total_hours": "Hours",
                    "play_count": "Plays",
                    "unique_artists": "Artists",
                    "unique_tracks": "Tracks"
                }),
                hide_index=True,
            )

    # Taste Similarity Matrix
    st.markdown("---")
    st.subheader("Taste Similarity")

    # Calculate pairwise similarity
    similarity_data = []
    for i, p1 in enumerate(compare_people):
        for p2 in compare_people[i+1:]:
            sim = comparison.taste_similarity_score(compare_df, p1, p2)
            similarity_data.append({
                "Pair": f"{p1} & {p2}",
                "Similarity": f"{sim['similarity_score']:.1f}%" if sim['similarity_score'] else "N/A",
                "Shared Artists": sim["shared_artists"],
            })

    if similarity_data:
        st.dataframe(pd.DataFrame(similarity_data), hide_index=True)

    # Artist Overlap Matrix
    st.markdown("---")
    st.subheader("Artist Overlap Matrix")
    overlap_matrix = comparison.artist_overlap_matrix(compare_df)

    if not overlap_matrix.empty:
        st.dataframe(overlap_matrix, hide_index=True)
        st.caption("Diagonal shows total artists; off-diagonal shows shared artists between pairs")

    # Genre Comparison
    if "artist_genres" in compare_df.columns:
        st.markdown("---")
        st.subheader("Genre Preferences Comparison")
        genre_comp = comparison.genre_comparison(compare_df, compare_people, limit=10)

        if not genre_comp.empty:
            # Create grouped bar chart
            genre_melted = genre_comp.melt(id_vars=["genre"], var_name="person", value_name="percentage")
            chart = alt.Chart(genre_melted).mark_bar().encode(
                x=alt.X("genre:N", title="Genre"),
                y=alt.Y("percentage:Q", title="% of Listening Time"),
                color=alt.Color("person:N", title="Person"),
                xOffset="person:N"
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

    # Artist Network Graph
    st.markdown("---")
    st.subheader("Artist Network Graph")
    st.caption("Visualize shared artists between listeners")

    if len(compare_people) >= 2:
        import math

        network_limit = st.slider("Number of artists to show:", min_value=5, max_value=50, value=20, key="network_limit")
        network_data = comparison.artist_network_data(compare_df, compare_people, min_listeners=2, limit=network_limit)

        if not network_data["nodes"].empty and not network_data["edges"].empty:
            nodes_df = network_data["nodes"]
            edges_df = network_data["edges"]

            # Create circular layout for network visualization
            n_people = len([n for n in nodes_df.itertuples() if n.type == "person"])
            n_artists = len([n for n in nodes_df.itertuples() if n.type == "artist"])

            # Position people in inner circle, artists in outer circle
            positions = {}
            person_idx = 0
            artist_idx = 0

            for _, node in nodes_df.iterrows():
                if node["type"] == "person":
                    angle = 2 * math.pi * person_idx / n_people
                    positions[node["id"]] = {
                        "x": 150 * math.cos(angle),
                        "y": 150 * math.sin(angle)
                    }
                    person_idx += 1
                else:
                    angle = 2 * math.pi * artist_idx / n_artists
                    positions[node["id"]] = {
                        "x": 350 * math.cos(angle),
                        "y": 350 * math.sin(angle)
                    }
                    artist_idx += 1

            # Add positions to nodes
            nodes_df = nodes_df.copy()
            nodes_df["x"] = nodes_df["id"].map(lambda x: positions[x]["x"])
            nodes_df["y"] = nodes_df["id"].map(lambda x: positions[x]["y"])

            # Add positions to edges
            edges_df = edges_df.copy()
            edges_df["x"] = edges_df["source"].map(lambda x: positions[x]["x"])
            edges_df["y"] = edges_df["source"].map(lambda x: positions[x]["y"])
            edges_df["x2"] = edges_df["target"].map(lambda x: positions[x]["x"])
            edges_df["y2"] = edges_df["target"].map(lambda x: positions[x]["y"])

            # Create edge layer (lines)
            edge_chart = alt.Chart(edges_df).mark_rule(opacity=0.3).encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                x2="x2:Q",
                y2="y2:Q",
                strokeWidth=alt.StrokeWidth("weight:Q", scale=alt.Scale(range=[0.5, 3]), legend=None),
                color=alt.value("#888888")
            )

            # Create person node layer
            person_nodes = nodes_df[nodes_df["type"] == "person"]
            person_chart = alt.Chart(person_nodes).mark_circle(size=400).encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                color=alt.value("#1DB954"),  # Spotify green
                tooltip=["label", alt.Tooltip("hours:Q", format=".1f", title="Total Hours")]
            )

            person_labels = alt.Chart(person_nodes).mark_text(fontSize=12, fontWeight="bold").encode(
                x="x:Q",
                y=alt.Y("y:Q"),
                text="label:N",
                color=alt.value("white")
            )

            # Create artist node layer
            artist_nodes = nodes_df[nodes_df["type"] == "artist"]
            artist_chart = alt.Chart(artist_nodes).mark_circle().encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                size=alt.Size("hours:Q", scale=alt.Scale(range=[50, 300]), legend=None),
                color=alt.Color("listeners:O", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Listeners")),
                tooltip=["label", "listeners", alt.Tooltip("hours:Q", format=".1f", title="Total Hours")]
            )

            # Combine layers
            network_chart = (edge_chart + artist_chart + person_chart + person_labels).properties(
                width=700,
                height=700,
                title="Shared Artist Network"
            ).configure_view(strokeWidth=0)

            st.altair_chart(network_chart, use_container_width=True)

            # Show shared artists table
            with st.expander("View Shared Artists Details"):
                shared_artists = network_data["shared_artists"]
                display_shared = shared_artists[["artist_name", "listeners", "total_hours"]].rename(columns={
                    "artist_name": "Artist",
                    "listeners": "Listeners",
                    "total_hours": "Total Hours"
                })
                display_shared["Total Hours"] = display_shared["Total Hours"].apply(lambda x: f"{x:.1f}")
                st.dataframe(display_shared, hide_index=True)
        else:
            st.info("No shared artists found between the selected people")
    else:
        st.info("Select at least 2 people to see the artist network")

    # Shared vs Unique Artists
    st.markdown("---")
    st.subheader("Shared Artists Analysis")

    if len(compare_people) == 2:
        p1, p2 = compare_people
        shared = comparison.shared_artists(compare_df, p1, p2)

        if not shared.empty:
            st.write(f"**Artists both {p1} and {p2} listen to:**")
            st.dataframe(
                shared.head(20).rename(columns={
                    "artist_name": "Artist",
                    f"{p1}_hours": f"{p1} Hours",
                    f"{p2}_hours": f"{p2} Hours",
                    "combined_hours": "Combined Hours"
                }),
                hide_index=True,
            )

        # Unique to each
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Unique to {p1}:**")
            unique_p1 = comparison.unique_artists(compare_df, p1, [p2])
            if not unique_p1.empty:
                st.dataframe(
                    unique_p1.head(10)[["artist_name", "total_hours", "play_count"]].rename(
                        columns={"artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

        with col2:
            st.write(f"**Unique to {p2}:**")
            unique_p2 = comparison.unique_artists(compare_df, p2, [p1])
            if not unique_p2.empty:
                st.dataframe(
                    unique_p2.head(10)[["artist_name", "total_hours", "play_count"]].rename(
                        columns={"artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
                    ),
                    hide_index=True,
                )

    # First Listener Analysis
    st.markdown("---")
    st.subheader("Who Discovered Who First?")
    first_listeners = comparison.first_listener_analysis(compare_df, limit=30)

    if not first_listeners.empty:
        st.dataframe(
            first_listeners[["artist_name", "first_listener", "first_listen_date", "total_listeners"]].rename(
                columns={
                    "artist_name": "Artist",
                    "first_listener": "First Listener",
                    "first_listen_date": "First Listen",
                    "total_listeners": "Total Listeners"
                }
            ),
            hide_index=True,
        )

    # Music Matchmaker (Taste Twin Finder)
    st.markdown("---")
    st.subheader("Music Matchmaker")
    st.caption("Find your taste twin - the person with most similar music taste")

    matchmaker_person = st.selectbox(
        "Find taste twin for:",
        options=compare_people,
        key="matchmaker_person"
    )

    if matchmaker_person:
        taste_twins = comparison.find_taste_twin(compare_df, matchmaker_person)

        if not taste_twins.empty:
            # Show top match prominently
            top_match = taste_twins.iloc[0]
            st.success(f"**{matchmaker_person}'s taste twin is {top_match['person']}!** (Similarity: {top_match['similarity_score']:.1f}%)")
            st.write(f"Top shared artists: {top_match['top_shared']}")

            st.write("**All Matches:**")
            display_df = taste_twins[["person", "similarity_score", "shared_artists", "their_unique_artists", "top_shared"]].rename(
                columns={
                    "person": "Person",
                    "similarity_score": "Similarity %",
                    "shared_artists": "Shared Artists",
                    "their_unique_artists": "Their Unique",
                    "top_shared": "Top Shared Artists"
                }
            )
            display_df["Similarity %"] = display_df["Similarity %"].apply(lambda x: f"{x:.1f}%" if x else "N/A")
            st.dataframe(display_df, hide_index=True)
        else:
            st.info("Not enough data to calculate taste matches")

    # Taste Compatibility Matrix
    st.markdown("---")
    st.subheader("Taste Compatibility Matrix")
    st.caption("How compatible is everyone's music taste?")

    compat_matrix = comparison.taste_compatibility_matrix(compare_df)

    if not compat_matrix.empty:
        # Create heatmap
        matrix_long = compat_matrix.melt(id_vars=["person"], var_name="person2", value_name="similarity")
        matrix_long = matrix_long[matrix_long["person"] != matrix_long["person2"]]

        heatmap = alt.Chart(matrix_long).mark_rect().encode(
            x=alt.X("person:N", title="Person"),
            y=alt.Y("person2:N", title="Person"),
            color=alt.Color("similarity:Q", scale=alt.Scale(scheme="greens", domain=[0, 100]), title="Similarity %"),
            tooltip=["person", "person2", alt.Tooltip("similarity:Q", format=".1f")]
        ).properties(width=400, height=300)

        st.altair_chart(heatmap, use_container_width=True)

        # Show as table too
        st.write("**Similarity Scores:**")
        display_matrix = compat_matrix.copy()
        display_matrix = display_matrix.set_index("person")
        # Format numbers
        display_matrix = display_matrix.applymap(lambda x: f"{x:.1f}%")
        st.dataframe(display_matrix)

    # "Introduce Me To..." Recommendations
    st.markdown("---")
    st.subheader("Introduce Me To...")
    st.caption("Get personalized artist recommendations from another person's library")

    col1, col2 = st.columns(2)
    with col1:
        rec_source = st.selectbox(
            "Get recommendations from:",
            options=compare_people,
            key="rec_source"
        )
    with col2:
        rec_target_options = [p for p in compare_people if p != rec_source]
        if rec_target_options:
            rec_target = st.selectbox(
                "Recommend to:",
                options=rec_target_options,
                key="rec_target"
            )
        else:
            rec_target = None
            st.info("Select a different source person")

    if rec_source and rec_target:
        recommendations = comparison.get_recommendations_from_person(
            compare_df, rec_source, rec_target, limit=20
        )

        if not recommendations.empty:
            st.write(f"**Artists {rec_source} loves that {rec_target} might enjoy:**")

            display_recs = recommendations.rename(columns={
                "artist_name": "Artist",
                "source_hours": "Hours",
                "source_plays": "Plays",
                "genres": "Genres",
                "recommendation_reason": "Why This Recommendation"
            })
            display_recs["Hours"] = display_recs["Hours"].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_recs, hide_index=True, use_container_width=True)

            # Show reverse recommendations
            with st.expander(f"See what {rec_target} could recommend to {rec_source}"):
                reverse_recs = comparison.get_recommendations_from_person(
                    compare_df, rec_target, rec_source, limit=10
                )
                if not reverse_recs.empty:
                    reverse_display = reverse_recs.rename(columns={
                        "artist_name": "Artist",
                        "source_hours": "Hours",
                        "source_plays": "Plays",
                        "genres": "Genres",
                        "recommendation_reason": "Why This Recommendation"
                    })
                    reverse_display["Hours"] = reverse_display["Hours"].apply(lambda x: f"{x:.1f}")
                    st.dataframe(reverse_display, hide_index=True)
                else:
                    st.info(f"{rec_target} doesn't have any artists that {rec_source} hasn't heard yet!")
        else:
            st.info(f"{rec_source} doesn't have any artists that {rec_target} hasn't already heard!")


# Page: Music Trends
elif page == "Music Trends":
    st.title("Music Trends")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Year-over-Year Comparison
    st.subheader("Year-over-Year Comparison")
    st.caption("Compare your listening habits between any two years")

    available_years = sorted(filtered_df["year"].unique().tolist())
    if len(available_years) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            yoy_year1 = st.selectbox("First Year:", options=available_years, index=0, key="yoy_year1")
        with col2:
            default_year2_idx = min(1, len(available_years) - 1)
            yoy_year2 = st.selectbox("Second Year:", options=available_years, index=default_year2_idx, key="yoy_year2")

        if yoy_year1 != yoy_year2:
            yoy_data = analysis.year_over_year_comparison(filtered_df, yoy_year1, yoy_year2)

            if yoy_data["year1_stats"] and yoy_data["year2_stats"]:
                # Side-by-side metrics
                st.write(f"**{yoy_year1} vs {yoy_year2}**")

                col1, col2, col3, col4, col5 = st.columns(5)
                metrics = [
                    ("Hours", "total_hours", ".0f"),
                    ("Streams", "total_streams", ","),
                    ("Artists", "unique_artists", ","),
                    ("Tracks", "unique_tracks", ","),
                    ("Albums", "unique_albums", ","),
                ]

                for col, (label, key, fmt) in zip([col1, col2, col3, col4, col5], metrics):
                    with col:
                        y1_val = yoy_data["year1_stats"][key]
                        y2_val = yoy_data["year2_stats"][key]
                        delta = yoy_data["deltas"].get(key)
                        delta_str = f"{delta:+.1f}%" if delta is not None else None
                        st.metric(
                            f"{label} ({yoy_year2})",
                            f"{y2_val:{fmt}}",
                            delta=delta_str,
                        )

                # Artist changes summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New Artists", yoy_data["new_artists_count"])
                with col2:
                    st.metric("Dropped Artists", yoy_data["dropped_artists_count"])
                with col3:
                    st.metric("Consistent Artists", yoy_data["consistent_artists_count"])

                # Top artists comparison
                col_left, col_right = st.columns(2)

                with col_left:
                    st.write(f"**Top New Artists in {yoy_year2}:**")
                    if not yoy_data["top_new_artists"].empty:
                        st.dataframe(
                            yoy_data["top_new_artists"].rename(columns={"artist_name": "Artist", "hours": "Hours"}),
                            hide_index=True,
                        )
                    else:
                        st.info("No new artists")

                with col_right:
                    st.write(f"**Dropped from {yoy_year1}:**")
                    if not yoy_data["top_dropped_artists"].empty:
                        st.dataframe(
                            yoy_data["top_dropped_artists"].rename(columns={"artist_name": "Artist", "hours": f"Hours ({yoy_year1})"}),
                            hide_index=True,
                        )
                    else:
                        st.info("No dropped artists")

                # Comparison chart
                comparison_df = pd.DataFrame({
                    "Metric": ["Hours", "Streams", "Artists", "Tracks", "Albums"],
                    str(yoy_year1): [
                        yoy_data["year1_stats"]["total_hours"],
                        yoy_data["year1_stats"]["total_streams"],
                        yoy_data["year1_stats"]["unique_artists"],
                        yoy_data["year1_stats"]["unique_tracks"],
                        yoy_data["year1_stats"]["unique_albums"],
                    ],
                    str(yoy_year2): [
                        yoy_data["year2_stats"]["total_hours"],
                        yoy_data["year2_stats"]["total_streams"],
                        yoy_data["year2_stats"]["unique_artists"],
                        yoy_data["year2_stats"]["unique_tracks"],
                        yoy_data["year2_stats"]["unique_albums"],
                    ],
                })

                comparison_melted = comparison_df.melt(id_vars=["Metric"], var_name="Year", value_name="Value")
                chart = alt.Chart(comparison_melted).mark_bar().encode(
                    x=alt.X("Metric:N", title=None),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Year:N", title="Year"),
                    xOffset="Year:N"
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            elif yoy_data["year1_stats"]:
                st.warning(f"No data available for {yoy_year2}")
            elif yoy_data["year2_stats"]:
                st.warning(f"No data available for {yoy_year1}")
            else:
                st.warning("No data available for either year")
        else:
            st.info("Select two different years to compare")
    else:
        st.info("Need at least 2 years of data for comparison")

    st.markdown("---")

    # Release Recency
    st.subheader("How Current is Your Music?")

    if "release_year" in filtered_df.columns:
        recency = analysis.release_recency_analysis(filtered_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if recency["avg_release_age"] is not None:
                st.metric("Avg Release Age", f"{recency['avg_release_age']:.1f} years")
        with col2:
            if recency["current_year_pct"] is not None:
                st.metric("Same Year Releases", f"{recency['current_year_pct']:.1f}%")
        with col3:
            if recency["last_5_years_pct"] is not None:
                st.metric("Last 5 Years", f"{recency['last_5_years_pct']:.1f}%")
        with col4:
            if recency["catalog_pct"] is not None:
                st.metric("Older Catalog", f"{recency['catalog_pct']:.1f}%")

        # Decade breakdown
        st.markdown("---")
        st.subheader("Listening by Release Decade")
        decade_df = analysis.decade_analysis(filtered_df)

        if not decade_df.empty:
            col_left, col_right = st.columns(2)
            with col_left:
                chart_data = decade_df.set_index("decade")["total_hours"]
                st.altair_chart(create_bar_chart(chart_data, "Decade", "Hours"), use_container_width=True)
            with col_right:
                st.dataframe(
                    decade_df.rename(columns={
                        "decade": "Decade",
                        "total_hours": "Hours",
                        "play_count": "Plays",
                        "unique_tracks": "Tracks"
                    }),
                    hide_index=True,
                )
    else:
        st.info("Release year data not available")

    # Artist Loyalty
    st.markdown("---")
    st.subheader("Artist Loyalty")
    st.caption("Artists you've been listening to the longest")

    loyalty = analysis.artist_loyalty(filtered_df, min_listens=20)

    if not loyalty.empty:
        st.dataframe(
            loyalty.head(20).rename(columns={
                "artist_name": "Artist",
                "loyalty_days": "Days Span",
                "total_hours": "Hours",
                "play_count": "Plays",
                "first_listen": "First Listen",
                "last_listen": "Last Listen"
            }),
            hide_index=True,
        )

    # Seasonal Patterns
    st.markdown("---")
    st.subheader("Seasonal Listening Patterns")

    seasonal = analysis.seasonal_patterns(filtered_df)

    if not seasonal.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            chart_data = seasonal.set_index("month_name")["avg_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Month", "Avg Hours/Year"), use_container_width=True)

        with col_right:
            st.dataframe(
                seasonal[["month_name", "avg_hours", "avg_plays"]].rename(
                    columns={"month_name": "Month", "avg_hours": "Avg Hours", "avg_plays": "Avg Plays"}
                ),
                hide_index=True,
            )

    # Time of Day Patterns
    st.markdown("---")
    st.subheader("Listening by Time of Day")

    hourly = analysis.listening_by_hour(filtered_df)
    daily = analysis.listening_by_day_of_week(filtered_df)

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("**By Hour**")
        if not hourly.empty:
            chart_data = hourly.set_index("hour")["total_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Hour", "Hours"), use_container_width=True)

    with col_right:
        st.write("**By Day of Week**")
        if not daily.empty:
            chart_data = daily.set_index("day_of_week")["total_hours"]
            st.altair_chart(create_bar_chart(chart_data, "Day", "Hours"), use_container_width=True)

    # Listening Heatmap
    st.markdown("---")
    st.subheader("Listening Heatmap")
    st.caption("When do you listen to music? Darker = more listening")

    heatmap_data = analysis.listening_heatmap_data(filtered_df)

    if not heatmap_data.empty:
        # Convert pivot to long format for Altair
        heatmap_long = heatmap_data.reset_index().melt(
            id_vars=["day_of_week"],
            var_name="hour",
            value_name="hours"
        )

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        heatmap_chart = alt.Chart(heatmap_long).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("day_of_week:O", title="Day", sort=day_order),
            color=alt.Color("hours:Q", scale=alt.Scale(scheme="blues"), title="Hours"),
            tooltip=["day_of_week", "hour", alt.Tooltip("hours:Q", format=".1f")]
        ).properties(width=700, height=300)

        st.altair_chart(heatmap_chart, use_container_width=True)
    else:
        st.info("Not enough data to generate heatmap")


# Page: Spotify Wrapped
elif page == "Spotify Wrapped":
    st.title("Spotify Wrapped Generator")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Year selector
    available_years = sorted(filtered_df["year"].unique().tolist(), reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Select Year:",
            options=available_years,
            index=0 if available_years else None
        )

    # Filter to selected year
    year_df = filtered_df[filtered_df["year"] == selected_year]

    if year_df.empty:
        st.warning(f"No data for {selected_year}")
        st.stop()

    # Calculate Wrapped stats
    st.markdown("---")
    st.header(f"Your {selected_year} Wrapped")

    # Total stats
    total_hours = year_df["hours_played"].sum()
    total_minutes = total_hours * 60
    total_streams = len(year_df)
    unique_artists = year_df["artist_name"].nunique()
    unique_tracks = year_df["track_name"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Minutes Listened", f"{total_minutes:,.0f}")
    with col2:
        st.metric("Streams", f"{total_streams:,}")
    with col3:
        st.metric("Different Artists", f"{unique_artists:,}")
    with col4:
        st.metric("Different Songs", f"{unique_tracks:,}")

    st.markdown("---")

    # Top 5 Artists
    st.subheader("Your Top Artists")
    top_5_artists = analysis.top_artists(year_df, limit=5)
    for i, (_, row) in enumerate(top_5_artists.iterrows(), 1):
        st.write(f"**#{i} {row['artist_name']}** - {row['total_hours']:.1f} hours ({row['play_count']:,} streams)")

    st.markdown("---")

    # Top 5 Songs
    st.subheader("Your Top Songs")
    top_5_tracks = analysis.top_tracks(year_df, limit=5)
    for i, (_, row) in enumerate(top_5_tracks.iterrows(), 1):
        st.write(f"**#{i} {row['track_name']}** by {row['artist_name']} - {row['play_count']:,} streams")

    st.markdown("---")

    # Top Genre (if available)
    if "artist_genres" in year_df.columns:
        st.subheader("Your Top Genre")
        top_genre = analysis.top_genres(year_df, limit=1)
        if not top_genre.empty:
            genre = top_genre.iloc[0]
            st.write(f"**{genre['genre']}** - {genre['total_hours']:.1f} hours across {genre['unique_artists']} artists")

    st.markdown("---")

    # Listening Personality
    st.subheader("Your Listening Personality")

    # Calculate personality based on patterns
    hipster = analysis.hipster_score(year_df)
    discovery = analysis.discovery_score(year_df)
    loyalty = analysis.artist_loyalty(year_df, min_listens=5)

    personality_traits = []

    # Hipster vs Mainstream
    if hipster["hipster_score"] is not None:
        if hipster["hipster_score"] > 60:
            personality_traits.append("The Hipster - You love discovering obscure music")
        elif hipster["hipster_score"] < 40:
            personality_traits.append("The Mainstream Maven - You know what's popular")
        else:
            personality_traits.append("The Balanced Listener - Mix of popular and obscure")

    # Explorer vs Loyalist
    if not loyalty.empty:
        avg_loyalty = loyalty["loyalty_days"].mean()
        if avg_loyalty > 365:
            personality_traits.append("The Loyalist - You stick with artists you love")
        else:
            personality_traits.append("The Explorer - Always finding new artists")

    # Variety
    tracks_per_artist = unique_tracks / unique_artists if unique_artists > 0 else 0
    if tracks_per_artist > 5:
        personality_traits.append("The Deep Diver - You explore artist catalogs thoroughly")
    else:
        personality_traits.append("The Sampler - You prefer variety over depth")

    for trait in personality_traits:
        st.write(f"- {trait}")

    st.markdown("---")

    # Time Machine - What were you listening to this day last year?
    st.subheader("Time Machine")
    st.caption(f"What were you listening to on this day in {selected_year}?")

    from datetime import datetime
    today = datetime.now()
    target_date = today.replace(year=selected_year)

    # Find streams on that day
    time_machine_df = year_df[year_df["timestamp"].dt.date == target_date.date()]

    if not time_machine_df.empty:
        st.write(f"**On {target_date.strftime('%B %d, %Y')}, you listened to:**")
        day_tracks = time_machine_df.groupby(["track_name", "artist_name"]).agg(
            play_count=("track_name", "count"),
        ).reset_index().sort_values("play_count", ascending=False)

        for _, row in day_tracks.head(5).iterrows():
            st.write(f"- {row['track_name']} by {row['artist_name']}")
    else:
        st.info(f"No listening data found for {target_date.strftime('%B %d, %Y')}")

    st.markdown("---")

    # Monthly breakdown chart
    st.subheader("Your Listening Throughout the Year")
    monthly = year_df.copy()
    monthly["month_name"] = monthly["timestamp"].dt.strftime("%B")
    monthly_grouped = monthly.groupby("month").agg(
        total_hours=("hours_played", "sum"),
        month_name=("month_name", "first"),
    ).reset_index().sort_values("month")

    if not monthly_grouped.empty:
        chart_data = monthly_grouped.set_index("month_name")["total_hours"]
        st.altair_chart(create_bar_chart(chart_data, "Month", "Hours"), use_container_width=True)


# Page: Artist Deep-Dive
elif page == "Artist Deep-Dive":
    st.title("Artist Deep-Dive")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Get all unique artists sorted by listening time
    all_artists = analysis.top_artists(filtered_df, limit=len(filtered_df))["artist_name"].tolist()

    selected_artist = st.selectbox(
        "Select an artist:",
        options=all_artists,
        index=all_artists.index("Bright Eyes") if "Bright Eyes" in all_artists else 0,
    )

    if selected_artist:
        artist_data = analysis.artist_deep_dive(filtered_df, selected_artist)

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
        timeline = analysis.artist_timeline(filtered_df, selected_artist)
        if not timeline.empty:
            chart_data = timeline.set_index("year_month")["total_hours"]
            st.altair_chart(create_line_chart(chart_data, "Month", "Hours"), use_container_width=True)

        # If multiple people, show who listens to this artist
        if "person" in filtered_df.columns and len(selected_people) > 1:
            st.markdown("---")
            st.subheader("Who Listens to This Artist?")
            artist_df = filtered_df[filtered_df["artist_name"] == selected_artist]
            person_breakdown = artist_df.groupby("person").agg(
                total_hours=("hours_played", "sum"),
                play_count=("track_name", "count"),
            ).reset_index().sort_values("total_hours", ascending=False)

            st.dataframe(
                person_breakdown.rename(columns={
                    "person": "Person",
                    "total_hours": "Hours",
                    "play_count": "Plays"
                }),
                hide_index=True,
            )


# Page: Song Deep-Dive
elif page == "Song Deep-Dive":
    st.title("Song Deep-Dive")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Get all unique tracks sorted by listening time
    all_tracks = analysis.top_tracks(filtered_df, limit=len(filtered_df))

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
        song_data = analysis.song_deep_dive(filtered_df, track_name, artist_name)

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
            st.subheader("Track Info")
            st.write(f"**Artist:** {artist_name}")

            # Show popularity if available
            if "track_popularity" in filtered_df.columns:
                track_df = filtered_df[(filtered_df["track_name"] == track_name) &
                                       (filtered_df["artist_name"] == artist_name)]
                if not track_df.empty and track_df["track_popularity"].notna().any():
                    popularity = track_df["track_popularity"].dropna().iloc[0]
                    st.write(f"**Spotify Popularity:** {popularity:.0f}/100")

            # Get artist rank
            all_artists = analysis.top_artists(filtered_df, limit=len(filtered_df))
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
        timeline = analysis.song_timeline(filtered_df, track_name, artist_name)
        if not timeline.empty:
            chart_data = timeline.set_index("year_month")["total_hours"]
            st.altair_chart(create_line_chart(chart_data, "Month", "Hours"), use_container_width=True)


# Page: Album Deep-Dive
elif page == "Album Deep-Dive":
    st.title("Album Deep-Dive")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Album Completion Analysis
    st.subheader("Album Completion Analysis")
    st.caption("How much of each album have you listened to?")

    completion_summary = analysis.album_completion_summary(filtered_df)

    if completion_summary["avg_completion"] is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Completion Rate", f"{completion_summary['avg_completion']:.1f}%")
        with col2:
            st.metric("Fully Completed", f"{completion_summary['full_albums']}")
        with col3:
            st.metric("Partially Completed", f"{completion_summary['partial_albums']}")
        with col4:
            st.metric("Single Track Only", f"{completion_summary['single_track_albums']}")

        # Show top albums by completion rate (filter to albums with >3 tracks)
        completion_df = analysis.album_completion_rate(filtered_df, limit=100)
        if not completion_df.empty and "completion_rate" in completion_df.columns:
            completion_valid = completion_df[
                (completion_df["completion_rate"].notna()) &
                (completion_df["total_tracks"].notna()) &
                (completion_df["total_tracks"] > 3)
            ].copy()

            if not completion_valid.empty:
                col_left, col_right = st.columns(2)

                with col_left:
                    st.write("**Most Completed Albums** (4+ tracks)")
                    top_completed = completion_valid.nlargest(10, "completion_rate")
                    # Create bar chart for completion rates
                    chart_df = top_completed[["album_name", "completion_rate"]].copy()
                    chart_df = chart_df.set_index("album_name")["completion_rate"]
                    completion_chart = alt.Chart(chart_df.reset_index()).mark_bar().encode(
                        x=alt.X("album_name:N", sort=None, title="Album"),
                        y=alt.Y("completion_rate:Q", scale=alt.Scale(domain=[0, 100]), title="Completion %")
                    ).properties(height=300)
                    st.altair_chart(completion_chart, use_container_width=True)

                with col_right:
                    st.write("**Top Albums by Completion Rate**")
                    display_cols = ["album_name", "artist_name", "tracks_played", "total_tracks", "completion_rate", "total_hours"]
                    display_df = top_completed[display_cols].rename(columns={
                        "album_name": "Album",
                        "artist_name": "Artist",
                        "tracks_played": "Played",
                        "total_tracks": "Total",
                        "completion_rate": "Completion %",
                        "total_hours": "Hours"
                    })
                    display_df["Completion %"] = display_df["Completion %"].apply(lambda x: f"{x:.1f}%")
                    display_df["Hours"] = display_df["Hours"].apply(lambda x: f"{x:.1f}")
                    st.dataframe(display_df, hide_index=True)
    else:
        st.info("Album completion data not available. Requires enriched data with album information.")

    st.markdown("---")
    st.subheader("Individual Album Details")

    # Get all unique albums sorted by listening time
    all_albums = analysis.top_albums(filtered_df, limit=len(filtered_df))

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
        album_data = analysis.album_deep_dive(filtered_df, album_name, artist_name)

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

            # Show album popularity and label if available
            if "album_popularity" in filtered_df.columns:
                album_df = filtered_df[(filtered_df["album_name"] == album_name) &
                                       (filtered_df["artist_name"] == artist_name)]
                if not album_df.empty:
                    if album_df["album_popularity"].notna().any():
                        pop = album_df["album_popularity"].dropna().iloc[0]
                        st.write(f"**Album Popularity:** {pop:.0f}/100")
                    if "label" in album_df.columns and album_df["label"].notna().any():
                        label = album_df["label"].dropna().iloc[0]
                        st.write(f"**Label:** {label}")

            # Get artist rank
            all_artists = analysis.top_artists(filtered_df, limit=len(filtered_df))
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
        timeline = analysis.album_timeline(filtered_df, album_name, artist_name)
        if not timeline.empty:
            chart_data = timeline.set_index("year_month")["total_hours"]
            st.altair_chart(create_line_chart(chart_data, "Month", "Hours"), use_container_width=True)


# Page: Time Range Analysis
elif page == "Time Range Analysis":
    st.title("Time Range Analysis")

    if len(selected_people) < len(all_people):
        st.info(f"Showing data for: {', '.join(selected_people)}")

    # Date range picker
    min_date = filtered_df["timestamp"].min().date()
    max_date = filtered_df["timestamp"].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date:", value=date(2020, 1, 1), min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End date:", value=date(2021, 12, 31), min_value=min_date, max_value=max_date)

    # Filter data
    range_df = analysis.filter_by_date_range(filtered_df, start_date, end_date)

    if range_df.empty:
        st.warning("No data found for the selected date range.")
        st.stop()

    # Stats for filtered range
    filtered_stats = analysis.generate_summary_stats(range_df)
    overall_stats = analysis.generate_summary_stats(filtered_df)

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
        top_artists = analysis.top_artists(range_df, limit=10)
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
        top_tracks = analysis.top_tracks(range_df, limit=10)
        st.dataframe(
            top_tracks[["track_name", "artist_name", "total_hours", "play_count"]].rename(
                columns={"track_name": "Track", "artist_name": "Artist", "total_hours": "Hours", "play_count": "Plays"}
            ),
            hide_index=True,
        )

    with col_right:
        st.subheader("Top 10 Albums")
        top_albums = analysis.top_albums(range_df, limit=10)
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
