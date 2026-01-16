# Spotify Streaming History Analyzer

Analyze 13+ years of Spotify streaming history (2011-2024) with visualizations and statistics.

## Project Structure

```
claude_spotify/
├── requirements.txt          # pandas, matplotlib, seaborn, streamlit
├── spotify_data/             # Place JSON files here
├── src/
│   ├── __init__.py
│   ├── loader.py             # Load and combine all JSON files
│   ├── analysis.py           # Statistical analysis functions
│   └── visualize.py          # Chart generation
├── analyze.py                # Main CLI entry point
├── app.py                    # Streamlit web app
├── output/                   # Generated charts (created automatically)
└── CLAUDE.md                 # This file
```

## Setup

```bash
cd claude_spotify
pip install -r requirements.txt
```

## Data Format

Place your Spotify extended streaming history JSON files in `spotify_data/`. The files should contain records with:

- `ts`: timestamp
- `ms_played`: milliseconds played
- `master_metadata_track_name`: song name
- `master_metadata_album_artist_name`: artist
- `master_metadata_album_album_name`: album
- `spotify_track_uri`: track ID
- `platform`: device type (ios, android, etc.)
- `skipped`: whether track was skipped
- `shuffle`: shuffle mode status

## Usage

```bash
# Overall statistics
python analyze.py summary

# Top artists (default: 20)
python analyze.py top-artists
python analyze.py top-artists --limit 50

# Top tracks
python analyze.py top-tracks
python analyze.py top-tracks -n 30

# Top albums
python analyze.py top-albums

# Listening timeline by year
python analyze.py timeline

# Time of day analysis
python analyze.py time-of-day

# Platform usage
python analyze.py platforms

# Run all analyses and generate charts
python analyze.py all
```

## Streamlit Web App

Interactive web interface for exploring listening history.

```bash
streamlit run app.py
```

Opens at http://localhost:8501

### Pages

**Overview** (default)
- Summary stats (total hours, streams, unique artists/tracks/albums)
- Top 10 artists bar chart
- Listening by year bar chart

**Artist Deep-Dive**
- Searchable dropdown to select any artist
- Total listening time and play count
- First/last listen dates
- Top 10 tracks by that artist (chart + table)
- Top 5 albums by that artist (chart + table)
- Monthly listening timeline for that artist

**Time Range Analysis**
- Date range picker (start/end dates)
- Stats for selected range
- Top 10 artists, tracks, albums in range
- Comparison table vs overall stats

## Output

Charts are saved to the `output/` folder:

- `top_artists.png` - Bar chart of top artists
- `top_tracks.png` - Bar chart of top tracks
- `top_albums.png` - Bar chart of top albums
- `listening_timeline.png` - Line chart by year
- `monthly_timeline.png` - Line chart by month
- `hour_distribution.png` - Bar chart by hour of day
- `day_distribution.png` - Bar chart by day of week
- `listening_heatmap.png` - Heatmap (hour x day)
- `platform_distribution.png` - Pie chart of platforms

## Modules

### `src/loader.py`
- `load_streaming_history(data_dir)` - Load all JSON files into DataFrame
- `get_data_summary(df)` - Quick summary statistics

### `src/analysis.py`
- `top_artists(df, limit, by)` - Top artists by time or count
- `top_tracks(df, limit, by)` - Top tracks by time or count
- `top_albums(df, limit)` - Top albums by time
- `listening_by_year(df)` - Yearly aggregates
- `listening_by_month(df)` - Monthly aggregates
- `listening_by_hour(df)` - Hour of day patterns
- `listening_by_day_of_week(df)` - Day of week patterns
- `listening_heatmap_data(df)` - Pivot for heatmap
- `platform_usage(df)` - Platform breakdown
- `skip_analysis(df, limit)` - Skip behavior analysis
- `listening_streaks(df)` - Consecutive listening days
- `generate_summary_stats(df)` - Comprehensive stats
- `filter_by_date_range(df, start_date, end_date)` - Filter to date range
- `artist_deep_dive(df, artist_name)` - Detailed stats for one artist
- `artist_timeline(df, artist_name)` - Monthly timeline for one artist

### `src/visualize.py`
- `plot_top_artists(df, output_dir)` - Bar chart
- `plot_top_tracks(df, output_dir)` - Bar chart
- `plot_top_albums(df, output_dir)` - Bar chart
- `plot_listening_timeline(df, output_dir)` - Line chart
- `plot_monthly_timeline(df, output_dir)` - Line chart
- `plot_hour_distribution(df, output_dir)` - Bar chart
- `plot_day_distribution(df, output_dir)` - Bar chart
- `plot_heatmap(df, output_dir)` - Seaborn heatmap
- `plot_platform_pie(df, output_dir)` - Pie chart
- `generate_all_charts(df, results, output_dir)` - All charts

### `app.py`
Streamlit web app with three pages: Overview, Artist Deep-Dive, Time Range Analysis. Uses session state for caching loaded data.
