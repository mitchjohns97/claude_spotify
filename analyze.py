#!/usr/bin/env python3
"""
Spotify Streaming History Analyzer

Analyze 13+ years of Spotify listening history with visualizations and statistics.

Usage:
    python analyze.py summary              # Overall stats
    python analyze.py top-artists [--limit N]  # Top artists
    python analyze.py top-tracks [--limit N]   # Top tracks
    python analyze.py top-albums [--limit N]   # Top albums
    python analyze.py timeline             # Listening over time
    python analyze.py time-of-day          # When you listen
    python analyze.py platforms            # Platform usage
    python analyze.py all                  # Generate all analyses with charts
"""

import argparse
import sys
from pathlib import Path

from src.loader import load_streaming_history, get_data_summary
from src import analysis
from src import visualize


def format_hours(hours: float) -> str:
    """Format hours into a readable string."""
    if hours >= 24:
        days = hours / 24
        return f"{hours:,.1f} hours ({days:,.1f} days)"
    return f"{hours:,.1f} hours"


def print_summary(df):
    """Print overall summary statistics."""
    stats = analysis.generate_summary_stats(df)

    print("\n" + "=" * 60)
    print("SPOTIFY STREAMING HISTORY SUMMARY")
    print("=" * 60)

    if "first_stream" in stats:
        print(f"\nDate Range: {stats['first_stream'].strftime('%Y-%m-%d')} to {stats['last_stream'].strftime('%Y-%m-%d')}")
        print(f"Years of Data: {stats['years_span']:.1f} years")

    print(f"\nTotal Streams: {stats['total_streams']:,}")
    print(f"Total Listening Time: {format_hours(stats['total_hours'])}")
    print(f"Unique Tracks: {stats['unique_tracks']:,}")
    print(f"Unique Artists: {stats['unique_artists']:,}")
    print(f"Unique Albums: {stats['unique_albums']:,}")

    print(f"\nListening Days: {stats.get('total_listening_days', 0):,}")
    print(f"Longest Streak: {stats.get('longest_streak', 0)} consecutive days")

    if "avg_hours_per_year" in stats:
        print(f"\nAverage per Year: {stats['avg_hours_per_year']:.1f} hours")
    if "avg_hours_per_day" in stats:
        print(f"Average per Listening Day: {stats['avg_hours_per_day']:.1f} hours")

    print()


def print_top_artists(df, limit=20):
    """Print top artists."""
    artists = analysis.top_artists(df, limit=limit)

    print("\n" + "=" * 60)
    print(f"TOP {limit} ARTISTS BY LISTENING TIME")
    print("=" * 60 + "\n")

    for i, row in artists.iterrows():
        print(f"{i+1:2}. {row['artist_name'][:40]:<40} {row['total_hours']:>7.1f}h  ({row['play_count']:,} plays)")

    print()


def print_top_tracks(df, limit=20):
    """Print top tracks."""
    tracks = analysis.top_tracks(df, limit=limit)

    print("\n" + "=" * 60)
    print(f"TOP {limit} TRACKS BY LISTENING TIME")
    print("=" * 60 + "\n")

    for i, row in tracks.iterrows():
        track_artist = f"{row['track_name'][:30]} - {row['artist_name'][:20]}"
        print(f"{i+1:2}. {track_artist:<52} {row['total_hours']:>6.1f}h  ({row['play_count']:,} plays)")

    print()


def print_top_albums(df, limit=20):
    """Print top albums."""
    albums = analysis.top_albums(df, limit=limit)

    print("\n" + "=" * 60)
    print(f"TOP {limit} ALBUMS BY LISTENING TIME")
    print("=" * 60 + "\n")

    for i, row in albums.iterrows():
        album_artist = f"{row['album_name'][:30]} - {row['artist_name'][:20]}"
        print(f"{i+1:2}. {album_artist:<52} {row['total_hours']:>6.1f}h")

    print()


def print_timeline(df):
    """Print listening timeline by year."""
    yearly = analysis.listening_by_year(df)

    print("\n" + "=" * 60)
    print("LISTENING TIMELINE BY YEAR")
    print("=" * 60 + "\n")

    print(f"{'Year':<8} {'Hours':>10} {'Streams':>12} {'Artists':>10} {'Tracks':>10}")
    print("-" * 54)

    for _, row in yearly.iterrows():
        print(f"{int(row['year']):<8} {row['total_hours']:>10.1f} {row['play_count']:>12,} {row['unique_artists']:>10,} {row['unique_tracks']:>10,}")

    print()


def print_time_of_day(df):
    """Print time-of-day listening patterns."""
    by_hour = analysis.listening_by_hour(df)
    by_day = analysis.listening_by_day_of_week(df)

    print("\n" + "=" * 60)
    print("LISTENING BY TIME OF DAY")
    print("=" * 60 + "\n")

    # Find peak hours
    peak_hour = by_hour.loc[by_hour["total_hours"].idxmax()]
    print(f"Peak Hour: {int(peak_hour['hour']):02d}:00 ({peak_hour['total_hours']:.1f} hours total)")

    # Morning/Afternoon/Evening/Night breakdown
    morning = by_hour[(by_hour["hour"] >= 6) & (by_hour["hour"] < 12)]["total_hours"].sum()
    afternoon = by_hour[(by_hour["hour"] >= 12) & (by_hour["hour"] < 18)]["total_hours"].sum()
    evening = by_hour[(by_hour["hour"] >= 18) & (by_hour["hour"] < 22)]["total_hours"].sum()
    night = by_hour[(by_hour["hour"] >= 22) | (by_hour["hour"] < 6)]["total_hours"].sum()

    total = morning + afternoon + evening + night
    print(f"\nTime of Day Breakdown:")
    print(f"  Morning (6am-12pm):   {morning:>8.1f}h ({morning/total*100:>5.1f}%)")
    print(f"  Afternoon (12pm-6pm): {afternoon:>8.1f}h ({afternoon/total*100:>5.1f}%)")
    print(f"  Evening (6pm-10pm):   {evening:>8.1f}h ({evening/total*100:>5.1f}%)")
    print(f"  Night (10pm-6am):     {night:>8.1f}h ({night/total*100:>5.1f}%)")

    print("\n" + "-" * 40)
    print("LISTENING BY DAY OF WEEK")
    print("-" * 40 + "\n")

    for _, row in by_day.iterrows():
        bar_len = int(row["total_hours"] / by_day["total_hours"].max() * 30)
        bar = "#" * bar_len
        print(f"{row['day_of_week']:<10} {row['total_hours']:>8.1f}h  {bar}")

    print()


def print_platforms(df):
    """Print platform usage statistics."""
    platforms = analysis.platform_usage(df)

    if platforms.empty:
        print("\nNo platform data available.")
        return

    print("\n" + "=" * 60)
    print("PLATFORM USAGE")
    print("=" * 60 + "\n")

    for _, row in platforms.iterrows():
        print(f"{row['platform']:<30} {row['total_hours']:>8.1f}h  ({row['percentage']:>5.1f}%)")

    print()


def run_all(df, output_dir: Path):
    """Run all analyses and generate charts."""
    print_summary(df)
    print_top_artists(df, limit=20)
    print_top_tracks(df, limit=20)
    print_top_albums(df, limit=20)
    print_timeline(df)
    print_time_of_day(df)
    print_platforms(df)

    # Gather analysis results for charts
    analysis_results = {
        "top_artists": analysis.top_artists(df, limit=20),
        "top_tracks": analysis.top_tracks(df, limit=20),
        "top_albums": analysis.top_albums(df, limit=20),
        "yearly": analysis.listening_by_year(df),
        "monthly": analysis.listening_by_month(df),
        "by_hour": analysis.listening_by_hour(df),
        "by_day": analysis.listening_by_day_of_week(df),
        "heatmap": analysis.listening_heatmap_data(df),
        "platform": analysis.platform_usage(df),
    }

    # Generate charts
    print("\n" + "=" * 60)
    print("GENERATING CHARTS")
    print("=" * 60 + "\n")

    charts = visualize.generate_all_charts(df, analysis_results, output_dir)

    print(f"\n{len(charts)} charts saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Spotify streaming history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "command",
        choices=["summary", "top-artists", "top-tracks", "top-albums",
                 "timeline", "time-of-day", "platforms", "all"],
        help="Analysis command to run"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Number of results to show (default: 20)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to spotify_data folder (default: ./spotify_data)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Directory for chart output (default: ./output)"
    )

    args = parser.parse_args()

    # Load data
    print("Loading Spotify streaming history...")
    try:
        df = load_streaming_history(args.data_dir)
        summary = get_data_summary(df)
        print(f"Loaded {summary['total_streams']:,} streams from {summary['unique_artists']:,} artists")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure your spotify_data folder contains the streaming history JSON files.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Run requested analysis
    if args.command == "summary":
        print_summary(df)
    elif args.command == "top-artists":
        print_top_artists(df, args.limit)
    elif args.command == "top-tracks":
        print_top_tracks(df, args.limit)
    elif args.command == "top-albums":
        print_top_albums(df, args.limit)
    elif args.command == "timeline":
        print_timeline(df)
    elif args.command == "time-of-day":
        print_time_of_day(df)
    elif args.command == "platforms":
        print_platforms(df)
    elif args.command == "all":
        run_all(df, args.output_dir)


if __name__ == "__main__":
    main()
