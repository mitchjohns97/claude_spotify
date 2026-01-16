"""Chart generation for Spotify streaming history."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100


def ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_top_artists(df: pd.DataFrame, output_dir: Path, limit: int = 20) -> Path:
    """Create bar chart of top artists by listening time."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 8))

    data = df.head(limit)
    colors = sns.color_palette("viridis", len(data))

    bars = ax.barh(range(len(data)), data["total_hours"], color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["artist_name"])
    ax.invert_yaxis()

    ax.set_xlabel("Hours Listened")
    ax.set_title(f"Top {limit} Artists by Listening Time")
    ax.set_xlim(left=0)

    # Add value labels
    for bar, hours in zip(bars, data["total_hours"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{hours:.1f}h", va="center", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "top_artists.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_top_tracks(df: pd.DataFrame, output_dir: Path, limit: int = 20) -> Path:
    """Create bar chart of top tracks by listening time."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 8))

    data = df.head(limit)
    labels = [f"{row['track_name'][:30]} - {row['artist_name'][:20]}"
              for _, row in data.iterrows()]

    colors = sns.color_palette("magma", len(data))
    bars = ax.barh(range(len(data)), data["total_hours"], color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Hours Listened")
    ax.set_title(f"Top {limit} Tracks by Listening Time")
    ax.set_xlim(left=0)

    for bar, hours in zip(bars, data["total_hours"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{hours:.1f}h", va="center", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "top_tracks.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_top_albums(df: pd.DataFrame, output_dir: Path, limit: int = 20) -> Path:
    """Create bar chart of top albums by listening time."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 8))

    data = df.head(limit)
    labels = [f"{row['album_name'][:30]} - {row['artist_name'][:20]}"
              for _, row in data.iterrows()]

    colors = sns.color_palette("rocket", len(data))
    bars = ax.barh(range(len(data)), data["total_hours"], color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Hours Listened")
    ax.set_title(f"Top {limit} Albums by Listening Time")
    ax.set_xlim(left=0)

    for bar, hours in zip(bars, data["total_hours"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{hours:.1f}h", va="center", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "top_albums.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_listening_timeline(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create line chart of listening over time by year."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["year"], df["total_hours"], marker="o", linewidth=2, markersize=8,
            color="#1DB954")  # Spotify green
    ax.fill_between(df["year"], df["total_hours"], alpha=0.3, color="#1DB954")

    ax.set_xlabel("Year")
    ax.set_ylabel("Hours Listened")
    ax.set_title("Listening Time Over the Years")
    ax.set_ylim(bottom=0)

    # Add value labels
    for _, row in df.iterrows():
        ax.annotate(f"{row['total_hours']:.0f}h",
                    (row["year"], row["total_hours"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.set_xticks(df["year"])
    plt.tight_layout()
    output_path = output_dir / "listening_timeline.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_monthly_timeline(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create line chart of listening by month."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Sample if too many points
    if len(df) > 60:
        # Show every nth point for readability
        step = len(df) // 60 + 1
        plot_df = df.iloc[::step]
    else:
        plot_df = df

    ax.plot(range(len(plot_df)), plot_df["total_hours"], linewidth=1.5, color="#1DB954")
    ax.fill_between(range(len(plot_df)), plot_df["total_hours"], alpha=0.3, color="#1DB954")

    # Show subset of x-labels
    n_labels = min(20, len(plot_df))
    step = max(1, len(plot_df) // n_labels)
    ax.set_xticks(range(0, len(plot_df), step))
    ax.set_xticklabels(plot_df["year_month"].iloc[::step], rotation=45, ha="right")

    ax.set_xlabel("Month")
    ax.set_ylabel("Hours Listened")
    ax.set_title("Monthly Listening Time")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / "monthly_timeline.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_hour_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create bar chart of listening by hour of day."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("coolwarm", len(df))
    ax.bar(df["hour"], df["total_hours"], color=colors)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total Hours")
    ax.set_title("When Do You Listen? (Hour of Day)")
    ax.set_xticks(range(0, 24))
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / "hour_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_day_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create bar chart of listening by day of week."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(df))
    ax.bar(df["day_of_week"], df["total_hours"], color=colors)

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Hours")
    ax.set_title("Listening by Day of Week")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_path = output_dir / "day_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_heatmap(pivot_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create heatmap of listening patterns (hour x day of week)."""
    output_dir = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(pivot_df, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Hours"})

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Listening Heatmap (Hour × Day of Week)")

    plt.tight_layout()
    output_path = output_dir / "listening_heatmap.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def plot_platform_pie(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create pie chart of platform usage."""
    output_dir = ensure_output_dir(output_dir)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Combine small platforms into "Other"
    threshold = 2  # percent
    main_platforms = df[df["percentage"] >= threshold].copy()
    other_hours = df[df["percentage"] < threshold]["total_hours"].sum()

    if other_hours > 0:
        other_row = pd.DataFrame({
            "platform": ["Other"],
            "total_hours": [other_hours],
            "percentage": [other_hours / df["total_hours"].sum() * 100]
        })
        main_platforms = pd.concat([main_platforms, other_row], ignore_index=True)

    colors = sns.color_palette("Set2", len(main_platforms))
    wedges, texts, autotexts = ax.pie(
        main_platforms["total_hours"],
        labels=main_platforms["platform"],
        autopct="%1.1f%%",
        colors=colors,
        explode=[0.02] * len(main_platforms)
    )

    ax.set_title("Platform Distribution")

    plt.tight_layout()
    output_path = output_dir / "platform_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def generate_all_charts(
    df: pd.DataFrame,
    analysis_results: dict,
    output_dir: Path
) -> list[Path]:
    """Generate all available charts."""
    output_dir = ensure_output_dir(output_dir)
    generated = []

    print("Generating charts...")

    if "top_artists" in analysis_results:
        path = plot_top_artists(analysis_results["top_artists"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "top_tracks" in analysis_results:
        path = plot_top_tracks(analysis_results["top_tracks"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "top_albums" in analysis_results:
        path = plot_top_albums(analysis_results["top_albums"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "yearly" in analysis_results:
        path = plot_listening_timeline(analysis_results["yearly"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "monthly" in analysis_results:
        path = plot_monthly_timeline(analysis_results["monthly"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "by_hour" in analysis_results:
        path = plot_hour_distribution(analysis_results["by_hour"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "by_day" in analysis_results:
        path = plot_day_distribution(analysis_results["by_day"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "heatmap" in analysis_results:
        path = plot_heatmap(analysis_results["heatmap"], output_dir)
        generated.append(path)
        print(f"  Created: {path.name}")

    if "platform" in analysis_results and not analysis_results["platform"].empty:
        path = plot_platform_pie(analysis_results["platform"], output_dir)
        if path:
            generated.append(path)
            print(f"  Created: {path.name}")

    return generated
