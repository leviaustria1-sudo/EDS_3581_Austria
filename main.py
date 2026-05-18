# Required module

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import seaborn as sns
import shutil
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Define paths and parameters

# File paths
project_root = Path(r"C:\Users\levia\OneDrive\Desktop - Copy\COMPROG\EDS_3581_Austria")

# Data and output folders
data_dir = project_root / "data"
output_dir = project_root / "outputs"

# Create directories if they don't exist
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Project root: {project_root}")
print(f"Data folder: {data_dir}")
print(f"Output folder: {output_dir}")

# Original data files
openweather_file = data_dir / "openweather_weather_airpollution_top3cities_per_country.csv"

# output cleaned file
cleaned_file = data_dir / "dataset_cleaned.csv"

# Define unique filtering parameters
START_HOUR = 0
END_HOUR = 23
PRESSURE_THRESHOLD = 900
WIND_SPEED_MAX = 10
CO_THRESHOLD = 0
MAX_TEMP_RANGE = 2.0      
TARGET_CITY = "Dubai"
MIN_AQI = 0

# Helper functions

def load_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded: {file_path.name} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def filter_by_unique_logic(df):
    initial_rows = len(df)
    
    if "city_name" in df.columns:
        df = df[df["city_name"] == TARGET_CITY].copy()
        print(f"\nFiltered by city: {TARGET_CITY} ({len(df)} rows)")
    else:
        print("Warning: 'city_name' column not found")
    
    if "collection_time_utc" in df.columns:
        df["DATETIME"] = pd.to_datetime(df["collection_time_utc"])
        df["HOUR"] = df["DATETIME"].dt.hour
        print(f"Hour range: {df['HOUR'].min()} to {df['HOUR'].max()}")
    else:
        print("Warning: No datetime column found")
    
    if "HOUR" in df.columns:
        df = df[(df["HOUR"] >= START_HOUR) & (df["HOUR"] <= END_HOUR)].copy()
        print(f"Filtered by time range: {START_HOUR}:00 to {END_HOUR}:00 UTC ({len(df)} rows)")
    
    if "pressure" in df.columns:
        df = df[df["pressure"] > PRESSURE_THRESHOLD].copy()
        print(f"Filtered by pressure: > {PRESSURE_THRESHOLD} hPa ({len(df)} rows)")
    
    if "wind_speed" in df.columns:
        df = df[df["wind_speed"] < WIND_SPEED_MAX].copy()
        print(f"Filtered by wind speed: < {WIND_SPEED_MAX} m/s ({len(df)} rows)")
    
    if "co" in df.columns:
        df = df[df["co"] > CO_THRESHOLD].copy()
        print(f"Filtered by CO: > {CO_THRESHOLD} µg/m³ ({len(df)} rows)")
    else:
        print("Note: 'co' column not found - skipping CO filter")
    
    if "target_qi_class" in df.columns:
        df = df[df["target_qi_class"] > MIN_AQI].copy()
        print(f"Filtered by AQI: > {MIN_AQI} ({len(df)} rows)")
    elif "aqi" in df.columns:
        df = df[df["aqi"] > MIN_AQI].copy()
        print(f"Filtered by AQI: > {MIN_AQI} ({len(df)} rows)")
    
    rows_removed = initial_rows - len(df)
    print(f"\nFinal filtered data: {len(df)} rows (removed {rows_removed} rows)")
    
    return df

def clean_data(df):
    initial_rows = len(df)

    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Duplicates removed: {duplicates_removed}")

    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"Missing values found in: {missing_cols}")
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"    - Filled '{col}' with median: {median_val:.2f}")
    else:
        print(f"No missing values found")

    print(f"Final cleaned data: {len(df)} rows")
    return df

def compute_statistics_numpy(df):
    stats_results = {}

    if "pressure" in df.columns:
        pressure_data = df["pressure"].values
        stats_results["pressure_mean"] = np.mean(pressure_data)
        stats_results["pressure_median"] = np.median(pressure_data)
        stats_results["pressure_std"] = np.std(pressure_data)
        stats_results["pressure_var"] = np.var(pressure_data)
        print(f"\nPressure (Stability Indicator):")
        print(f"    Mean: {stats_results['pressure_mean']:.1f} hPa")
        print(f"    Median: {stats_results['pressure_median']:.1f} hPa")
        print(f"    Std Dev: {stats_results['pressure_std']:.1f} hPa")
        print(f"    Variance: {stats_results['pressure_var']:.2f}")

    if "wind_speed" in df.columns:
        wind_data = df["wind_speed"].values
        stats_results["wind_mean"] = np.mean(wind_data)
        stats_results["wind_median"] = np.median(wind_data)
        stats_results["wind_std"] = np.std(wind_data)
        stats_results["wind_var"] = np.var(wind_data)
        print(f"\nWind Speed (Dispersion Factor):")
        print(f"    Mean: {stats_results['wind_mean']:.2f} m/s")
        print(f"    Median: {stats_results['wind_median']:.2f} m/s")
        print(f"    Std Dev: {stats_results['wind_std']:.2f} m/s")
        print(f"    Variance: {stats_results['wind_var']:.2f}")

    if "temp" in df.columns:
        temp_data = df["temp"].values
        stats_results["temp_mean"] = np.mean(temp_data)
        stats_results["temp_median"] = np.median(temp_data)
        stats_results["temp_std"] = np.std(temp_data)
        stats_results["temp_var"] = np.var(temp_data)
        print(f"\nTemperature:")
        print(f"    Mean: {stats_results['temp_mean']:.1f} °C")
        print(f"    Median: {stats_results['temp_median']:.1f} °C")
        print(f"    Std Dev: {stats_results['temp_std']:.1f} °C")
        print(f"    Variance: {stats_results['temp_var']:.2f}")

    # Correlation between Pressure and Temperature
    if "pressure" in df.columns and "temp" in df.columns:
        correlation = np.corrcoef(df["pressure"].values, df["temp"].values)[0, 1]
        stats_results["pressure_temp_correlation"] = correlation
        print(f"\nCorrelation (Pressure vs Temperature): {correlation:.4f}")

    return stats_results

# Distribution Analysis Function (Skewness)
def distribution_analysis(df):
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS - SKEWNESS")
    print("=" * 60)
    
    if "temp" in df.columns:
        temp_data = df["temp"].values
        n = len(temp_data)
        mean = np.mean(temp_data)
        std = np.std(temp_data)
        
        # Calculate skewness manually
        skewness = np.sum((temp_data - mean)**3) / (n * std**3)
        
        print(f"\nTemperature Distribution:")
        print(f"    Skewness: {skewness:.4f}")
        print(f"    Data Spread (Std Dev): {std:.2f}°C")
        print(f"    Range: {temp_data.min():.1f}°C to {temp_data.max():.1f}°C")
        
        # Interpretation
        if skewness > 0.5:
            print("    Interpretation: Right-skewed (more hot days than cold)")
        elif skewness < -0.5:
            print("    Interpretation: Left-skewed (more cold days than hot)")
        else:
            print("    Interpretation: Approximately symmetric")
        
        # Outlier detection
        outliers = temp_data[(temp_data < mean - 2*std) | (temp_data > mean + 2*std)]
        print(f"    Outliers detected: {len(outliers)} ({len(outliers)/n*100:.1f}%)")
    
    return skewness

# Comparative Analysis Function (Day vs Night)
def comparative_analysis(df):
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS - DAY VS NIGHT")
    print("=" * 60)
    
    # Day: 10:00-16:00, Night: 22:00-04:00
    day_df = df[(df["HOUR"] >= 10) & (df["HOUR"] <= 16)]
    night_df = df[(df["HOUR"] >= 22) | (df["HOUR"] <= 4)]
    
    if len(day_df) == 0 or len(night_df) == 0:
        print("  Not enough data for day/night comparison")
        print("  Using Morning vs Afternoon comparison instead.\n")
        
        morning_df = df[(df["HOUR"] >= 6) & (df["HOUR"] <= 11)]
        afternoon_df = df[(df["HOUR"] >= 12) & (df["HOUR"] <= 17)]
        
        if len(morning_df) > 0 and len(afternoon_df) > 0:
            print(f"  Morning (06:00-11:00) - {len(morning_df)} records:")
            if "temp" in morning_df.columns:
                print(f"    Mean Temperature: {morning_df['temp'].mean():.1f} °C")
            if "pressure" in morning_df.columns:
                print(f"    Mean Pressure: {morning_df['pressure'].mean():.1f} hPa")
            
            print(f"\n  Afternoon (12:00-17:00) - {len(afternoon_df)} records:")
            if "temp" in afternoon_df.columns:
                print(f"    Mean Temperature: {afternoon_df['temp'].mean():.1f} °C")
            if "pressure" in afternoon_df.columns:
                print(f"    Mean Pressure: {afternoon_df['pressure'].mean():.1f} hPa")
            
            if "temp" in morning_df.columns and "temp" in afternoon_df.columns:
                temp_diff = afternoon_df['temp'].mean() - morning_df['temp'].mean()
                print(f"\n  Temperature Difference: Afternoon is {temp_diff:.1f}°C warmer than Morning")
        else:
            print("  Not enough data for comparison.")
        return
    
    print(f"\n  Daytime (10:00-16:00) - {len(day_df)} records:")
    if "temp" in day_df.columns:
        print(f"    Mean Temperature: {day_df['temp'].mean():.1f} °C")
        print(f"    Temperature Std: {day_df['temp'].std():.1f} °C")
    if "pressure" in day_df.columns:
        print(f"    Mean Pressure: {day_df['pressure'].mean():.1f} hPa")
    
    print(f"\n  Nighttime (22:00-04:00) - {len(night_df)} records:")
    if "temp" in night_df.columns:
        print(f"    Mean Temperature: {night_df['temp'].mean():.1f} °C")
        print(f"    Temperature Std: {night_df['temp'].std():.1f} °C")
    if "pressure" in night_df.columns:
        print(f"    Mean Pressure: {night_df['pressure'].mean():.1f} hPa")
    
    if "temp" in day_df.columns and "temp" in night_df.columns:
        temp_diff = day_df['temp'].mean() - night_df['temp'].mean()
        print(f"\n  Temperature Difference: Day is {temp_diff:.1f}°C warmer than Night")


# Figure 1: Boxplot of Temperature by Hour
def create_boxplot_temp_by_hour(df):
    print("\n  Creating boxplot (Temperature by Hour)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    hourly_data = [
        df[df["HOUR"] == h]["temp"].values for h in range(START_HOUR, END_HOUR + 1)
    ]
    hours = [f"{h:02d}:00" for h in range(START_HOUR, END_HOUR + 1)]

    bp = ax.boxplot(
        hourly_data,
        labels=hours,
        patch_artist=True,
        boxprops=dict(linewidth=2),
        medianprops=dict(linewidth=2, color="red"),
    )

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(hours)))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])

    ax.set_xlabel("Hour (UTC)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        "Figure 1: Temperature Variation Throughout the Day",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure1_Boxplot_Temp_by_Hour.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

# Figure 2: Scatter plot of Pressure vs Temperature
def create_scatter_pressure_vs_temp(df):
    print("  Creating scatter plot (Pressure vs Temperature)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df["pressure"],
        df["temp"],
        c=df["HOUR"],
        cmap="viridis",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    if len(df) > 1:
        z = np.polyfit(df["pressure"], df["temp"], 1)
        p = np.poly1d(z)
        corr = np.corrcoef(df["pressure"], df["temp"])[0, 1]
        ax.plot(
            np.sort(df["pressure"]),
            p(np.sort(df["pressure"])),
            "r--",
            linewidth=2,
            label=f"Trend (r={corr:.3f})",
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Hour (UTC)", fontsize=10)

    ax.set_xlabel("Atmospheric Pressure (hPa)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        "Figure 2: Pressure vs Temperature Relationship",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure2_Scatter_Pressure_vs_Temp.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

# Figure 3: Correlation heatmap
def create_correlation_heatmap(df):
    print("  Creating correlation heatmap...")

    fig, ax = plt.subplots(figsize=(10, 8))

    corr_cols = []
    for col in ["pressure", "wind_speed", "temp", "humidity", "clouds_all"]:
        if col in df.columns:
            corr_cols.append(col)

    if len(corr_cols) < 2:
        print("  Not enough columns for correlation heatmap")
        return

    corr_data = df[corr_cols].copy()
    corr_matrix = corr_data.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 10},
    )

    ax.set_title(
        "Figure 3: Correlation Matrix - Weather Variables",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure3_Correlation_Heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

# Figure 4: Time series of Temperature
def create_temperature_time_series(df):
    print("  Creating temperature time series...")

    fig, ax = plt.subplots(figsize=(12, 6))

    df_sorted = df.sort_values("DATETIME")

    ax.plot(df_sorted["DATETIME"], df_sorted["temp"], "b-", linewidth=1.5, label="Temperature")
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        "Figure 4: Temperature Time Series",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure4_Temperature_Time_Series.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

# Animation 1: Time series animation
def create_animated_time_series(df):
    print("\n  Creating animated time series...")

    df_sorted = df.sort_values("DATETIME")
    timestamps = df_sorted["DATETIME"].unique()
    timestamps = np.sort(timestamps)

    if len(timestamps) > 100:
        timestamps = timestamps[::2]

    fig, ax = plt.subplots(figsize=(12, 6))

    (line1,) = ax.plot([], [], "b-", linewidth=2, label="Temperature")
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.set_title(
        "Animation 1: Temperature Evolution Over Time",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    all_temp = df_sorted["temp"].values
    if len(all_temp) > 0:
        ax.set_ylim(min(all_temp) - 5, max(all_temp) + 5)
    ax.set_xlim(timestamps[0], timestamps[-1])

    def init():
        line1.set_data([], [])
        return line1,

    def update(frame):
        current_data = df_sorted[df_sorted["DATETIME"] <= timestamps[frame]]
        if len(current_data) > 0:
            line1.set_data(current_data["DATETIME"], current_data["temp"])
        return line1,

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(timestamps),
        init_func=init,
        interval=100,
        repeat=False,
        blit=True,
    )
    anim.save(
        output_dir / "Animation1_Temperature_Evolution.gif",
        writer=PillowWriter(fps=20),
    )
    plt.close()

    freeze_indices = [0, len(timestamps) // 2, len(timestamps) - 1]
    for i, idx in enumerate(freeze_indices):
        fig_frame, ax_frame = plt.subplots(figsize=(12, 6))
        
        current_data = df_sorted[df_sorted["DATETIME"] <= timestamps[idx]]
        
        ax_frame.plot(current_data["DATETIME"], current_data["temp"], "b-", linewidth=1.5)
        ax_frame.set_ylabel("Temperature (°C)")
        ax_frame.set_xlabel("Date/Time")
        ax_frame.set_title(f"Freeze-Frame {i+1}: Data up to {pd.to_datetime(timestamps[idx]).strftime('%Y-%m-%d %H:%M')}")
        ax_frame.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(
            output_dir / f"Animation1_FreezeFrame_{i+1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

# Animation 2: Hourly temperature bars - FIXED VERSION
def create_animated_hourly_bars(df):
    print("  Creating animated hourly bars...")

    hourly_temp = df.groupby("HOUR")["temp"].mean().reset_index()
    hourly_temp = hourly_temp.sort_values("HOUR")

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        range(len(hourly_temp)),
        [0] * len(hourly_temp),
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )

    hour_labels = [f"{int(h):02d}:00" for h in hourly_temp["HOUR"]]
    ax.set_xticks(range(len(hourly_temp)))
    ax.set_xticklabels(hour_labels)
    ax.set_ylabel("Average Temperature (°C)", fontsize=12)
    ax.set_xlabel("Hour (UTC)", fontsize=12)
    ax.set_title(
        "Animation 2: Hourly Temperature Pattern",
        fontsize=14,
        fontweight="bold",
    )
    if len(hourly_temp) > 0:
        ax.set_ylim(0, hourly_temp["temp"].max() * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    def update(frame):
        for i, bar in enumerate(bars):
            if i <= frame:
                bar.set_height(hourly_temp.iloc[i]["temp"])
        return bars

    anim = animation.FuncAnimation(
        fig, update, frames=len(hourly_temp), interval=500, repeat=False, blit=False
    )
    anim.save(
        output_dir / "Animation2_Hourly_Temperature.gif",
        writer=PillowWriter(fps=10),
    )
    plt.close()

    # Save freeze-frames - FIXED VERSION
    freeze_indices = [0, len(hourly_temp) // 2, len(hourly_temp) - 1]
    for i, idx in enumerate(freeze_indices):
        fig_frame, ax_frame = plt.subplots(figsize=(12, 6))

        x_pos = range(idx + 1)
        heights = hourly_temp.iloc[: idx + 1]["temp"].values

        ax_frame.bar(x_pos, heights, color="steelblue", alpha=0.7, edgecolor="black")
        ax_frame.set_xticks(x_pos)
        freeze_labels = [f"{int(h):02d}:00" for h in hourly_temp.iloc[: idx + 1]["HOUR"]]
        ax_frame.set_xticklabels(freeze_labels)
        ax_frame.set_ylabel("Average Temperature (°C)")
        ax_frame.set_xlabel("Hour (UTC)")
        ax_frame.set_title(f"Freeze-Frame {i+1}: First {idx+1} hours")
        ax_frame.grid(True, alpha=0.3, axis="y")
        if len(hourly_temp) > 0:
            ax_frame.set_ylim(0, hourly_temp["temp"].max() * 1.2)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"Animation2_FreezeFrame_{i+1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

# Backup function
def create_backup(df, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"backup_{filename}_{timestamp}.csv"
    df.to_csv(backup_path, index=False)
    print(f"Backup saved: {backup_path.name}")

# Main Execution
def main():
    print("\n" + "=" * 70)
    print("    ENV-01: THERMAL INVERSION & CO TRAPPING ANALYSIS")
    print("    Weather Data Analytics Pipeline")
    print("=" * 70)

    if not openweather_file.exists():
        print(f"\n[X] ERROR: CSV file not found!")
        print(f"    Expected location: {openweather_file}")
        return

    df = load_csv_file(openweather_file)
    if df is None:
        print("\n[X] ERROR: Could not load data file. Please check file path.")
        return

    print(f"\nAvailable cities: {df['city_name'].unique()}")

    df = filter_by_unique_logic(df)
    if len(df) == 0:
        print("\n[X] ERROR: No data remaining after filtering. Check filter parameters.")
        print("   Try: Changing TARGET_CITY to a different city")
        return

    df = clean_data(df)
    stats = compute_statistics_numpy(df)
    
    # Distribution Analysis
    distribution_analysis(df)
    
    # Comparative Analysis
    comparative_analysis(df)
    
    create_backup(df, "cleaned_data.csv")
    df.to_csv(cleaned_file, index=False)
    print(f"\n[/] Cleaned data saved to: {cleaned_file}")

    create_boxplot_temp_by_hour(df)
    create_scatter_pressure_vs_temp(df)
    create_correlation_heatmap(df)
    create_temperature_time_series(df)
    create_animated_time_series(df)
    create_animated_hourly_bars(df)

    sep = "=" * 60
    print(f"\n{sep}")
    print("FINAL SUMMARY - WEATHER DATA ANALYSIS")
    print(sep)
    print(f"\n  Dataset Overview:")
    print(f"    - Total records analyzed: {len(df)}")
    print(f"    - City: {TARGET_CITY}")
    print(f"    - Time window: {START_HOUR}:00 to {END_HOUR}:00 UTC")
    print(f"\n  Key Findings:")
    if "pressure_mean" in stats:
        print(f"    - Mean pressure: {stats['pressure_mean']:.1f} hPa")
    if "pressure_var" in stats:
        print(f"    - Pressure variance: {stats['pressure_var']:.2f}")
    if "wind_mean" in stats:
        print(f"    - Mean wind speed: {stats['wind_mean']:.2f} m/s")
    if "temp_mean" in stats:
        print(f"    - Mean temperature: {stats['temp_mean']:.1f} °C")
    if "temp_var" in stats:
        print(f"    - Temperature variance: {stats['temp_var']:.2f}")
    if "pressure_temp_correlation" in stats:
        print(f"    - Pressure-Temp correlation: {stats['pressure_temp_correlation']:.4f}")

    print(f"\n{sep}")
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("\n  Generated files:")
    print("    - dataset_cleaned.csv")
    print("    - Figure1_Boxplot_Temp_by_Hour.png")
    print("    - Figure2_Scatter_Pressure_vs_Temp.png")
    print("    - Figure3_Correlation_Heatmap.png")
    print("    - Figure4_Temperature_Time_Series.png")
    print("    - Animation1_Temperature_Evolution.gif")
    print("    - Animation2_Hourly_Temperature.gif")
    print("    - Animation1_FreezeFrame_1/2/3.png")
    print("    - Animation2_FreezeFrame_1/2/3.png")
    print(sep)


if __name__ == "__main__":
    main()