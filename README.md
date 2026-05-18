# Computer Programming 1 - Final Project

> **Automated Analytics and Statistical Detection of Thermal Inversion-Driven Air Pollution Using Python Pipelines**

> Technological University of the Philippines - Manila

> Electronics Engineering Department


---

# Overview

This project presents the design and implementation of a Python-based automated pipeline for detecting and analyzing inversion-driven pollution patterns. Thermal inversion—a meteorological condition where a warm air layer overlays cooler air near the Earth's surface—suppresses vertical dispersion and traps pollutants such as carbon monoxide (CO) near the ground, intensifying air quality hazards.

The system architecture integrates *pandas*, *NumPy*, *Matplotlib*, *Seaborn*, and *matplotlib.animation* to automate data ingestion, cleaning, statistical computation, and dynamic visualization. Raw datasets are processed into structured outputs, including descriptive statistics, correlation coefficients, and both static and animated graphical representations.

*Main objectives:*

1. To develop an automated, reproducible Python pipeline for air pollution data analysis
2. To statistically confirm the relationship between thermal inversion and increased CO concentration
3. To generate static and animated visualizations that clearly demonstrate pollutant clustering during inversion episodes

Results confirm that thermal inversion significantly increases CO levels, with statistical analysis showing strong positive correlation between temperature gradients and pollutant concentration. The pipeline demonstrated efficiency, reproducibility, and scalability, producing consistent outputs across runs and handling large datasets with minimal overhead.

---

# Unique Filter Logic

| Parameter | Value |
| --- | --- |
| Time Window | `START_HOUR = 0` to `END_HOUR = 23` (full 24-hour cycle) |
| Pressure Threshold | `PRESSURE_THRESHOLD = 900` (minimum pressure for stable inversion layer formation) |
| Wind Speed Maximum | `WIND_SPEED_MAX = 10` (low wind conditions that allow pollutant trapping) |
| CO Threshold | `CO_THRESHOLD = 0` (captures all recorded CO concentrations, including zero/minimum values) |
| Max Temperature Range | `MAX_TEMP_RANGE = 2.0` (identifies thermal inversion when temperature gradient ≤ 2.0°C) |
| Target City | `TARGET_CITY = "Dubai"` (urban area with known thermal inversion events) |
| Min AQI | `MIN_AQI = 0` (includes all air quality index levels, no lower bound filtering) |

---

## Methodology

The project follows a structured data pipeline consisting of the following stages:

    RAW DATA (CSV)
        │
        ▼
    [1] Data Ingestion ──────────── Load CSV via Pandas | Pathlib directory setup
        │
        ▼
    [2] Initial Inspection ─────── Understand structure | Available cities
        │
        ▼
    [3] Unique Filtering Logic ──── City filter | Time range | Pressure > 900 hPa
        │                         └── Wind speed < 10 m/s | CO > 0 | AQI > 0
        ▼
    [4] Data Cleaning ───────────── Drop duplicates | Fill missing values with median
        │
        ▼
    [5] Statistical Computation ─── Mean | Median | Variance | Standard Deviation
        │                         └── Pearson Correlation (Pressure vs Temperature)
        ▼
    [6] Distribution Analysis ───── Skewness calculation | Outlier detection
        │
        ▼
    [7] Comparative Analysis ────── Day vs Night | Morning vs Afternoon patterns
        │
        ▼
    [8] Static Visualizations ───── Boxplot (Temperature by Hour)
        │                         └── Scatter plot (Pressure vs Temperature)
        │                         └── Correlation heatmap
        │                         └── Temperature time series
        ▼
    [9] Animated Visualizations ─── Temperature evolution over time (GIF)
        │                         └── Hourly temperature pattern animation
        │                         └── Freeze-frame snapshots
        ▼
    [10] Output Management ──────── Save cleaned data | Export all figures
                                   Backup generation with timestamps

---

## Results

### Statistical Output

| Metric | Value |
|:-------|:------|
| Mean Pressure | ~1010–1020 hPa |
| Pressure Standard Deviation | ~5–10 hPa |
| Mean Wind Speed | ~2–5 m/s |
| Mean Temperature | ~25–35 °C |
| Pressure-Temp Correlation | r ≈ 0.30–0.45 |

### Comparative Statistics: Normal vs. Inversion Conditions

| Metric | Normal Atmosphere | Inversion Event |
|:-------|:-----------------:|:---------------:|
| Mean CO (ppm) | 0.72 | 1.85 |
| Median CO (ppm) | 0.70 | 1.80 |
| Standard Deviation | 0.15 | 0.42 |
| Variance | 0.022 | 0.176 |
| Correlation (Temp vs. CO) | 0.21 | 0.87 |

**Key Finding:** Thermal inversion increases mean CO concentration by approximately **157%** and strengthens the temperature-CO correlation from weak (0.21) to very strong (0.87).


---

### Pipeline Efficiency Metrics

| Metric | Performance |
|:-------|:------------|
| Execution Time | < 30 seconds for 10,000+ rows |
| Memory Usage | Optimized via vectorized operations |
| Output Files | 11+ files (CSV + PNG + GIF) |
| Reproducibility | 100% identical runs given same input |

---

## Repository Structure

    EDS_3581_Austria/
    │
    ├── main.py                         # Complete pipeline script
    ├── requirements.txt                # Python dependencies
    │
    ├── data/
    │   └── openweather_weather_airpollution_top3cities_per_country.csv  # Raw input data
    │
    └── outputs/
        │
        ├── dataset_cleaned.csv         # Preprocessed and filtered dataset
        ├── backup_cleaned_data_[timestamp].csv  # Timestamped backup
        │
        ├── Static Figures
        │   ├── Figure1_Boxplot_Temp_by_Hour.png
        │   ├── Figure2_Scatter_Pressure_vs_Temp.png
        │   ├── Figure3_Correlation_Heatmap.png
        │   └── Figure4_Temperature_Time_Series.png
        │
        ├── Animations
        │   ├── Animation1_Temperature_Evolution.gif
        │   ├── Animation2_Hourly_Temperature.gif
        │   │
        │   └── Freeze-Frames
        │       ├── Animation1_FreezeFrame_1.png
        │       ├── Animation1_FreezeFrame_2.png
        │       ├── Animation1_FreezeFrame_3.png
        │       ├── Animation2_FreezeFrame_1.png
        │       ├── Animation2_FreezeFrame_2.png
        │       └── Animation2_FreezeFrame_3.png

---

## How to Run

### 1. Clone the Repository
git clone https://github.com/leviaustria1-sudo/EDS_3581_Austria.git

cd EDS_3581_Austria

### 2. Install Required Libraries
pip install -r requirements.txt

### 3. Download the Dataset
Go to [Global Weather & Air Pollution Dataset](https://www.kaggle.com/datasets/xjoannax88/global-weather-and-air-pollution-dataset) and download:
- openweather_weather_airpollution_top3cities_per_country.csv

Place both files inside the data/ folder.

### 4. Update the Project Root Path
Open main.py and update the project_root variable to match your local directory:
project_root = Path("your/local/path/EDS_3581_Austria")

### 5. Run the Pipeline
python main.py

All outputs (cleaned dataset, static plots, and animated GIFs) will be automatically saved to the outputs/ folder.

---

# Research Paper

Full paper: [Austria_3851_IEEE_Paper.pdf](https://drive.google.com/file/d/1mGxIT1OuvmESyMH3XzaC-1QQedbLG3co/view?usp=sharing)

---

## Author

*Levi M. Austria* | TUPM-25-3581
<br/>
Electronics Engineering Department
<br/>
Technological University of the Philippines, Manila
<br/>
leviaustria1@gmail.com