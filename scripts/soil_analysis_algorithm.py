#!/usr/bin/env python3
"""
Soil Moisture & Irrigation Analysis (Dashboard-ready export)
------------------------------------------------------------
Performs soil moisture trend analysis, identifies saturation and irrigation events,
and exports results for dashboards (ReactJS, Grafana, ThingsBoard, etc.)

Usage:
    python soil_moisture_analysis.py input.csv [--soil loam] [--plot]
"""

import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# --- Soil type defaults ---
SOIL_PROPERTIES = {
    "sand": {"FC": 15, "WP": 5},
    "sandy_loam": {"FC": 25, "WP": 15},
    "sandy_loam_loamy_sand": {"FC": 32, "WP": 2},
    "loam": {"FC": 35, "WP": 15},
    "clay_loam": {"FC": 45, "WP": 20},
    "clay": {"FC": 50, "WP": 30},
}

ET0_COEFF = 0.174  # Calibrated coefficient for ETo estimation on Mediterranean summer
JUMP_THRESHOLD = 5.0   # sudden % jump = possible irrigation/rain
RAIN_THRESHOLD = 2.0    # mm = rain event

def get_soil_params(soil_type: str):
    return SOIL_PROPERTIES.get(soil_type.lower(), SOIL_PROPERTIES["loam"])

def calculate_smi(moisture, fc, wp):
    smi = (moisture - wp) / (fc - wp)
    return np.clip(smi, 0, 1)

def estimate_eto(temp, humidity):
    """Simple temperature–humidity based evapotranspiration."""
    return ET0_COEFF * (temp + 17.8) * (1 - humidity / 100.0)

def analyze_trends(df, fc, wp):
    """Analyze soil moisture trends, irrigation needs, and saturation events."""
    # [df.rename(columns={c: 'Temperature (oC)'}) for c in df.columns if "temperature" in c]
    # [df.rename(columns={c: 'Rain'}) for c in df.columns if "precipitation" in c]
    # [df.rename(columns={c: 'Humidity (%)'}) for c in df.columns if "humidity" in c]
    # print([c for c in df.columns])
    # print(df.columns)
    moisture_cols = [c for c in df.columns if ("soil_moisture") in c]
    df["Avg_Soil_Moisture"] = df[moisture_cols].mean(axis=1)
    df["SMI"] = calculate_smi(df["Avg_Soil_Moisture"], fc, wp)
    df["ETo"] = estimate_eto(df["Temperature (oC)"], df["Humidity (%)"])
    df["ΔSMI"] = df["SMI"].diff()
    df["Water_Balance"] = df["Rain"] - df["ETo"]

    # Irrigation recommendation
    df["Irrigation_Need"] = np.where(
        (df["SMI"] < 0.4) &
        (df["Rain"].rolling(2, min_periods=1).sum() < 2) &
        (df["ETo"].rolling(3, min_periods=1).sum() > 5),
        "Irrigate",
        "OK"
    )

    # Detect saturation events
    sat_events, event_type = [], []
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i-1]
        for col in moisture_cols:
            diff = row[col] - prev[col]
            if row[col] > fc or diff > JUMP_THRESHOLD:
                if row["Rain"] > RAIN_THRESHOLD:
                    event_type.append("Rain-triggered")
                elif prev["Irrigation_Need"] == "Irrigate":
                    event_type.append("Irrigation-triggered")
                else:
                    event_type.append("Unknown")
                sat_events.append(row["Date"])
                break

    df["Saturation_Event"] = df["Date"].isin(sat_events)
    df["Saturation_Type"] = np.nan
    df.loc[df["Saturation_Event"], "Saturation_Type"] = event_type
    return df, moisture_cols

def export_dashboard_data(df, output_prefix="analysis_output"):
    """Export time-series CSV + JSON event list for dashboards."""
    df_export = df.copy()
    df_export["Date"] = df_export["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # --- Time series CSV ---
    csv_path = Path(f"{output_prefix}_timeseries.csv")
    df_export.to_csv(csv_path, index=False)
    print(f"✅ Exported time-series CSV: {csv_path}")

    # --- Events JSON ---
    events = []
    for _, row in df[df["Saturation_Event"] | (df["Irrigation_Need"] == "Irrigate")].iterrows():
        events.append({
            "datetime": row["Date"].strftime("%Y-%m-%dT%H:%M:%S"),
            "event": "Saturation" if row["Saturation_Event"] else "Irrigation",
            "type": row.get("Saturation_Type", None),
            "rain": row["Rain"],
            "smi": row["SMI"],
            "avg_moisture": row["Avg_Soil_Moisture"]
        })

    json_path = Path(f"{output_prefix}_events.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    print(f"✅ Exported event list JSON: {json_path}")

def plot_smi(df, fc, wp):
    plt.figure(figsize=(10,6))
    plt.plot(df["Date"], df["SMI"], label="SMI (Normalized Moisture)", linewidth=2)
    plt.axhline(0.3, color="red", linestyle="--", label="Dry threshold (0.3)")
    plt.axhline(0.8, color="blue", linestyle="--", label="Saturation threshold (0.8)")
    irrigate_dates = df.loc[df["Irrigation_Need"] == "Irrigate", "Date"]
    plt.scatter(irrigate_dates, df.loc[df["Irrigation_Need"] == "Irrigate", "SMI"],
                color="orange", label="Irrigation Recommended", zorder=5)
    plt.xlabel("Date")
    plt.ylabel("Soil Moisture Index (SMI)")
    plt.title("Soil Moisture Index Trend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_moisture_per_depth(df, moisture_cols, fc, wp):
    plt.figure(figsize=(12,7))
    for col in moisture_cols:
        plt.plot(df["Date"], df[col], label=col, linewidth=1.8)
    for _, r in df[df["Saturation_Event"]].iterrows():
        color = "purple"
        if r["Saturation_Type"] == "Rain-triggered":
            color = "purple"
        elif r["Saturation_Type"] == "Irrigation-triggered":
            color = "orange"
        else:
            continue
        plt.axvline(r["Date"], color=color, linestyle="--", alpha=0.4)
    rain_dates = df.loc[df["Rain"] > RAIN_THRESHOLD, "Date"]
    irr_dates = df.loc[df["Irrigation_Need"] == "Irrigate", "Date"]
    plt.scatter(rain_dates, [fc + 2]*len(rain_dates), color="blue", s=60, label="Rain > 2 mm")
    plt.scatter(irr_dates, [fc + 4]*len(irr_dates), color="orange", s=60, label="Irrigation Needed")
    plt.axhline(fc, color="gray", linestyle="--", label=f"Field Capacity ({fc}%)")
    plt.axhline(wp, color="brown", linestyle="--", label=f"Wilting Point ({wp}%)")
    plt.xlabel("Date")
    plt.ylabel("Soil Moisture (%)")
    plt.title("Soil Moisture per Depth & Detected Events")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("--soil", default="loam", help="Soil type: sand, sandy_loam, loam, clay_loam, clay")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--prefix", default="analysis_output", help="Output filename prefix")
    args = parser.parse_args()

    soil_params = get_soil_params(args.soil)
    fc, wp = soil_params["FC"], soil_params["WP"]
    print(f"Using soil type '{args.soil}' (FC={fc}%, WP={wp}%)")

    df = pd.read_csv(args.csv_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df, moisture_cols = analyze_trends(df, fc, wp)

    export_dashboard_data(df, args.prefix)

    if args.plot:
        plot_smi(df, fc, wp)
        plot_moisture_per_depth(df, moisture_cols, fc, wp)

if __name__ == "__main__":
    main()
