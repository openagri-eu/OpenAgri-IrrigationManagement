from typing import List, Dict, Union, Tuple

from schemas import Dataset as DatasetScheme
from schemas import DatasetAnalysis

from datetime import datetime

from core import settings
from core.weights import global_weights_store

import pandas as pd
import numpy as np


def preprocess_dataset(data: List[DatasetScheme]) -> pd.DataFrame:
    """Standard preprocessing: convert to DataFrame, set timestamp index, fill missing rain."""
    df = pd.DataFrame(data)
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['rain'].fillna(0, inplace=True)
    return df



def weighted_average(values: List[Tuple[int, float]], weights: Dict[int, float]) -> float:
    """Compute weighted average across depths given [[depth, value], ...]."""
    total, weight_sum = 0.0, 0.0
    for depth, val in values:
        if depth in weights and val is not None:
            total += weights[depth] * val
            weight_sum += weights[depth]
    return round(total / weight_sum, 4) if weight_sum else None



def calculate_field_capacity(df: pd.DataFrame,
                             rain_threshold_mm=settings.RAIN_THRESHOLD_MM,
                             time_window_hours=settings.FIELD_CAPACITY_WINDOW_HOURS) -> Union[float, None]:
    """Calculates weighted field capacity using rain events."""
    soil_moisture_cols = {int(col.split('_')[2]): col for col in df.columns if 'soil_moisture' in col}

    # Forward-fill missing values
    for col in soil_moisture_cols.values():
        df[col].fillna(method='ffill', inplace=True)

    # Identify major rain events
    major_rain_events = df[df['rain'] >= rain_threshold_mm]
    if major_rain_events.empty:
        return None

    field_capacity_candidates = {col: [] for col in soil_moisture_cols.values()}

    for event_timestamp in major_rain_events.index:
        end_of_rain_candidates = df.loc[event_timestamp:][df.loc[event_timestamp:]['rain'] == 0].index
        if end_of_rain_candidates.empty:
            continue
        end_of_rain = end_of_rain_candidates[0]
        search_period = df.loc[end_of_rain:end_of_rain + pd.Timedelta(hours=time_window_hours)]
        for col in soil_moisture_cols.values():
            if not search_period.empty and not search_period[col].isnull().all():
                fc_candidate = search_period[col].max()
                field_capacity_candidates[col].append(fc_candidate)

    final_field_capacity = {depth: np.median(field_capacity_candidates[col]) / 100 if field_capacity_candidates[col] else None
                            for depth, col in soil_moisture_cols.items()}

    # Weighted flattening
    fc_list = [[depth, val] for depth, val in final_field_capacity.items() if val is not None]
    return weighted_average(fc_list, global_weights_store)


def detect_weighted_moisture(df: pd.DataFrame) -> pd.Series:
    """Compute vectorized weighted soil moisture across all timestamps."""
    valid_depths = [depth for depth in global_weights_store if f"soil_moisture_{depth}" in df.columns]
    if not valid_depths:
        return pd.Series([], dtype=float)

    weights = np.array([global_weights_store[d] for d in valid_depths])
    soil_cols = [f"soil_moisture_{d}" for d in valid_depths]

    # Normalize soil moisture to fraction
    moisture_values = df[soil_cols].div(100)
    weighted_sum = moisture_values.mul(weights, axis=1).sum(axis=1)
    weighted_avg = weighted_sum / weights.sum()
    return weighted_avg


def detect_weighted_stress_days(df: pd.DataFrame, weighted_fc: float,
                                stress_threshold_fraction=settings.STRESS_THRESHOLD_FRACTION) -> List[datetime]:
    """Vectorized detection of stress days."""
    if weighted_fc is None:
        return []
    weighted_moisture = detect_weighted_moisture(df)
    stress_threshold = weighted_fc * stress_threshold_fraction
    return weighted_moisture[weighted_moisture < stress_threshold].index.tolist()


def detect_weighted_oversaturation(df: pd.DataFrame, weighted_fc: float) -> List[datetime]:
    """Vectorized detection of oversaturation days."""
    if weighted_fc is None:
        return []
    weighted_moisture = detect_weighted_moisture(df)
    return weighted_moisture[weighted_moisture > weighted_fc].index.tolist()



def calculate_soil_analysis_metrics(dataset: List[DatasetScheme]) -> DatasetAnalysis:
    df = preprocess_dataset(dataset)

    # 1. Time period
    start_date, end_date = df.index.min().isoformat(), df.index.max().isoformat()

    # 2. Irrigation/precipitation events
    irrigation_events_detected = len(df[(df['rain'] > 0) & (df['rain'] < settings.LOW_DOSE_THRESHOLD_MM)])
    precipitation_events = len(df[df['rain'] > 0])
    high_dose_irrigation_events = len(df[df['rain'] >= settings.HIGH_DOSE_THRESHOLD_MM])
    high_dose_irrigation_events_dates = df[df['rain'] >= settings.HIGH_DOSE_THRESHOLD_MM].index.tolist()

    # 3. Field capacity (weighted)
    weighted_fc = calculate_field_capacity(df)

    # 4. Stress and oversaturation detection
    stress_threshold_fraction = settings.STRESS_THRESHOLD_FRACTION
    oversaturation_dates = detect_weighted_oversaturation(df, weighted_fc)
    stress_dates = detect_weighted_stress_days(df, weighted_fc, stress_threshold_fraction)

    # 5. Format results
    return DatasetAnalysis(
        dataset_id=getattr(dataset[0], "dataset_id", "unknown") if dataset else "unknown",
        time_period=[start_date, end_date],
        irrigation_events_detected=irrigation_events_detected,
        precipitation_events=precipitation_events,
        high_dose_irrigation_events=high_dose_irrigation_events,
        high_dose_irrigation_events_dates=[d.isoformat() for d in high_dose_irrigation_events_dates],
        field_capacity=weighted_fc,
        stress_level=round(weighted_fc * stress_threshold_fraction, 4) if weighted_fc else None,
        number_of_saturation_days=len(set(d.date().isoformat() for d in oversaturation_dates)),
        saturation_dates=sorted(list(set(d.isoformat() for d in oversaturation_dates))),
        no_of_stress_days=len(set(d.date().isoformat() for d in stress_dates)),
        stress_dates=sorted(list(d.isoformat() for d in stress_dates)),
    )