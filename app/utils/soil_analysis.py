from typing import List, Dict, Union

from schemas import Dataset as DatasetScheme
from schemas import DatasetAnalysis

from datetime import datetime

from core import settings

import pandas as pd
import numpy as np

def calculate_soil_analysis_metrics(dataset: list[DatasetScheme]) -> DatasetAnalysis:

    df = pd.DataFrame(dataset)
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['rain'].fillna(0, inplace=True)

    # 1. Get time period
    start_date = df.index.min().isoformat()
    end_date = df.index.max().isoformat()


    # 2. Get precipitation events and high dose events (using rain > 5mm as a proxy) -> change to .env

    irrigation_events_detected = len(df[(df['rain'] > 0) & (df['rain'] < 5)])
    precipitation_events = len(df[df['rain'] > 0])
    high_dose_irrigation_events = len(df[df['rain'] >= 15])
    high_dose_irrigation_events_dates = df[df['rain'] >= 15].index.tolist()


    # 3. Calculate field capacity
    field_capacity_data = calculate_field_capacity(dataset)


    # 4. Detect over-saturation and stress days
    stress_threshold_fraction = 0.5 # change to .env
    oversaturation_report = detect_oversaturation_per_depth(dataset, field_capacity_data['field_capacity'])
    stress_report = detect_stress_days_per_depth(dataset, field_capacity_data['field_capacity'], stress_threshold_fraction)


    # 5. Process reports into final format
    saturation_dates = []
    if oversaturation_report:
        for dates_list in oversaturation_report.values():
            saturation_dates.extend(dates_list)
    unique_saturation_dates = sorted(list(set(date.isoformat() for date in saturation_dates)))
    number_of_saturation_days = len(unique_saturation_dates)

    stress_dates = []
    if stress_report:
        for dates_list in stress_report.values():
            stress_dates.extend(dates_list)
    unique_stress_dates = sorted(list(set(date.isoformat() for date in stress_dates)))
    no_of_stress_days = len(unique_stress_dates)

    # 6. Prepare stress level output (FC * stress_threshold)
    stress_level = []
    for depth, fc in field_capacity_data['field_capacity']:
        stress_level.append([depth, round(fc * stress_threshold_fraction, 4)])


    result = DatasetAnalysis(
        dataset_id=dataset[0]['dataset_id'] if dataset else "unknown",
        time_period=[start_date, end_date],
        irrigation_events_detected=irrigation_events_detected,
        precipitation_events=precipitation_events,
        high_dose_irrigation_events=high_dose_irrigation_events,
        high_dose_irrigation_events_dates=[d.isoformat() for d in high_dose_irrigation_events_dates],
        field_capacity=field_capacity_data['field_capacity'],
        stress_level=stress_level,
        number_of_saturation_days=number_of_saturation_days,
        saturation_dates=unique_saturation_dates,
        no_of_stress_days=no_of_stress_days,
        stress_dates=unique_stress_dates
    )

    return result


def calculate_field_capacity(data: list[DatasetScheme], rain_threshold_mm=5, time_window_hours=24): # change to .env
    """Calculates field capacity for each depth."""
    df = pd.DataFrame(data)
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['rain'].fillna(0, inplace=True)
    soil_moisture_cols = {
        int(col.split('_')[2]): col
        for col in df.columns
        if 'soil_moisture' in col
    }
    for col in soil_moisture_cols.values():
        df[col].fillna(method='ffill', inplace=True)
    major_rain_events = df[df['rain'] >= rain_threshold_mm]
    major_rain_event_timestamps = major_rain_events.index.tolist()
    if not major_rain_event_timestamps:
        return {"field_capacity": []}
    field_capacity_candidates = {col: [] for col in soil_moisture_cols.values()}
    for event_timestamp in major_rain_event_timestamps:
        end_of_rain_candidates = df.loc[event_timestamp:][df.loc[event_timestamp:]['rain'] == 0].index
        if not end_of_rain_candidates.empty:
            end_of_rain = end_of_rain_candidates[0]
        else:
            continue
        search_start = end_of_rain
        search_end = end_of_rain + pd.Timedelta(hours=time_window_hours)
        search_period = df.loc[search_start:search_end]
        for col in soil_moisture_cols.values():
            if not search_period.empty and not search_period[col].isnull().all():
                fc_candidate = search_period[col].max()
                field_capacity_candidates[col].append(fc_candidate)
    final_field_capacity = {
        col: np.median(candidates) if candidates else None
        for col, candidates in field_capacity_candidates.items()
    }
    output_list = []
    for depth, col_name in soil_moisture_cols.items():
        if final_field_capacity[col_name] is not None:
            output_list.append([depth, round(final_field_capacity[col_name]/100, 4)])
    return {"field_capacity": output_list}


def detect_oversaturation_per_depth(data: list[DatasetScheme], field_capacity_data: List[List[Union[int, float]]]) -> Dict[str, List[datetime]]:
    """Detects over-saturation events per depth."""
    if not field_capacity_data:
        return {}
    field_capacity_map = {item[0]: item[1] for item in field_capacity_data}
    df = pd.DataFrame(data)
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    over_saturation_events = {}
    for depth, fc_value in field_capacity_map.items():
        column_name = f'soil_moisture_{depth}'
        if column_name in df.columns:
            is_oversaturated = df[column_name] > (fc_value * 100)
            oversaturated_dates = df.loc[is_oversaturated].index.tolist()
            if oversaturated_dates:
                over_saturation_events[f'depth_{depth}cm'] = oversaturated_dates
    return over_saturation_events


def detect_stress_days_per_depth(data: list[DatasetScheme], field_capacity_data: List[List[Union[int, float]]], stress_threshold_fraction=0.5) -> Dict[str, List[datetime]]:
    """Detects stress days per depth."""
    if not field_capacity_data:
        return {}
    field_capacity_map = {item[0]: item[1] for item in field_capacity_data}
    df = pd.DataFrame(data)
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    stress_day_events = {}
    for depth, fc_value in field_capacity_map.items():
        column_name = f'soil_moisture_{depth}'
        if column_name in df.columns:
            stress_threshold = fc_value * 100 * stress_threshold_fraction
            is_stressed = df[column_name] < stress_threshold
            stressed_dates = df.loc[is_stressed].index.tolist()
            if stressed_dates:
                stress_day_events[f'depth_{depth}cm'] = stressed_dates
    return stress_day_events