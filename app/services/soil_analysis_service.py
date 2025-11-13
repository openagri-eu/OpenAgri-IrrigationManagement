"""
Soil Analysis Service (v2)

Modular, extendable soil analysis computation engine.
Encapsulates all algorithm logic from scripts/soil_analysis_algorithm.py
in a reusable service class.

ARCHITECTURE:
- SoilAnalysisService: Main orchestrator
- AnalysisConfig: Configurable thresholds and parameters
- SoilAnalysisResult: Data transfer object for results

EXTENSIBILITY:
- Config-driven thresholds (easy to tune)
- Input validation separate from computation
- Output structured for easy extension (e.g., add new metrics)
- Strategy pattern for irrigation need calculation
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import date
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for soil analysis algorithm."""
    
    # ETo estimation coefficient (calibrated for Mediterranean summer)
    et0_coefficient: float = 0.174
    
    # Thresholds for event detection
    jump_threshold: float = 5.0      # Sudden moisture jump (%)
    rain_threshold: float = 2.0      # Significant rain event (mm)
    
    # Thresholds for irrigation need calculation
    smi_dry_threshold: float = 0.4   # SMI below this = plant stress
    rain_lookback_days: int = 2      # Rolling window for recent rain
    eto_lookback_days: int = 3       # Rolling window for ETo demand
    eto_demand_threshold: float = 5.0  # mm, cumulative ETo demand
    
    # SMI thresholds for visualization
    smi_stress_threshold: float = 0.3     # Red zone (drought stress)
    smi_saturation_threshold: float = 0.8  # Blue zone (saturation)


@dataclass
class TimeseriesRecord:
    """Single analysis result record for a timestamp."""
    date: date
    avg_soil_moisture: float
    smi: float
    eto: float
    water_balance: float
    irrigation_need: str  # "Irrigate" or "OK"
    saturation_event: bool
    saturation_type: Optional[str] = None


@dataclass
class EventAggregation:
    """Aggregated event summary."""
    event_type: str
    count: int
    first_occurrence: Optional[date] = None
    last_occurrence: Optional[date] = None


@dataclass
class SoilAnalysisResult:
    """Complete analysis result for a dataset."""
    timeseries: List[TimeseriesRecord] = field(default_factory=list)
    events: List[EventAggregation] = field(default_factory=list)
    
    # Summary statistics
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    total_records: int = 0
    avg_smi: float = 0.0
    max_smi: float = 0.0
    min_smi: float = 0.0
    total_precipitation: float = 0.0
    total_eto: float = 0.0
    stress_days: int = 0
    saturation_days: int = 0


class SoilAnalysisService:
    """
    Main soil analysis computation engine.
    
    Processes raw sensor data (moisture, temperature, humidity, rain)
    and produces analysis results (SMI, irrigation needs, events).
    """
    
    def __init__(self, config: AnalysisConfig = None):
        """
        Initialize analysis service.
        
        Args:
            config: AnalysisConfig with tunable parameters. Defaults to standard config.
        """
        self.config = config or AnalysisConfig()
        logger.debug(f"SoilAnalysisService initialized with config: {self.config}")
    
    def calculate_smi(self, moisture: float, field_capacity: float, wilting_point: float) -> float:
        """
        Calculate Soil Moisture Index (normalized between WP and FC).
        
        SMI = (moisture - WP) / (FC - WP)
        - 0.0 = wilting point (plant stress)
        - 1.0 = field capacity (saturation)
        - Clipped to [0, 1] to handle boundary conditions
        
        Args:
            moisture: Soil moisture (%)
            field_capacity: Field capacity (%)
            wilting_point: Wilting point (%)
            
        Returns:
            SMI value in [0, 1]
        """
        if field_capacity <= wilting_point:
            logger.warning(
                f"Invalid soil parameters: FC={field_capacity} <= WP={wilting_point}. "
                f"Returning moisture as-is."
            )
            return np.clip(moisture / 100.0, 0, 1)
        
        smi = (moisture - wilting_point) / (field_capacity - wilting_point)
        return np.clip(smi, 0, 1)
    
    def estimate_eto(self, temperature: float, humidity: float) -> float:
        """
        Estimate evapotranspiration from temperature and humidity.
        
        Simple empirical formula:
        ETo = ET0_COEFF * (T + 17.8) * (1 - RH/100)
        
        Calibrated for Mediterranean summer conditions.
        
        Args:
            temperature: Air temperature (°C)
            humidity: Relative humidity (%)
            
        Returns:
            ETo (mm/day)
        """
        if pd.isna(temperature) or pd.isna(humidity):
            return 0.0
        
        eto = self.config.et0_coefficient * (temperature + 17.8) * (1 - humidity / 100.0)
        return max(0.0, eto)  # ETo cannot be negative
    
    def calculate_irrigation_need(self, row: pd.Series, config_params: Dict[str, Any]) -> str:
        """
        Determine irrigation recommendation for a record.
        
        Strategy: Irrigate if ALL conditions met:
        1. SMI < dry_threshold (soil drying)
        2. Recent rain < threshold (no natural water supply)
        3. Recent ETo > threshold (high evaporative demand)
        
        Args:
            row: DataFrame row with SMI, Rain, ETo, and rolling calculations
            config_params: Dict with rolling window calculations
            
        Returns:
            "Irrigate" or "OK"
        """
        needs_irrigation = (
            row['SMI'] < self.config.smi_dry_threshold and
            config_params['recent_rain'] < self.config.rain_threshold and
            config_params['recent_eto'] > self.config.eto_demand_threshold
        )
        return "Irrigate" if needs_irrigation else "OK"
    
    def detect_saturation_events(self, df: pd.DataFrame, moisture_cols: List[str],
                                  field_capacity: float) -> Tuple[List[date], List[str]]:
        """
        Detect saturation events (sudden moisture jumps or exceeding FC).
        
        EVENT DETECTION LOGIC:
        1. Soil moisture exceeds field capacity at any depth
        2. Sudden jump in moisture (> jump_threshold) at any depth
        
        EVENT CLASSIFICATION:
        - Rain-triggered: If rain > rain_threshold on same day
        - Irrigation-triggered: If irrigation was recommended day before
        - Unknown: Neither rain nor irrigation explains the jump
        
        Args:
            df: DataFrame with computed columns (SMI, Rain, Irrigation_Need)
            moisture_cols: List of soil moisture column names
            field_capacity: Field capacity (%)
            
        Returns:
            (sat_events, event_types) - parallel lists of dates and classifications
        """
        sat_events = []
        event_types = []
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            
            saturation_detected = False
            
            # Check each soil moisture depth
            for col in moisture_cols:
                current_moisture = row.get(col)
                prev_moisture = prev_row.get(col)
                
                if pd.isna(current_moisture):
                    continue
                
                # Condition 1: Exceeds field capacity
                if current_moisture > field_capacity:
                    saturation_detected = True
                    break
                
                # Condition 2: Sudden jump (if previous value exists)
                if not pd.isna(prev_moisture):
                    diff = current_moisture - prev_moisture
                    if diff > self.config.jump_threshold:
                        saturation_detected = True
                        break
            
            if saturation_detected:
                # Classify event
                if row.get('Rain', 0) > self.config.rain_threshold:
                    event_type = "Rain-triggered"
                elif prev_row.get('Irrigation_Need') == "Irrigate":
                    event_type = "Irrigation-triggered"
                else:
                    event_type = "Unknown"
                
                sat_events.append(row['Date'])
                event_types.append(event_type)
        
        logger.debug(f"Detected {len(sat_events)} saturation events: "
                    f"{sum(1 for et in event_types if et == 'Rain-triggered')} rain-triggered, "
                    f"{sum(1 for et in event_types if et == 'Irrigation-triggered')} irrigation-triggered")
        
        return sat_events, event_types
    
    def compute_timeseries(self, df: pd.DataFrame, field_capacity: float,
                           wilting_point: float) -> Tuple[pd.DataFrame, List[str]]:
        """
        Compute analysis metrics for each row.
        
        Columns added:
        - Avg_Soil_Moisture: Average across all depth sensors
        - SMI: Normalized soil moisture index
        - ETo: Estimated evapotranspiration
        - Water_Balance: Rain - ETo
        - Irrigation_Need: "Irrigate" or "OK"
        - Saturation_Event: Boolean
        - Saturation_Type: Event classification
        
        Args:
            df: Input DataFrame with raw data (Date, soil_moisture_*, temperature, humidity, rain)
            field_capacity: Field capacity (%)
            wilting_point: Wilting point (%)
            
        Returns:
            (df_computed, moisture_cols) - DataFrame with added columns, list of moisture column names
        """
        # Identify soil moisture columns (all columns containing "soil_moisture")
        moisture_cols = [c for c in df.columns if "soil_moisture" in c.lower()]
        
        if not moisture_cols:
            raise ValueError("No soil moisture columns found in DataFrame")
        
        logger.info(f"Found {len(moisture_cols)} soil moisture depths: {moisture_cols}")
        
        # Step 1: Average soil moisture across depths
        df["Avg_Soil_Moisture"] = df[moisture_cols].mean(axis=1)
        
        # Step 2: Calculate SMI for each row
        df["SMI"] = df["Avg_Soil_Moisture"].apply(
            lambda x: self.calculate_smi(x, field_capacity, wilting_point) if pd.notna(x) else np.nan
        )
        
        # Step 3: Estimate ETo
        df["ETo"] = df.apply(
            lambda row: self.estimate_eto(row.get('Temperature', 0), row.get('Humidity', 50)),
            axis=1
        )
        
        # Step 4: Calculate water balance
        df["Water_Balance"] = (df.get('Rain', 0) - df["ETo"]).fillna(0)
        
        # Step 5: Calculate rolling windows for irrigation need logic
        df["Rain_Rolling"] = df.get('Rain', 0).rolling(
            window=self.config.rain_lookback_days, min_periods=1
        ).sum()
        df["ETo_Rolling"] = df["ETo"].rolling(
            window=self.config.eto_lookback_days, min_periods=1
        ).sum()
        
        # Step 6: Determine irrigation needs
        df["Irrigation_Need"] = df.apply(
            lambda row: self.calculate_irrigation_need(row, {
                'recent_rain': row.get('Rain_Rolling', 0),
                'recent_eto': row.get('ETo_Rolling', 0)
            }),
            axis=1
        )
        
        # Step 7: Detect saturation events
        sat_events, event_types = self.detect_saturation_events(df, moisture_cols, field_capacity)
        df["Saturation_Event"] = df["Date"].isin(sat_events)
        
        # Map event types to rows
        event_type_map = {event_date: event_type 
                         for event_date, event_type in zip(sat_events, event_types)}
        df["Saturation_Type"] = df["Date"].map(event_type_map)
        
        # Clean up temporary rolling columns
        df.drop(columns=["Rain_Rolling", "ETo_Rolling"], inplace=True)
        
        logger.info(f"Computed timeseries: {len(df)} records with all metrics")
        
        return df, moisture_cols
    
    def aggregate_events(self, df: pd.DataFrame) -> List[EventAggregation]:
        """
        Pre-compute event aggregations from timeseries.
        
        Groups saturation and irrigation events by type and counts occurrences.
        
        Args:
            df: DataFrame with Saturation_Event, Saturation_Type, Irrigation_Need columns
            
        Returns:
            List of EventAggregation records
        """
        aggregations = []
        
        # Count saturation events by type
        saturation_events = df[df["Saturation_Event"]]
        
        if len(saturation_events) > 0:
            for event_type in saturation_events["Saturation_Type"].dropna().unique():
                events_of_type = saturation_events[saturation_events["Saturation_Type"] == event_type]
                aggregations.append(EventAggregation(
                    event_type=event_type,
                    count=len(events_of_type),
                    first_occurrence=events_of_type["Date"].min(),
                    last_occurrence=events_of_type["Date"].max()
                ))
        
        # Count irrigation needs
        irrigation_events = df[df["Irrigation_Need"] == "Irrigate"]
        if len(irrigation_events) > 0:
            aggregations.append(EventAggregation(
                event_type="Irrigation",
                count=len(irrigation_events),
                first_occurrence=irrigation_events["Date"].min(),
                last_occurrence=irrigation_events["Date"].max()
            ))
        
        logger.info(f"Aggregated {len(aggregations)} event types")
        
        return aggregations
    
    def compute_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for analysis.
        
        Args:
            df: Computed timeseries DataFrame
            
        Returns:
            Dict with aggregated statistics
        """
        stats = {
            'date_range_start': df["Date"].min(),
            'date_range_end': df["Date"].max(),
            'total_records': len(df),
            'avg_smi': df["SMI"].mean(),
            'max_smi': df["SMI"].max(),
            'min_smi': df["SMI"].min(),
            'total_precipitation': df.get('Rain', 0).sum(),
            'total_eto': df["ETo"].sum(),
            'stress_days': (df["SMI"] < self.config.smi_stress_threshold).sum(),
            'saturation_days': (df["SMI"] > self.config.smi_saturation_threshold).sum(),
        }
        
        logger.info(f"Summary stats: {stats['total_records']} records, "
                   f"avg_SMI={stats['avg_smi']:.2f}, "
                   f"rain={stats['total_precipitation']:.1f}mm, "
                   f"stress_days={stats['stress_days']}")
        
        return stats
    
    def analyze(self, df: pd.DataFrame, field_capacity: float, wilting_point: float) -> SoilAnalysisResult:
        """
        Execute complete analysis pipeline.
        
        WORKFLOW:
        1. Validate input
        2. Compute timeseries metrics
        3. Detect events
        4. Aggregate statistics
        5. Return structured result
        
        Args:
            df: Input DataFrame with raw data (must have Date column)
            field_capacity: Soil field capacity (%)
            wilting_point: Soil wilting point (%)
            
        Returns:
            SoilAnalysisResult with timeseries, events, and summary stats
            
        Raises:
            ValueError: If input is invalid
        """
        logger.info(f"Starting analysis on {len(df)} records (FC={field_capacity}%, WP={wilting_point}%)")
        
        # Validate input
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty")
        
        if field_capacity <= wilting_point:
            raise ValueError(f"Invalid soil params: FC={field_capacity} must be > WP={wilting_point}")
        
        if "Date" not in df.columns:
            raise ValueError("Input must have 'Date' column")
        
        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
        
        # Step 1: Compute timeseries
        df_computed, moisture_cols = self.compute_timeseries(df, field_capacity, wilting_point)
        
        # Step 2: Build result timeseries records
        timeseries_records = []
        for _, row in df_computed.iterrows():
            timeseries_records.append(TimeseriesRecord(
                date=row["Date"].date() if hasattr(row["Date"], 'date') else row["Date"],
                avg_soil_moisture=row.get("Avg_Soil_Moisture"),
                smi=row.get("SMI"),
                eto=row.get("ETo"),
                water_balance=row.get("Water_Balance"),
                irrigation_need=row.get("Irrigation_Need", "OK"),
                saturation_event=row.get("Saturation_Event", False),
                saturation_type=row.get("Saturation_Type")
            ))
        
        # Step 3: Aggregate events
        events = self.aggregate_events(df_computed)
        
        # Step 4: Compute summary stats
        stats = self.compute_summary_stats(df_computed)
        
        # Step 5: Build result
        result = SoilAnalysisResult(
            timeseries=timeseries_records,
            events=events,
            date_range_start=stats['date_range_start'],
            date_range_end=stats['date_range_end'],
            total_records=stats['total_records'],
            avg_smi=stats['avg_smi'],
            max_smi=stats['max_smi'],
            min_smi=stats['min_smi'],
            total_precipitation=stats['total_precipitation'],
            total_eto=stats['total_eto'],
            stress_days=stats['stress_days'],
            saturation_days=stats['saturation_days']
        )
        
        logger.info(f"Analysis complete: {len(timeseries_records)} timeseries records, "
                   f"{len(events)} event types")
        
        return result
