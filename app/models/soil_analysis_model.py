"""
Soil Analysis Result Models (v2)

Stores computed analysis results after running soil_analysis_algorithm.py.

Two models handle different query patterns:
1. SoilAnalysisTimeseries: Complete time-series for charting (paginated)
2. SoilAnalysisEvent: Pre-aggregated events (fast lookup, event lists)

DESIGN RATIONALE:
- Pre-computing and storing results avoids expensive algorithm re-runs on each query
- Separates raw data (Dataset) from computed data for clarity and flexibility
- Event pre-aggregation enables fast event-type filtering without scanning all timeseries

CONSIDERATIONS:
1. Storage: Timeseries can be large (10k+ rows per dataset). Consider partitioning by 
   dataset_id or date range if storage becomes an issue. Index on (dataset_id, date) is critical.
2. Refresh Strategy: Results are immutable once computed. Re-analysis requires deleting old
   results first or using version numbering if you want to track multiple analyses.
3. Event Types: Align with soil_analysis_algorithm.py (Rain-triggered, Irrigation-triggered, Unknown)
4. Cascading Deletes: When Dataset deleted, should we keep results or delete them?
   Current: No cascade (preserve historical data). Consider soft-delete if audit trail needed.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, DateTime, String, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship, Mapped
import enum

from db.base_class import Base


class EventType(str, enum.Enum):
    """
    Classification of saturation/irrigation events.
    
    Matches event_type values from soil_analysis_algorithm.py::export_dashboard_data()
    """
    RAIN_TRIGGERED = "Rain-triggered"
    IRRIGATION_TRIGGERED = "Irrigation-triggered"
    UNKNOWN = "Unknown"
    SATURATION = "Saturation"
    IRRIGATION_NEEDED = "Irrigation"


class SoilAnalysisTimeseries(Base):
    """
    Per-row analysis results from soil_analysis_algorithm.py.
    
    Stores computed metrics for every timestamp in the original dataset.
    Designed for time-series visualization (e.g., React charts, Grafana).
    
    QUERY PATTERN:
    - SELECT * FROM soil_analysis_timeseries 
      WHERE dataset_id = ? AND date BETWEEN ? AND ?
      ORDER BY date ASC
      LIMIT ? OFFSET ?
    
    This is ideal for charting: retrieve date range with pagination.
    """
    __tablename__ = "soil_analysis_timeseries"

    # Primary key
    id = Column(Integer, primary_key=True)
    
    # FK to Dataset (immutable reference to source data)
    # RECOMMENDATION: Add CHECK constraint: dataset_id NOT NULL
    dataset_id = Column(Integer, ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Timestamp (copied from source Dataset.date for convenience)
    date = Column(Date, nullable=False, index=True)
    
    # ========== RAW & DERIVED METRICS ==========
    # Average soil moisture across all depths (%) - from algorithm
    avg_soil_moisture = Column(Float, nullable=False)
    
    # Soil Moisture Index (SMI): normalized between Wilting Point and Field Capacity
    # Formula: SMI = (moisture - WP) / (FC - WP), clipped to [0, 1]
    # 0.0 = wilting point (plant stress), 1.0 = field capacity (saturation)
    smi = Column(Float, nullable=False)
    
    # Evapotranspiration (mm) - estimated from temp & humidity
    eto = Column(Float, nullable=False)
    
    # Water balance (mm): Rain - ETo
    # Positive = surplus (wetter), Negative = deficit (drier)
    water_balance = Column(Float, nullable=False)
    
    # ========== IRRIGATION & EVENT FLAGS ==========
    # Irrigation recommendation: "Irrigate" or "OK"
    # Set to "Irrigate" when SMI < 0.4 AND low recent rain AND high ETo demand
    irrigation_need = Column(String(50), nullable=False)  # "Irrigate" or "OK"
    
    # Boolean: Saturation event detected at this timestamp
    # True if soil_moisture > field_capacity OR sudden jump detected
    saturation_event = Column(Boolean, nullable=False, default=False)
    
    # Event classification (if saturation_event=True)
    # CONSIDERATION: Make nullable since not all rows have events
    saturation_type = Column(Enum(EventType), nullable=True)
    
    # ========== METADATA ==========
    # Timestamp when analysis was computed
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset")
    
    def __repr__(self) -> str:
        return (f"SoilAnalysisTimeseries(dataset_id={self.dataset_id}, date={self.date}, "
                f"SMI={self.smi:.2f}, irrigation={self.irrigation_need})")


class SoilAnalysisEvent(Base):
    """
    Pre-aggregated event summary per dataset.
    
    Instead of querying all timeseries rows to count events, this table
    stores pre-computed event statistics for fast dashboard queries.
    
    QUERY PATTERN (fast):
    - SELECT * FROM soil_analysis_event 
      WHERE dataset_id = ? AND event_type IN ('Rain-triggered', 'Irrigation-triggered')
    
    vs. without pre-aggregation (slow):
    - SELECT event_type, COUNT(*) FROM soil_analysis_timeseries
      WHERE dataset_id = ? AND saturation_event = TRUE
      GROUP BY event_type
    
    CONSIDERATIONS:
    1. Consistency: Must be updated whenever SoilAnalysisTimeseries rows change
    2. Uniqueness: Composite primary key (dataset_id, event_type) ensures one entry per event type
    3. Refresh: If re-analyzing same dataset, delete old SoilAnalysisEvent first
    """
    __tablename__ = "soil_analysis_event"

    # Composite primary key: (dataset_id, event_type)
    id = Column(Integer, primary_key=True)
    
    # FK to Dataset
    dataset_id = Column(Integer, ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Event type classification
    event_type = Column(Enum(EventType), nullable=False, index=True)
    
    # ========== AGGREGATED METRICS ==========
    # Count of events of this type
    count = Column(Integer, nullable=False, default=0)
    
    # First and last occurrence dates of this event type
    first_occurrence = Column(Date, nullable=True)
    last_occurrence = Column(Date, nullable=True)
    
    # ========== METADATA ==========
    # When aggregation was computed
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset")
    
    def __repr__(self) -> str:
        return (f"SoilAnalysisEvent(dataset_id={self.dataset_id}, event_type={self.event_type}, "
                f"count={self.count}, first={self.first_occurrence}, last={self.last_occurrence})")
