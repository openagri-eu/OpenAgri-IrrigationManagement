"""
Dataset Model - v2 Extension

Extended to support soil analysis v2:
- soil_id: FK to unified Soil model (nullable for backward compatibility with v1)
- uploaded_at: Timestamp when CSV was uploaded
- analysis_status: Track analysis job status (PENDING, PROCESSING, COMPLETED, FAILED)

BACKWARD COMPATIBILITY:
- All new columns are NULLABLE to maintain v1 compatibility
- v1 endpoints work without soil_id (analysis_status defaults to NULL)
- v2 endpoints REQUIRE soil_id and will enforce analysis_status tracking

CONSIDERATIONS:
1. Data Migration: Existing v1 datasets have soil_id=NULL. Provide migration endpoint
   to backfill soil_id via UI dropdown selector.
2. CASCADE vs SET NULL: Using SET NULL on soil delete to preserve dataset history
3. Analysis Status Values:
   - PENDING: CSV uploaded, awaiting analysis execution
   - PROCESSING: Analysis job running (async)
   - COMPLETED: Analysis finished successfully, results in SoilAnalysisTimeseries
   - FAILED: Analysis failed (error logged in app logs)
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, String, ForeignKey, DateTime, Enum
from sqlalchemy.orm import relationship, Mapped
import enum

from db.base_class import Base


class AnalysisStatus(str, enum.Enum):
    """Analysis job status enumeration"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Dataset(Base):
    """
    Raw dataset records from CSV upload.
    
    Original columns store raw sensor/weather data at specific soil depths.
    New v2 columns enable soil type association and analysis tracking.
    """
    __tablename__ = "dataset"

    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Original v1 columns (sensor data from CSV)
    dataset_id = Column(String, nullable=True, index=True)  # User-provided dataset name
    date = Column(DateTime, nullable=True, index=True)
    soil_moisture_10 = Column(Float, nullable=True)
    soil_moisture_20 = Column(Float, nullable=True)
    soil_moisture_30 = Column(Float, nullable=True)
    soil_moisture_40 = Column(Float, nullable=True)
    soil_moisture_50 = Column(Float, nullable=True)
    soil_moisture_60 = Column(Float, nullable=True)
    rain = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    
    # v2 Extension columns
    # FK to Soil model - NULL for backward compatibility with v1 datasets
    # RECOMMENDATION: Add constraint check: (soil_id IS NOT NULL) OR (analysis_status IS NULL)
    # This prevents creating v2 datasets without soil association
    soil_id = Column(Integer, ForeignKey("soil.id", ondelete="SET NULL"), nullable=True)
    
    # Timestamp when CSV was uploaded
    uploaded_at = Column(DateTime, nullable=True, default=datetime.utcnow, index=True)
    
    # Analysis job status (see AnalysisStatus enum)
    # NULL = not yet analyzed (v1 compatibility)
    # Used to track async analysis job progress
    analysis_status = Column(Enum(AnalysisStatus), nullable=True, default=AnalysisStatus.PENDING)
    
    # Relationship to Soil model
    # CONSIDERATION: Use lazy='joined' if N+1 queries expected; default 'select' is fine for most cases
    soil: Mapped["Soil"] = relationship(
        "Soil",
        back_populates="datasets",
        foreign_keys=[soil_id]
    )
    
    def __repr__(self) -> str:
        return f"Dataset(id={self.id}, dataset_id={self.dataset_id}, date={self.date}, soil_id={self.soil_id})"
