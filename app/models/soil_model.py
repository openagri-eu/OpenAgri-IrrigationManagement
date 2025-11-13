"""
Unified Soil Model (v2)

Merges SoilType and SoilProperties into a single model to reduce schema complexity.
This replaces the need for separate SoilType -> SoilProperties relationship.

DESIGN NOTES:
- field_capacity: Maximum water holding capacity (%)
- wilting_point: Minimum plant-available water (%)
- et0_coefficient: Calibrated coefficient for evapotranspiration estimation (default 0.174)
- Unique constraint on 'name' prevents duplicate soil type definitions
- created_at/updated_at for audit trail and versioning

CONSIDERATIONS:
1. Custom Soils: If multi-tenant support needed, consider adding user_id + composite unique(name, user_id)
2. Soil Defaults: Pre-load standard soil types during app startup (see app/init/preload_soils.py)
3. Immutability: Once Soil is referenced by Dataset, avoid modifying its parameters to preserve 
   historical analysis results. Consider soft-delete or versioning if updates needed.
4. Validation: field_capacity must be > wilting_point; recommend FC >= 2 * WP for typical soils
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped
from typing import List

from db.base_class import Base


class Soil(Base):
    """
    Unified soil type and properties model.
    
    Consolidates soil characteristics into a single record for easier management
    and query performance (single table instead of 1:1 relationship).
    """
    __tablename__ = "soil"

    # Primary key
    id = Column(Integer, primary_key=True, unique=True, nullable=False)
    
    # Soil identification
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(String(500), nullable=True)
    
    # Soil water characteristics
    field_capacity = Column(Float, nullable=False)  # % soil water at field capacity
    wilting_point = Column(Float, nullable=False)   # % soil water at wilting point
    
    # ETo (evapotranspiration) coefficient
    # Default: 0.174 (calibrated for Mediterranean summer conditions)
    # See: app/scripts/soil_analysis_algorithm.py::ET0_COEFF
    et0_coefficient = Column(Float, nullable=False, default=0.174)
    
    # Audit timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # RECOMMENDATION: Set cascade="all, delete-orphan" ONLY if you want Datasets deleted with Soil
    # Current design uses SET NULL in Dataset FK to preserve historical data if soil is removed
    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset",
        back_populates="soil",
        foreign_keys="Dataset.soil_id"
    )
    
    # Unique constraint to prevent duplicate soil type names
    __table_args__ = (
        UniqueConstraint('name', name='uq_soil_name'),
    )

    def __repr__(self) -> str:
        return f"Soil(id={self.id}, name={self.name}, FC={self.field_capacity}%, WP={self.wilting_point}%)"
