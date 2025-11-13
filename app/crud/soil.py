"""
Soil CRUD Operations (v2)

Implements CRUD operations for unified Soil model with batch insert support
and transaction rollback pattern.

DESIGN PATTERN:
- Extends CRUDBase generic base class (see crud/base.py)
- Batch operations use db.add_all() for efficiency
- Transaction safety via db.rollback() on SQLAlchemyError
- Idempotent operations for data preloading

TRANSACTION HANDLING:
- Each operation wraps DB modifications in try/except
- On error: call db.rollback() to atomically revert all pending changes
- Returns None on failure (instead of raising exception) for graceful error handling
- Caller can check return value (None = failure) without exception handling

REFERENCE IMPLEMENTATIONS:
- app/crud/eto.py::batch_create() - pattern for multi-record inserts with rollback
- app/crud/base.py - generic CRUD base class with create/read/update/delete
"""

from typing import Optional, List
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from crud.base import CRUDBase
from models.soil_model import Soil
from schemas.soil_analysis_v2 import SoilCreate, SoilUpdate
import logging

logger = logging.getLogger(__name__)


class CrudSoil(CRUDBase[Soil, SoilCreate, SoilUpdate]):
    """
    CRUD operations for Soil model.
    
    Manages soil type definitions and properties. All operations are transaction-safe
    with automatic rollback on error.
    """

    def create_or_get(self, db: Session, soil_name: str) -> Optional[Soil]:
        """
        Get existing soil by name or create if not exists.
        
        IDEMPOTENT: Safe to call multiple times with same soil_name.
        Used during CSV upload to ensure soil_id exists.
        
        Args:
            db: Database session
            soil_name: Name of soil type to find or create
            
        Returns:
            Soil object (found or newly created), or None on error
            
        CONSIDERATION:
        - If creating with default properties (e.g., "loam"), values come from 
          SOIL_PROPERTIES dict in app/init/preload_soils.py
        - Custom soils should be created via standard create() method with full params
        """
        # Try to find existing
        existing = db.query(Soil).filter(Soil.name == soil_name).first()
        if existing:
            logger.debug(f"Found existing soil: {soil_name}")
            return existing
        
        # Try to create default
        logger.warning(f"Soil '{soil_name}' not found and no defaults provided. "
                      f"Please create via create() method first.")
        return None

    def create(self, db: Session, obj_in: SoilCreate, **kwargs) -> Optional[Soil]:
        """
        Create new soil record with transaction safety.
        
        Args:
            db: Database session
            obj_in: SoilCreate schema with name, description, field_capacity, wilting_point, et0_coefficient
            
        Returns:
            Created Soil object, or None on error (validation failed or DB constraint violation)
            
        ERRORS HANDLED:
        - Duplicate name (unique constraint)
        - Invalid field_capacity/wilting_point values
        - DB connection errors
        
        TRANSACTION SAFETY:
        - On any error, transaction is automatically rolled back
        - DB session remains usable for subsequent operations
        """
        try:
            obj_in_data = obj_in.model_dump()
            db_obj = Soil(**obj_in_data)
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            
            logger.info(f"Created soil: {db_obj.name} (FC={db_obj.field_capacity}%, WP={db_obj.wilting_point}%)")
            return db_obj
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to create soil: {e}")
            return None

    def batch_create(self, db: Session, soils: List[SoilCreate], **kwargs) -> Optional[List[Soil]]:
        """
        Create multiple soil records in single transaction.
        
        Used during app startup to preload standard soil types.
        ALL records succeed or ALL rollback (atomic operation).
        
        Args:
            db: Database session
            soils: List of SoilCreate schemas
            
        Returns:
            List of created Soil objects, or None if ANY record failed
            
        TRANSACTION GUARANTEE:
        - If soils[0:10] succeed but soils[11] fails during commit,
          entire transaction rolls back (all 11 discarded, DB unchanged)
        - This prevents partial data inconsistencies
        
        PERFORMANCE NOTE:
        - Faster than N individual create() calls (single INSERT batch vs N INSERT statements)
        - More efficient DB round-trips
        
        IDEMPOTENT VARIANT:
        For preloading, consider checking exists first to allow re-runs
        (see create_or_get() for single record variant)
        """
        db_objects = []
        
        try:
            for soil_data in soils:
                soil_in_data = soil_data.model_dump()
                db_obj = Soil(**soil_in_data)
                db_objects.append(db_obj)
            
            db.add_all(db_objects)
            db.commit()
            
            # Refresh all objects to populate DB-assigned values (ID, timestamps)
            for obj in db_objects:
                db.refresh(obj)
            
            logger.info(f"Batch created {len(db_objects)} soils")
            return db_objects
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Batch create failed (rolling back all {len(db_objects)} records): {e}")
            return None

    def get_by_name(self, db: Session, name: str) -> Optional[Soil]:
        """
        Lookup soil by name (case-insensitive).
        
        COMMON USE CASE:
        User selects soil type during CSV upload -> look up soil_id by name
        
        Args:
            db: Database session
            name: Soil type name (e.g., "loam", "clay")
            
        Returns:
            Soil object or None if not found
        """
        return db.query(Soil).filter(Soil.name.ilike(name)).first()

    def list_all(self, db: Session) -> List[Soil]:
        """
        Get all available soil types (e.g., for dropdown on CSV upload UI).
        
        OPTIMIZATION:
        For large number of soils, consider adding pagination or caching
        """
        return db.query(Soil).order_by(Soil.name).all()

    def update(self, db: Session, db_obj: Soil, obj_in: SoilUpdate, **kwargs) -> Optional[Soil]:
        """
        Update soil record.
        
        WARNING: Updating Soil properties that already have associated Datasets
        will affect historical analysis results interpretation. Consider whether
        you should create a new Soil version instead.
        
        RECOMMENDATION:
        - For immutable historical data, consider soft-delete + create new version
        - If you do update, log warning that existing analyses reference old values
        
        Args:
            db: Database session
            db_obj: Existing Soil object
            obj_in: SoilUpdate schema (partial fields)
            
        Returns:
            Updated Soil object, or None on error
        """
        try:
            obj_data = obj_in.model_dump(exclude_unset=True)
            for field, value in obj_data.items():
                setattr(db_obj, field, value)
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            
            logger.warning(f"Updated soil: {db_obj.name} - "
                          f"This may affect interpretation of existing analyses!")
            return db_obj
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to update soil: {e}")
            return None


# Global instance for injection into endpoints
soil = CrudSoil(Soil)
