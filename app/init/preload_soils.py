"""
Soil Type Preloading (v2 Initialization)

Preloads standard soil types from SOIL_PROPERTIES dictionary during app startup.
Enables users to select predefined soils during CSV upload without manual entry.

SOURCE:
- SOIL_PROPERTIES from app/scripts/soil_analysis_algorithm.py
- Standard Mediterranean soil types: sand, sandy_loam, loam, clay_loam, clay
- Each includes field_capacity, wilting_point, and default et0_coefficient

INTEGRATION:
- Called during FastAPI lifespan (lifespan event) in app/main.py
- Runs once on app startup
- Idempotent: safe to call multiple times (checks existence first)
- Logs preloading status and summary

CONSIDERATIONS:
1. Idempotent Design:
   - Query DB for existing Soil records first
   - Only create missing soil types
   - Safe to re-deploy or restart without duplicates

2. Data Consistency:
   - If Soil exists in DB but definition changed in code,
     should you update it? Current: no (preserve historical data)
   - Consider version field if updates needed in future

3. Deployment:
   - On app startup, automatically populates soil table if empty
   - No manual migration needed
   - Alembic migrations can reference these predefined soils

4. Extensibility:
   - Easy to add custom soils later via API
   - Standard soils provide baseline for most scenarios
"""

from typing import Dict
from sqlalchemy.orm import Session
import logging

from crud.soil import soil as soil_crud
from schemas.soil_analysis_v2 import SoilCreate

logger = logging.getLogger(__name__)


# Standard soil types extracted from soil_analysis_algorithm.py
STANDARD_SOILS = {
    "sand": {
        "description": "Sandy soil with low water retention",
        "field_capacity": 15,
        "wilting_point": 5,
    },
    "sandy_loam": {
        "description": "Sandy loam with moderate water retention",
        "field_capacity": 25,
        "wilting_point": 15,
    },
    "sandy_loam_loamy_sand": {
        "description": "Sandy loam - loamy sand transition",
        "field_capacity": 32,
        "wilting_point": 2,
    },
    "loam": {
        "description": "Loam soil, well-balanced water retention",
        "field_capacity": 35,
        "wilting_point": 15,
    },
    "clay_loam": {
        "description": "Clay loam with high water retention",
        "field_capacity": 45,
        "wilting_point": 20,
    },
    "clay": {
        "description": "Clay soil with very high water retention",
        "field_capacity": 50,
        "wilting_point": 30,
    },
}

# Default ETo coefficient (calibrated for Mediterranean summer conditions)
DEFAULT_ET0_COEFFICIENT = 0.174


def preload_soils(db: Session) -> Dict[str, any]:
    """
    Preload standard soil types into database.
    
    IDEMPOTENT: Safe to call multiple times
    - Checks which soils already exist
    - Only creates missing soils
    - Returns summary of created/skipped soils
    
    Args:
        db: Database session
        
    Returns:
        Dict with preload summary:
        {
            'total': 6,           # Total standard soil types
            'created': 4,         # Newly created
            'existing': 2,        # Already in DB
            'failed': 0,          # Failed to create
            'details': {...}      # Per-soil status
        }
        
    LOGGING:
    - INFO: Summary of preload operation
    - DEBUG: Per-soil creation details
    - ERROR: Any failures
    """
    summary = {
        'total': len(STANDARD_SOILS),
        'created': 0,
        'existing': 0,
        'failed': 0,
        'details': {}
    }
    
    logger.info(f"Starting soil preload: {len(STANDARD_SOILS)} standard types")
    
    for soil_name, soil_props in STANDARD_SOILS.items():
        try:
            # Check if soil already exists
            existing_soil = soil_crud.get_by_name(db, soil_name)
            
            if existing_soil:
                logger.debug(f"Soil '{soil_name}' already exists (id={existing_soil.id})")
                summary['existing'] += 1
                summary['details'][soil_name] = {
                    'status': 'existing',
                    'id': existing_soil.id
                }
                continue
            
            # Create new soil
            soil_create = SoilCreate(
                name=soil_name,
                description=soil_props.get('description', ''),
                field_capacity=soil_props['field_capacity'],
                wilting_point=soil_props['wilting_point'],
                et0_coefficient=DEFAULT_ET0_COEFFICIENT
            )
            
            created_soil = soil_crud.create(db, soil_create)
            
            if created_soil:
                logger.debug(f"Created soil '{soil_name}' (id={created_soil.id}, "
                           f"FC={created_soil.field_capacity}%, WP={created_soil.wilting_point}%)")
                summary['created'] += 1
                summary['details'][soil_name] = {
                    'status': 'created',
                    'id': created_soil.id,
                    'field_capacity': created_soil.field_capacity,
                    'wilting_point': created_soil.wilting_point
                }
            else:
                logger.error(f"Failed to create soil '{soil_name}'")
                summary['failed'] += 1
                summary['details'][soil_name] = {
                    'status': 'failed',
                    'error': 'Creation returned None (likely DB constraint violation)'
                }
        
        except Exception as e:
            logger.error(f"Exception preloading soil '{soil_name}': {e}", exc_info=True)
            summary['failed'] += 1
            summary['details'][soil_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Summary log
    logger.info(
        f"Soil preload complete: {summary['created']} created, "
        f"{summary['existing']} existing, {summary['failed']} failed"
    )
    
    if summary['failed'] > 0:
        logger.warning("Some soils failed to preload. Check logs for details.")
    
    return summary


async def async_preload_soils(db: Session) -> Dict[str, any]:
    """
    Async wrapper for preload_soils (for use in FastAPI lifespan).
    
    USAGE in app/main.py:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        session = SessionLocal()
        await async_preload_soils(session)
        session.close()
        
        # ... rest of startup
        yield
        
        # Shutdown
        ...
    
    Args:
        db: Database session
        
    Returns:
        Preload summary dict (same as preload_soils)
    """
    # CONSIDERATION: If preloading becomes slow (many soils, DB latency),
    # run in thread pool to avoid blocking event loop:
    # return await asyncio.get_event_loop().run_in_executor(None, preload_soils, db)
    
    return preload_soils(db)
