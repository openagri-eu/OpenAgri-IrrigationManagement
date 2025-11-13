"""
Soil Analysis v2 Endpoints

FastAPI endpoints for:
- CSV dataset upload with auto-column detection
- Background soil analysis job submission
- Paginated timeseries query
- Event aggregation query
- Summary statistics query
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
import logging

from api import deps
from crud.soil import soil as soil_crud
from crud.dataset_operations_v2 import CrudDatasetV2
from models import Dataset
from models.soil_analysis_model import SoilAnalysisTimeseries, SoilAnalysisEvent
from schemas.soil_analysis_v2 import (
    SoilCreate, SoilRead, SoilUpdate,
    DatasetUploadResponse, DatasetRawRow, CSVValidationError,
    SoilAnalysisTimeseriesRead, PaginatedTimeseriesResponse,
    SoilAnalysisEventRead, SoilAnalysisSummary
)
from schemas import Message
from jobs.soil_analysis_job_v2 import submit_analysis_job
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
crud_dataset_v2 = CrudDatasetV2(Dataset)


# ============================================================================
# SOIL MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/v2/soils/", response_model=SoilRead, dependencies=[Depends(deps.get_jwt)])
def create_soil(
    soil: SoilCreate,
    db: Session = Depends(deps.get_db)
):
    """Create a new soil type."""
    try:
        existing = soil_crud.get_by_name(db, soil.name)
        if existing:
            raise HTTPException(status_code=400, detail=f"Soil '{soil.name}' already exists")
        
        created = soil_crud.create(db, soil)
        return created
    except Exception as e:
        logger.error(f"Error creating soil: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v2/soils/", response_model=List[SoilRead], dependencies=[Depends(deps.get_jwt)])
def list_soils(
    db: Session = Depends(deps.get_db)
):
    """List all available soil types."""
    return soil_crud.list_all(db)


@router.get("/v2/soils/{soil_id}", response_model=SoilRead, dependencies=[Depends(deps.get_jwt)])
def get_soil(
    soil_id: int,
    db: Session = Depends(deps.get_db)
):
    """Get soil details by ID."""
    soil = soil_crud.get(db, soil_id)
    if not soil:
        raise HTTPException(status_code=404, detail="Soil not found")
    return soil


# ============================================================================
# DATASET UPLOAD & ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/v2/upload", response_model=DatasetUploadResponse, dependencies=[Depends(deps.get_jwt)])
async def upload_dataset(
    file: UploadFile = File(...),
    soil_id: int = Query(),
    db: Session = Depends(deps.get_db)
):
    """
    Upload CSV file with sensor data.
    
    Auto-detects columns: date, temperature, humidity, rain, soil_moisture_*
    
    Response includes:
    - dataset_id: ID for querying results
    - rows_inserted: Number of rows successfully inserted
    - validation_errors: List of row-level errors (if any)
    
    Query Parameters:
    - soil_id: Optional, associate dataset with soil type
    """
    try:
        # Validate soil_id if provided
        if soil_id:
            soil = soil_crud.get(db, soil_id)
            if not soil:
                raise HTTPException(status_code=404, detail=f"Soil ID {soil_id} not found")
        
        # Read CSV file
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Batch insert with auto-detection
        result = crud_dataset_v2.batch_insert_from_csv(
            db=db,
            csv_content=text_content,
            dataset_name=file.filename if file.filename else "uploaded_dataset",
            soil_id=soil_id
        )
        response = DatasetUploadResponse(
            success=True,
            dataset_id=file.filename,
            row_count=len(result[0]),
            errors=result[1]
        )
        if result is None:
            raise HTTPException(status_code=400, detail="CSV validation failed")

        logger.info(f"Dataset uploaded: {response.dataset_id}, {response.row_count} rows")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@router.post("/v2/{dataset_id}/analyze", response_model=Message, dependencies=[Depends(deps.get_jwt)])
async def analyze_dataset(
    dataset_id: int,
    db: Session = Depends(deps.get_db)
):
    """
    Submit dataset for soil analysis.
    
    Queues an async job that:
    1. Retrieves dataset records
    2. Computes SMI, ETo, irrigation need, events
    3. Stores timeseries and event results
    4. Updates dataset status
    
    Returns immediately with job ID. Check status via GET /datasets/v2/{dataset_id}/status
    """
    try:
        # Verify dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Submit to background job
        job_id = await submit_analysis_job(dataset_id, settings.scheduler)
        
        logger.info(f"Analysis job submitted for dataset {dataset_id}, job_id={job_id}")
        
        return Message(message=f"Analysis submitted. Job ID: {job_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting analysis job: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v2/{dataset_id}/status", dependencies=[Depends(deps.get_jwt)])
def get_dataset_status(
    dataset_id: int,
    db: Session = Depends(deps.get_db)
):
    """Get dataset analysis status: PENDING, PROCESSING, COMPLETED, FAILED."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": dataset_id,
        "status": dataset.analysis_status.value if dataset.analysis_status else "UNKNOWN"
    }


# ============================================================================
# TIMESERIES QUERY ENDPOINTS
# ============================================================================

@router.get("/v2/{dataset_id}/timeseries", response_model=PaginatedTimeseriesResponse, dependencies=[Depends(deps.get_jwt)])
def get_timeseries(
    dataset_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(deps.get_db)
):
    """
    Get paginated soil analysis timeseries.
    
    Returns computed metrics for each timestamp:
    - smi: Soil Moisture Index [0, 1]
    - eto: Evapotranspiration (mm)
    - water_balance: Rain - ETo (mm)
    - irrigation_need: "Irrigate" or "OK"
    - saturation_event: Boolean, detected saturation
    """
    try:
        # Verify dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Query timeseries
        query = db.query(SoilAnalysisTimeseries).filter(
            SoilAnalysisTimeseries.dataset_id == dataset_id
        ).order_by(SoilAnalysisTimeseries.date)
        
        total = query.count()
        items = query.limit(limit).offset(offset).all()
        
        return PaginatedTimeseriesResponse(
            total=total,
            limit=limit,
            offset=offset,
            items=[SoilAnalysisTimeseriesRead.from_orm(item) for item in items]
        )
    
    except Exception as e:
        logger.error(f"Error querying timeseries: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# EVENT QUERY ENDPOINTS
# ============================================================================

@router.get("/v2/{dataset_id}/events", response_model=List[SoilAnalysisEventRead], dependencies=[Depends(deps.get_jwt)])
def get_events(
    dataset_id: int,
    event_type: Optional[str] = Query(None),
    db: Session = Depends(deps.get_db)
):
    """
    Get aggregated saturation/irrigation events.
    
    Event types:
    - 'rain_triggered': Saturation after rain
    - 'irrigation_triggered': Saturation after irrigation
    - 'unknown': Saturation of unknown cause
    
    Returns:
    - event_type: Type of event
    - count: Number of occurrences
    - first_occurrence: Date of first event
    - last_occurrence: Date of last event
    """
    try:
        query = db.query(SoilAnalysisEvent).filter(
            SoilAnalysisEvent.dataset_id == dataset_id
        )
        
        if event_type:
            query = query.filter(SoilAnalysisEvent.event_type == event_type)
        
        events = query.all()
        
        return [SoilAnalysisEventRead.from_orm(event) for event in events]
    
    except Exception as e:
        logger.error(f"Error querying events: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# SUMMARY ENDPOINTS
# ============================================================================

@router.get("/v2/{dataset_id}/summary", response_model=SoilAnalysisSummary, dependencies=[Depends(deps.get_jwt)])
def get_summary(
    dataset_id: int,
    db: Session = Depends(deps.get_db)
):
    """
    Get analysis summary statistics.
    
    Includes:
    - Average/Max/Min SMI
    - Total precipitation, total ETo
    - Stress days (SMI < 0.3)
    - Saturation days (SMI > 0.8)
    - Event counts by type
    """
    try:
        # Query all timeseries for this dataset
        records = db.query(SoilAnalysisTimeseries).filter(
            SoilAnalysisTimeseries.dataset_id == dataset_id
        ).all()
        
        if not records:
            raise HTTPException(status_code=404, detail="No analysis results found for dataset")
        
        # Compute statistics
        smis = [r.smi for r in records if r.smi is not None]
        etos = [r.eto for r in records if r.eto is not None]
        rains = [r.water_balance for r in records if r.water_balance is not None]
        
        stress_days = sum(1 for r in records if r.smi and r.smi < 0.3)
        saturation_days = sum(1 for r in records if r.smi and r.smi > 0.8)
        irrigation_days = sum(1 for r in records if r.irrigation_need == "Irrigate")
        
        # Query events
        events = db.query(SoilAnalysisEvent).filter(
            SoilAnalysisEvent.dataset_id == dataset_id
        ).all()
        
        event_summary = {}
        for event in events:
            event_summary[event.event_type] = event.count
        
        return SoilAnalysisSummaryRead(
            dataset_id=dataset_id,
            total_records=len(records),
            avg_smi=sum(smis) / len(smis) if smis else 0.0,
            max_smi=max(smis) if smis else 0.0,
            min_smi=min(smis) if smis else 0.0,
            total_precipitation=sum(r.water_balance for r in records if r.water_balance and r.water_balance > 0) if records else 0.0,
            total_eto=sum(etos) if etos else 0.0,
            stress_days=stress_days,
            saturation_days=saturation_days,
            irrigation_days=irrigation_days,
            events=event_summary
        )
    
    except Exception as e:
        logger.error(f"Error computing summary: {e}")
        raise HTTPException(status_code=400, detail=str(e))
