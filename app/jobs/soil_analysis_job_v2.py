"""
Soil Analysis Background Job (v2)

Async job for executing soil analysis algorithm on uploaded datasets.
Orchestrates data retrieval, analysis service execution, and result storage.

WORKFLOW:
1. Retrieve Dataset records by dataset_id
2. Get associated Soil parameters (FC, WP)
3. Execute SoilAnalysisService.analyze()
4. Store SoilAnalysisTimeseries + SoilAnalysisEvent records
5. Update Dataset.analysis_status
6. Handle errors with automatic rollback
"""

from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
import pandas as pd
import logging
from typing import Optional, Dict, Any

from models.dataset_model import Dataset, AnalysisStatus
from models.soil_analysis_model import SoilAnalysisTimeseries, SoilAnalysisEvent, EventType
from services.soil_analysis_service import SoilAnalysisService, SoilAnalysisResult
from db.session import SessionLocal

logger = logging.getLogger(__name__)


def retrieve_dataset_records(db: Session, dataset_name: int) -> tuple[Optional[list], Optional[Dict], str]:
    """
    Retrieve raw dataset records and associated soil parameters.
    
    Args:
        db: Database session
        dataset_id: Primary key of dataset to analyze
        
    Returns:
        (records, soil_params, error_message)
        - records: List of Dataset ORM objects
        - soil_params: Dict with field_capacity, wilting_point
        - error_message: Error description if retrieval failed
    """
    try:
        # Query all records for this dataset ID
        records = db.query(Dataset).filter(Dataset.name == dataset_name).all()
        
        if not records:
            return None, None, f"No dataset found with name={dataset_name}"

        # Get first record to access metadata
        first_record = records[0]
        
        if not first_record.soil_id:
            return None, None, f"Dataset {dataset_name} has no associated soil_id"
        
        # Retrieve soil parameters
        soil = first_record.soil
        if not soil:
            return None, None, f"Soil record not found for soil_id={first_record.soil_id}"
        
        soil_params = {
            'field_capacity': soil.field_capacity,
            'wilting_point': soil.wilting_point,
            'soil_name': soil.name,
        }
        
        logger.info(f"Retrieved {len(records)} records for dataset_name={dataset_name} "
                   f"with soil='{soil.name}'")
        
        return records, soil_params, None
    
    except Exception as e:
        error_msg = f"Error retrieving dataset: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, None, error_msg


def build_dataframe(records: list) -> pd.DataFrame:
    """
    Convert Dataset ORM objects to pandas DataFrame.
    
    Args:
        records: List of Dataset ORM objects
        
    Returns:
        DataFrame with columns: Date, soil_moisture_10-60, temperature, humidity, rain, Dataset_ID
    """
    data = []
    for record in records:
        data.append({
            'Date': record.date,
            'soil_moisture_10': record.soil_moisture_10,
            'soil_moisture_20': record.soil_moisture_20,
            'soil_moisture_30': record.soil_moisture_30,
            'soil_moisture_40': record.soil_moisture_40,
            'soil_moisture_50': record.soil_moisture_50,
            'soil_moisture_60': record.soil_moisture_60,
            'Temperature': record.temperature,
            'Humidity': record.humidity,
            'Rain': record.rain,
            'Dataset_ID': record.id
        })
    
    df = pd.DataFrame(data)
    logger.debug(f"Built DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    
    return df


def store_timeseries_results(db: Session,
                            analysis_result: SoilAnalysisResult) -> tuple[int, Optional[str]]:
    """
    Store timeseries analysis results in SoilAnalysisTimeseries table.
    
    Args:
        db: Database session
        analysis_result: SoilAnalysisResult from service

    Returns:
        (record_count, error_message)
    """
    try:
        timeseries_records = []
        
        for ts_record in analysis_result.timeseries:
            db_record = SoilAnalysisTimeseries(
                dataset_id=ts_record.dataset_id,
                date=ts_record.date,
                avg_soil_moisture=ts_record.avg_soil_moisture,
                smi=ts_record.smi,
                eto=ts_record.eto,
                water_balance=ts_record.water_balance,
                irrigation_need=ts_record.irrigation_need,
                saturation_event=ts_record.saturation_event,
                saturation_type=ts_record.saturation_type if ts_record.saturation_type else None
            )
            timeseries_records.append(db_record)
        
        db.add_all(timeseries_records)
        db.commit()
        
        logger.info(f"Stored {len(timeseries_records)} timeseries records")
        return len(timeseries_records), None
    
    except SQLAlchemyError as e:
        db.rollback()
        error_msg = f"Failed to store timeseries records: {str(e)}"
        logger.error(error_msg)
        return 0, error_msg
    
    except Exception as e:
        db.rollback()
        error_msg = f"Unexpected error storing timeseries: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return 0, error_msg


def store_event_results(db: Session, dataset_name: str,
                       analysis_result) -> tuple[int, Optional[str]]:
    """
    Store pre-aggregated event results in SoilAnalysisEvent table.
    
    Args:
        db: Database session
        dataset_id: FK to Dataset
        analysis_result: SoilAnalysisResult from service
        
    Returns:
        (event_count, error_message)
    """
    try:
        event_records = []
        
        for event in analysis_result.events:
            db_record = SoilAnalysisEvent(
                dataset_name=dataset_name,
                event_type=event.event_type,
                count=event.count,
                first_occurrence=event.first_occurrence,
                last_occurrence=event.last_occurrence
            )
            event_records.append(db_record)
        
        db.add_all(event_records)
        db.commit()
        
        logger.info(f"Stored {len(event_records)} event aggregations")
        return len(event_records), None
    
    except SQLAlchemyError as e:
        db.rollback()
        error_msg = f"Failed to store event records: {str(e)}"
        logger.error(error_msg)
        return 0, error_msg
    
    except Exception as e:
        db.rollback()
        error_msg = f"Unexpected error storing events: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return 0, error_msg


async def analyze_dataset_job(dataset_name: str) -> Dict[str, Any]:
    """
    Main background job: orchestrate dataset analysis.
    
    EXECUTION FLOW:
    1. Retrieve dataset + soil params
    2. Convert to DataFrame
    3. Execute analysis service
    4. Store timeseries results
    5. Store event results
    6. Update dataset status
    
    Args:
        dataset_name: Dataset name to analyze
        
    Returns:
        Job result dict with success status and metrics
    """
    db = SessionLocal()
    result = {
        'success': False,
        'dataset_name': dataset_name,
        'timeseries_count': 0,
        'events_count': 0,
        'error': None,
        'completed_at': None
    }
    
    try:
        logger.info(f"Starting analysis job for dataset_name={dataset_name}")
        
        # Step 1: Retrieve dataset
        records, soil_params, error = retrieve_dataset_records(db, dataset_name)
        
        if error:
            result['error'] = error
            logger.error(error)
            return result
        
        logger.info(f"Analysis using soil '{soil_params['soil_name']}' "
                   f"(FC={soil_params['field_capacity']}%, WP={soil_params['wilting_point']}%)")
        
        # Step 2: Build DataFrame
        try:
            df = build_dataframe(records)
        except Exception as e:
            result['error'] = f"Failed to build DataFrame: {str(e)}"
            logger.error(result['error'], exc_info=True)
            return result
        
        # Step 3: Execute analysis
        try:
            analysis_service = SoilAnalysisService()
            analysis_result = analysis_service.analyze(
                df,
                soil_params['field_capacity'],
                soil_params['wilting_point']
            )
        except Exception as e:
            result['error'] = f"Analysis failed: {str(e)}"
            logger.error(result['error'], exc_info=True)
            return result
        
        # Step 4: Store timeseries results
        ts_count, ts_error = store_timeseries_results(db, analysis_result)
        
        if ts_error:
            result['error'] = ts_error
            return result
        
        result['timeseries_count'] = ts_count
        
        # Step 5: Store event results
        events_count, events_error = store_event_results(db, dataset_name, analysis_result)
        
        if events_error:
            result['error'] = events_error
            return result
        
        result['events_count'] = events_count
        
        # Step 6: Update dataset status
        try:
            for rec in records:
                rec.analysis_status = AnalysisStatus.COMPLETED
            db.commit()
            
            result['success'] = True
            result['completed_at'] = datetime.utcnow()

            logger.info(f"Analysis job complete: {ts_count} timeseries, {events_count} events")
            
        except SQLAlchemyError as e:
            db.rollback()
            result['error'] = f"Failed to update analysis status: {str(e)}"
            logger.error(result['error'])
            return result
        
        return result
    
    except Exception as e:
        logger.exception(f"Unexpected error in analysis job for dataset_id={dataset_name}")
        result['error'] = str(e)
        
        # Try to mark as FAILED
        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_name).first()
            if dataset:
                dataset.analysis_status = AnalysisStatus.FAILED
                db.commit()
        except Exception as inner_e:
            logger.error(f"Failed to mark dataset as FAILED: {inner_e}")
        
        return result
    
    finally:
        db.close()


def submit_analysis_job(dataset_name: str, scheduler) -> Optional[str]:
    """
    Submit analysis job to APScheduler.
    
    Args:
        dataset_id: Dataset to analyze
        scheduler: APScheduler AsyncIOScheduler instance
        
    Returns:
        Job ID for polling, or None if submission failed
    """
    try:
        job = scheduler.add_job(
            analyze_dataset_job,
            args=[dataset_name],
            name=f"Analyze dataset {dataset_name}"
        )
        
        logger.info(f"Submitted analysis job: job_id={job.id}, dataset_id={dataset_name}")
        return job.id
    
    except Exception as e:
        logger.error(f"Failed to submit analysis job for dataset_id={dataset_name}: {e}")
        return None
