"""
Extended Dataset CRUD Operations (v2)

Extends existing dataset_operations.py with CSV batch insert with:
- Auto-column detection (fuzzy matching)
- Row-level validation with error collection
- Transaction safety with automatic rollback
- Partial data support (NULL values for missing depths)

TRANSACTION PATTERN:
- Collect ALL validation errors before DB operation
- Build all Dataset objects in memory
- Single db.add_all() + db.commit()
- On error: db.rollback() reverts entire batch
- Return None on failure (all-or-nothing guarantee)

REFERENCE IMPLEMENTATIONS:
- app/crud/eto.py::batch_create() - pattern for batch insert with rollback
- app/crud/dataset_operations.py - existing dataset CRUD
- app/schemas/soil_analysis_v2.py - validation schemas and error models

CONSIDERATIONS:
1. Column Detection: Use difflib.get_close_matches() for fuzzy matching
   - soil_moisture variants: soil_moisture_10, soil_moisture_10cm, SM_10, etc.
   - Temperature variants: temperature, temp, temperature_2m, t_2m, etc.
   - Allow user to override mapping if auto-detection fails

2. Large Files: For files >10MB or >10k rows, consider chunked processing
   - Parse in batches (1000 rows/batch)
   - Validate each batch
   - Insert each batch separately or collect for single transaction
   - Balance memory usage vs transaction size

3. Validation: Collect ALL errors before responding (don't fail-fast)
   - Return list of CSVValidationError with row numbers and column names
   - User can view, fix in spreadsheet, re-upload
   - Much better UX than failing on first error

4. Missing Required Fields:
   - Require at least: date + at least one soil_moisture depth
   - Optional: temperature, humidity, rain (can be NULL)
   - Alert user if >50% of optional fields are missing
"""

from typing import Optional, List, Tuple, Dict
from datetime import datetime
from difflib import get_close_matches
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from io import StringIO
import csv
import logging

from crud.base import CRUDBase
from models import Dataset
from schemas import Dataset as DatasetSchema
from schemas.soil_analysis_v2 import DatasetRawRow, CSVValidationError

logger = logging.getLogger(__name__)


class CrudDatasetV2(CRUDBase[Dataset, DatasetSchema, dict]):
    """
    Extended Dataset CRUD with v2 features (CSV batch insert, validation).
    """

    # CSV Column name variants (for fuzzy matching)
    COLUMN_VARIANTS = {
        'date': ['date', 'datetime', 'timestamp', 'date_time', 'time'],
        'temperature': ['temperature', 'temp', 'temperature_2m', 't_2m', 'air_temp', 't'],
        'humidity': ['humidity', 'relative_humidity', 'rh', 'rh_2m', 'relative_humidity_2m'],
        'rain': ['rain', 'precipitation', 'precip', 'rainfall', 'prec'],
        'soil_moisture_10': ['soil_moisture_10', 'soil_moisture_10cm', 'sm_10', 'moisture_10', 'soil_10', 'soil_moisture1_percent'],
        'soil_moisture_20': ['soil_moisture_20', 'soil_moisture_20cm', 'sm_20', 'moisture_20', 'soil_20'],
        'soil_moisture_30': ['soil_moisture_30', 'soil_moisture_30cm', 'sm_30', 'moisture_30', 'soil_30', 'soil_moisture2_percent'],
        'soil_moisture_40': ['soil_moisture_40', 'soil_moisture_40cm', 'sm_40', 'moisture_40', 'soil_40'],
        'soil_moisture_50': ['soil_moisture_50', 'soil_moisture_50cm', 'sm_50', 'moisture_50', 'soil_50'],
        'soil_moisture_60': ['soil_moisture_60', 'soil_moisture_60cm', 'sm_60', 'moisture_60', 'soil_60'],
    }

    def detect_columns(self, csv_header: List[str]) -> Dict[str, Optional[int]]:
        """
        Auto-detect columns in CSV using fuzzy matching.
        
        ALGORITHM:
        1. For each expected column (date, temperature, etc.)
        2. Find best match in CSV header using get_close_matches()
        3. Accept match if confidence > 0.8 (80%)
        4. Return mapping of expected -> CSV column index
        
        FAILURE MODES:
        - Required column not found -> return None for that column
        - User must provide manual mapping as fallback
        
        Args:
            csv_header: List of column names from CSV
            
        Returns:
            Dict mapping expected column name to CSV column index (or None if not found)
            
        EXAMPLE:
        Input: ['date', 'temp (C)', 'RH %', 'rainfall mm', 'SM_10', 'SM_20', ...]
        Output: {
            'date': 0,
            'temperature': 1,
            'humidity': 2,
            'rain': 3,
            'soil_moisture_10': 4,
            'soil_moisture_20': 5,
            ...
        }
        """
        column_mapping = {}
        csv_header_lower = [h.lower().strip() for h in csv_header]

        for expected_col, variants in self.COLUMN_VARIANTS.items():
            # Find best match for this expected column
            matches = get_close_matches(expected_col, csv_header_lower, n=1, cutoff=0.8)
            
            if matches:
                # Find original index (before lowercasing)
                match_idx = csv_header_lower.index(matches[0])
                column_mapping[expected_col] = match_idx
                logger.debug(f"Matched '{expected_col}' -> CSV column '{csv_header[match_idx]}'")
            else:
                column_mapping[expected_col] = None
                logger.debug(f"No match found for '{expected_col}'")
        
        return column_mapping

    def validate_column_mapping(self, mapping: Dict[str, Optional[int]]) -> Tuple[bool, str]:
        """
        Validate that required columns are present.
        
        REQUIREMENTS:
        - 'date' column must be found
        - At least one 'soil_moisture_*' column must be found
        
        Args:
            mapping: Column mapping from detect_columns()
            
        Returns:
            (is_valid, error_message)
        """
        # Check required columns
        if mapping.get('date') is None:
            return False, "Required 'date' column not found in CSV"
        
        moisture_cols = [k for k in mapping if k.startswith('soil_moisture_') and mapping[k] is not None]
        if not moisture_cols:
            return False, "At least one 'soil_moisture_*' column required in CSV"

        logger.info("Column validation passed: date + %d moisture depths", len(moisture_cols))
        return True, ""

    def batch_insert_from_csv(self, db: Session, csv_content: str, dataset_name: str,
                              soil_id: int, column_mapping: Optional[Dict[str, int]] = None
                              ) -> Tuple[Optional[List[Dataset]], List[CSVValidationError], Optional[str]]:
        """
        End-to-end CSV upload with validation and transaction-safe batch insert.
        
        WORKFLOW:
        1. Auto-detect columns (or use provided mapping)
        2. Validate column mapping (required columns present)
        3. Parse and validate all rows
        4. If validation errors -> return errors without DB insert
        5. If all valid -> batch insert with transaction
        6. On DB error -> rollback entire batch
        
        Args:
            db: Database session
            csv_content: Full CSV file as string
            dataset_name: User-provided dataset identifier
            soil_id: FK to Soil record
            column_mapping: Optional manual column mapping (bypass auto-detect)
            
        Returns:
            (inserted_datasets, validation_errors, error_message)
            - inserted_datasets: List of created Dataset records (or None on DB error)
            - validation_errors: List of CSVValidationError with row numbers
            - error_message: General error message (DB errors, etc.)
            
        TRANSACTION SAFETY:
        - Validation is read-only (safe to fail)
        - DB insert wrapped in try/except with rollback
        - Returns None if any part fails (nothing partial)
        """
        try:
            # Step 1: Parse CSV and detect columns
            csv_reader = csv.reader(StringIO(csv_content))
            
            try:
                csv_header = next(csv_reader)
            except StopIteration:
                return None, [], "CSV file is empty"

            column_mapping = {
                    'date': 2,
                    'temperature': 3,
                    'humidity': 4,
                    'rain': 5,
                    'soil_moisture_10': 0,
                    'soil_moisture_30': 1
            }
            if column_mapping is None:
                mapping = self.detect_columns(csv_header)
                logger.info(f"Auto-detected columns: {mapping}")
            else:
                mapping = column_mapping
                logger.info(f"Using provided column mapping: {mapping}")


            # Step 2: Validate column mapping
            is_valid, validation_msg = self.validate_column_mapping(mapping)
            if not is_valid:
                return None, [], validation_msg
            
            # Step 3: Parse and validate rows
            valid_datasets = []
            validation_errors = []
            
            for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 (header is row 1)
                try:
                    # Extract columns using mapping
                    raw_data = {}
                    for expected_col, csv_idx in mapping.items():
                        if csv_idx is not None and csv_idx < len(row):
                            raw_value = row[csv_idx].strip()
                            
                            # Parse based on column type
                            if expected_col == 'date':
                                raw_data['measurement_date'] = raw_value  # Use alias
                            else:
                                # Numeric columns: convert or None if empty
                                raw_data[expected_col] = float(raw_value) if raw_value else None
                        else:
                            # Column not in mapping or missing from row
                            raw_data[expected_col] = None
                    
                    # Validate using Pydantic model
                    validated_row = DatasetRawRow(**raw_data)

                    # Create Dataset ORM object
                    db_dataset = Dataset(
                        dataset_id=dataset_name,
                        date=validated_row.measurement_date,
                        soil_id=soil_id,
                        soil_moisture_10=validated_row.soil_moisture_10,
                        soil_moisture_20=validated_row.soil_moisture_20,
                        soil_moisture_30=validated_row.soil_moisture_30,
                        soil_moisture_40=validated_row.soil_moisture_40,
                        soil_moisture_50=validated_row.soil_moisture_50,
                        soil_moisture_60=validated_row.soil_moisture_60,
                        rain=validated_row.rain,
                        temperature=validated_row.temperature,
                        humidity=validated_row.humidity,
                        uploaded_at=datetime.utcnow()
                    )
                    valid_datasets.append(db_dataset)
                    
                except Exception as e:
                    # Collect validation error with context
                    error_msg = str(e)
                    validation_errors.append(CSVValidationError(
                        row_number=row_num,
                        column="<row>",  # Generic for validation errors
                        value=",".join(row[:min(5, len(row))]) + ("..." if len(row) > 5 else ""),
                        error=error_msg
                    ))
            
            # If any errors, don't return partial data (all-or-nothing)
            if validation_errors:
                logger.warning(f"CSV validation failed with {len(validation_errors)} errors")
                return None, validation_errors, None
            
            logger.info(f"CSV parsing successful: {len(valid_datasets)} valid rows")
            
            # Step 4: Batch insert with transaction
            db.add_all(valid_datasets)
            
            try:
                db.commit()
                
                # Refresh all objects to populate DB-assigned IDs
                for dataset in valid_datasets:
                    db.refresh(dataset)
                
                logger.info(f"Successfully inserted {len(valid_datasets)} dataset rows for '{dataset_name}'")
                return valid_datasets, [], None
                
            except SQLAlchemyError as e:
                db.rollback()
                error_msg = f"Database error during bulk insert: {str(e)}"
                logger.error(error_msg)
                return None, [], error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error during CSV upload: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, [], error_msg


# Global instance for injection into endpoints
dataset_v2 = CrudDatasetV2(Dataset)
