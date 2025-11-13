"""
Soil Analysis v2 Pydantic Schemas

Validation schemas for v2 endpoints with:
- Soil type management (SoilCreate, SoilRead)
- CSV upload with validation (DatasetUploadRequest)
- Pagination models for time-series and event queries
- Data validation for sensor inputs

DESIGN PATTERN:
- Extend Pydantic BaseModel with ConfigDict(from_attributes=True)
  to allow ORM model -> schema conversion automatically
- Validators check data ranges and business logic
- Separate request (POST body) from response schemas
- Pagination models follow REST conventions (limit, offset, total, items)

CONSIDERATIONS:
1. Validation Ranges:
   - Temperature: -40°C to +60°C (typical sensor range)
   - Humidity: 0% to 100%
   - Soil Moisture: 0% to 100%
   - Rain: >= 0 mm (no negative rain!)
   - SMI: 0.0 to 1.0 (normalized index)
   
2. Field Capacity vs Wilting Point:
   - FC must be > WP (physically impossible otherwise)
   - Typical FC ≥ 2 * WP for most soil types
   
3. Error Response Format:
   - For CSV validation, collect ALL errors before responding
   - Don't fail-fast on first error (poor UX)
   - Return line numbers and column names for context

4. Backward Compatibility:
   - Dataset schema is nullable for v1 (no soil_id required)
   - SoilAnalysisTimeseries marked as v2-only in docstring
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Literal
from datetime import date, datetime


# ============================================================================
# SOIL TYPE SCHEMAS
# ============================================================================

class SoilCreate(BaseModel):
    """
    Request to create new soil type.
    
    VALIDATION:
    - name: Unique identifier, not already in DB
    - field_capacity > wilting_point: Physical requirement
    - Both FC and WP in valid range (0-100%)
    - et0_coefficient >= 0: Must be non-negative
    """
    name: str = Field(..., min_length=1, max_length=100, description="Unique soil type name (e.g., 'loam')")
    description: Optional[str] = Field(None, max_length=500, description="Optional soil description")
    field_capacity: float = Field(..., ge=0, le=100, description="Field capacity (%)")
    wilting_point: float = Field(..., ge=0, le=100, description="Wilting point (%)")
    et0_coefficient: Optional[float] = Field(0.174, ge=0, description="ETo coefficient (default: 0.174)")
    
    @model_validator(mode='after')
    def validate_field_capacity_greater_than_wilting_point(self) -> "SoilCreate":
        """
        Ensure field_capacity > wilting_point physically.
        
        REASON:
        - At wilting point, soil water too tightly bound for plant uptake
        - At field capacity, soil holds maximum gravity-drainable water
        - Must have FC > WP for soil moisture analysis to make sense
        """
        if self.field_capacity <= self.wilting_point:
            raise ValueError(
                f"Field Capacity ({self.field_capacity}%) must be greater than "
                f"Wilting Point ({self.wilting_point}%)"
            )
        
        # RECOMMENDATION: Check reasonable ratio
        # Typical range: FC/WP >= 1.5 (some soils can have FC/WP up to 3-4)
        ratio = self.field_capacity / self.wilting_point if self.wilting_point > 0 else float('inf')
        if ratio < 1.1:
            import warnings
            warnings.warn(
                f"Unusual FC/WP ratio ({ratio:.2f}). Most soils have ratio >= 1.5. "
                f"Please verify soil parameters."
            )
        
        return self


class SoilRead(BaseModel):
    """
    Response schema for soil type (read-only).
    
    Includes timestamps and DB-generated ID.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    description: Optional[str] = None
    field_capacity: float
    wilting_point: float
    et0_coefficient: float
    created_at: datetime
    updated_at: datetime


class SoilUpdate(BaseModel):
    """
    Partial update for soil type.
    
    WARNING: Updating soil properties affects interpretation of existing analyses.
    See app/crud/soil.py::update() for considerations.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    field_capacity: Optional[float] = Field(None, ge=0, le=100)
    wilting_point: Optional[float] = Field(None, ge=0, le=100)
    et0_coefficient: Optional[float] = Field(None, ge=0)


# ============================================================================
# CSV UPLOAD & VALIDATION SCHEMAS
# ============================================================================

class CSVValidationError(BaseModel):
    """
    Single validation error from CSV row.
    
    Includes context for user to fix and re-upload.
    """
    row_number: int = Field(..., description="1-based row number in CSV (excluding header)")
    column: str = Field(..., description="Column name (e.g., 'temperature')")
    value: str = Field(..., description="Actual value in CSV")
    error: str = Field(..., description="Human-readable error message")


class DatasetRawRow(BaseModel):
    """
    Single row of raw sensor data from CSV.
    
    Validates individual sensor measurements before batch insert.
    
    VALIDATION RULES:
    - Temperature: -40°C to +60°C (typical sensor range)
    - Humidity: 0% to 100%
    - Soil Moisture: 0% to 100% (per depth)
    - Rain: >= 0 mm (no negative rainfall!)
    - Date: Valid ISO date format
    
    CONSIDERATION:
    - Allow partial data (some depths missing) for flexible sensor setups
    - Missing value: represent as None, skip in analysis
    """
    measurement_date: datetime = Field(..., description="Measurement date")

    # Weather inputs
    temperature: Optional[float] = Field(None, ge=-40, le=60, description="Air temperature (°C)")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity (%)")
    rain: Optional[float] = Field(None, ge=0, description="Precipitation (mm)")
    
    # Soil moisture at different depths (cm)
    soil_moisture_10: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture at 10cm (%)")
    soil_moisture_20: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture at 20cm (%)")
    soil_moisture_30: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture at 30cm (%)")
    soil_moisture_40: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture at 40cm (%)")
    soil_moisture_50: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture at 50cm (%)")
    soil_moisture_60: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture at 60cm (%)")
    
    @model_validator(mode='after')
    def check_at_least_one_soil_moisture(self) -> "DatasetRawRow":
        """
        Ensure at least one soil moisture depth recorded.
        
        REASON: Can't compute analysis without soil moisture data
        """
        moisture_fields = [
            self.soil_moisture_10, self.soil_moisture_20, self.soil_moisture_30,
            self.soil_moisture_40, self.soil_moisture_50, self.soil_moisture_60
        ]
        if all(m is None for m in moisture_fields):
            raise ValueError("At least one soil moisture depth (soil_moisture_*) must be provided")
        return self


class DatasetUploadRequest(BaseModel):
    """
    Request to upload CSV dataset.
    
    WORKFLOW:
    1. Client sends file + soil_name
    2. Server auto-detects columns (fuzzy matching)
    3. Server validates all rows
    4. On success: bulk insert with transaction
    5. On validation error: return error list for user to fix
    
    CONSIDERATIONS:
    - file_content: CSV bytes (passed as base64 string in JSON)
    - soil_name: Must match existing Soil.name in DB
    - dataset_name: User-provided identifier for this dataset
    """
    file_content: str = Field(..., description="CSV file content (base64 encoded for JSON transmission)")
    soil_name: str = Field(..., description="Associated soil type (must exist in DB)")
    dataset_name: str = Field(..., description="User-friendly dataset identifier")


class DatasetUploadResponse(BaseModel):
    """
    Response after CSV upload attempt.
    
    SCENARIOS:
    1. Success: dataset_id + row_count
    2. Validation errors: errors[] with row numbers and details
    """
    success: bool
    dataset_id: Optional[str] = None
    row_count: Optional[int] = None
    uploaded_at: Optional[datetime] = None
    errors: List[CSVValidationError] = Field(default_factory=list)
    error_summary: Optional[str] = None


# ============================================================================
# ANALYSIS RESULT SCHEMAS
# ============================================================================

class SoilAnalysisTimeseriesRead(BaseModel):
    """
    Single time-series record from analysis results (v2).
    
    For charting: paginated query of these records ordered by date.
    
    FIELDS:
    - SMI (Soil Moisture Index): 0.0 (wilting point) to 1.0 (field capacity)
    - Irrigation_Need: "Irrigate" or "OK"
    - Event flags: saturation_event bool + saturation_type classification
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    dataset_id: int
    date: date
    avg_soil_moisture: float = Field(..., description="Average across all depth sensors (%)")
    smi: float = Field(..., ge=0, le=1, description="Soil Moisture Index [0.0=wilting, 1.0=saturated]")
    eto: float = Field(..., ge=0, description="Evapotranspiration (mm/day)")
    water_balance: float = Field(..., description="Rain - ETo (mm/day, negative=dry)")
    irrigation_need: Literal["Irrigate", "OK"]
    saturation_event: bool
    saturation_type: Optional[str] = None  # "Rain-triggered", "Irrigation-triggered", "Unknown"
    created_at: datetime


class PaginatedTimeseriesResponse(BaseModel):
    """
    Paginated response for time-series query.
    
    USAGE:
    GET /api/v1/datasets/v2/{dataset_id}/timeseries?limit=100&offset=0&from_date=2025-01-01&to_date=2025-01-31
    
    Response includes total count (for knowing pagination bounds) + current page items.
    """
    total: int = Field(..., description="Total matching records (across all pages)")
    limit: int = Field(..., description="Records per page")
    offset: int = Field(..., description="Starting record index")
    items: List[SoilAnalysisTimeseriesRead]


class SoilAnalysisEventRead(BaseModel):
    """
    Pre-aggregated event summary (v2).
    
    Fast queries without scanning all timeseries.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    dataset_id: int
    event_type: str  # "Rain-triggered", "Irrigation-triggered", etc.
    count: int
    first_occurrence: Optional[date] = None
    last_occurrence: Optional[date] = None
    computed_at: datetime


class PaginatedEventsResponse(BaseModel):
    """
    Paginated response for event queries.
    """
    total: int
    items: List[SoilAnalysisEventRead]


class SoilAnalysisSummary(BaseModel):
    """
    Summary statistics for a completed analysis.
    
    Used by dashboard to show overall dataset statistics.
    """
    dataset_id: int
    dataset_name: str
    soil_name: str
    total_records: int
    date_range_start: date
    date_range_end: date
    avg_smi: float = Field(..., description="Average SMI across analysis period")
    max_smi: float = Field(..., description="Peak SMI (saturation)")
    min_smi: float = Field(..., description="Lowest SMI (stress)")
    irrigation_events_count: int
    saturation_events_count: int
    total_precipitation: float = Field(..., description="Total rain in period (mm)")
    total_eto: float = Field(..., description="Total ETo in period (mm)")
    analysis_completed_at: datetime
