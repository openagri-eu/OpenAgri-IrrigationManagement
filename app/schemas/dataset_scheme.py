from pydantic import BaseModel, ConfigDict, Field
from typing import List, Tuple, Dict
from datetime import datetime


class WeightScheme(BaseModel):
    weights: Dict[int, float] = Field(
        default={
            10: 0.5,
            20: 0.25,
            30: 0.15,
            40: 0.05,
            50: 0.025,
            60: 0.025
        },
        description="A dictionary mapping soil depth (in cm) to a weight."
    )


class Dataset(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset_id: str
    date: datetime
    soil_moisture_10: float
    soil_moisture_20: float
    soil_moisture_30: float
    soil_moisture_40: float
    soil_moisture_50: float
    soil_moisture_60: float
    rain: float
    temperature: float
    humidity: float


class DatasetAnalysis(BaseModel):
    dataset_id: str
    time_period: List[datetime]
    irrigation_events_detected: int
    precipitation_events: int
    high_dose_irrigation_events: int
    high_dose_irrigation_events_dates: List[datetime]
    field_capacity: List[Tuple[int, float]]
    stress_level: List[Tuple[int, float]]
    number_of_saturation_days: int
    saturation_dates: List[datetime]
    no_of_stress_days: int
    stress_dates: List[datetime]
