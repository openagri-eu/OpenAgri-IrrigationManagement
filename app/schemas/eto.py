import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enum import Enum


class KcStage(str, Enum):
    kc_init = "KC_INIT"
    kc_mid = "KC_MID"
    kc_end = "KC_END"


class Calculation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: datetime.date
    value: Optional[float]


class EToResponse(BaseModel):
    calculations: List[Calculation]



class EToInputData(BaseModel):
    t_min: float
    t_max: float
    t_mean: float
    rh_mean: float
    u_z: float
    p: float

    sea_level: int


class EtoCreate(BaseModel):
    date: datetime.date
    value: float

    location_id: int


class EtoUpdate(BaseModel):
    pass


class CropCreate(BaseModel):
    crop: str = Field(..., min_length=1, max_length=64)
    kc_init: float = Field(..., gt=0, lt=2)
    kc_mid: float = Field(..., gt=0, lt=2)
    kc_end: float = Field(..., gt=0, lt=2)

    @field_validator("crop")
    @classmethod
    def normalize_crop(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class CropKcScheme(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    crop: str
    kc_init: float
    kc_mid: float
    kc_end: float
