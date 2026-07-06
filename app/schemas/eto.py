import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class CropUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop: Optional[str] = Field(default=None, min_length=1, max_length=64)
    kc_init: Optional[float] = Field(default=None, gt=0, lt=2)
    kc_mid: Optional[float] = Field(default=None, gt=0, lt=2)
    kc_end: Optional[float] = Field(default=None, gt=0, lt=2)

    @field_validator("crop")
    @classmethod
    def normalize_crop(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().lower().replace(" ", "_") if v is not None else v

    @model_validator(mode="after")
    def check_at_least_one_field(self) -> "CropUpdate":
        if self.crop is None and self.kc_init is None and self.kc_mid is None and self.kc_end is None:
            raise ValueError("At least one field must be provided to update")
        return self
