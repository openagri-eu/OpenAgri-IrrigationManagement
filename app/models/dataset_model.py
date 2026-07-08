import uuid

from sqlalchemy import Column, Integer, Float, Date, String
from sqlalchemy.dialects.postgresql import UUID

from db.base_class import Base


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(String)
    date = Column(Date)
    soil_moisture_10 = Column(Float)
    soil_moisture_20 = Column(Float)
    soil_moisture_30 = Column(Float)
    soil_moisture_40 = Column(Float)
    soil_moisture_50 = Column(Float)
    soil_moisture_60 = Column(Float)
    rain = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)


class SoilTypeValues(Base):
    __tablename__ = "soil_type_values"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    soil_type = Column(String, unique=True, nullable=False)

    field_capacity = Column(Float, nullable=False)
    wilting_point = Column(Float, nullable=False)
