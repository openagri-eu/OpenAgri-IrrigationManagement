import uuid

from sqlalchemy import Column, Integer, Date, ForeignKey, Float, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base_class import Base


class CropKc(Base):
    __tablename__ = 'crop_kc'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    crop = Column(String, unique=True, nullable=False)

    kc_init = Column(Float, nullable=False)
    kc_mid = Column(Float, nullable=False)
    kc_end = Column(Float, nullable=False)


class Eto(Base):
    __tablename__ = 'eto'
    id = Column(Integer, primary_key=True, unique=True, nullable=False)

    date = Column(Date, nullable=False)
    value = Column(Float, nullable=False)

    location_id: Mapped[int] = mapped_column(ForeignKey("location.id"))
    location: Mapped["Location"] = relationship(back_populates="calculations")
