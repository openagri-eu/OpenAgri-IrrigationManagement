import datetime
import uuid

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from api import deps
from models import User, Dataset, SoilTypeValues
from schemas import Dataset as DatasetScheme
from schemas import WeightScheme
from schemas import Message
from schemas import IrrigationDatapoints, SoilTypeCreate, SoilTypeUpdate, SoilTypeValuesScheme
from crud import dataset as crud_dataset
from api.deps import get_jwt

from utils import calculate_soil_analysis_metrics, calculate_irrigation_datapoints

from utils import jsonld_get_dataset, jsonld_analyse_soil_moisture

from core.config import settings


router = APIRouter()


@router.post("/weights/", response_model=Message, dependencies=[Depends(deps.get_jwt)])
async def set_weights(
        weight_scheme: WeightScheme
):
    """
    Sets the weights for soil analysis.
    """

    new_weights = {
        10: weight_scheme.val_10,
        20: weight_scheme.val_20,
        30: weight_scheme.val_30,
        40: weight_scheme.val_40,
        50: weight_scheme.val_50,
        60: weight_scheme.val_60,
    }

    settings.GLOBAL_WEIGHTS.clear()
    settings.GLOBAL_WEIGHTS.update(new_weights)

    msg = Message(message="Successfully uploaded weights per depths")

    return msg

@router.get("/weights/", response_model=WeightScheme, dependencies=[Depends(deps.get_jwt)])
async def get_weights(

) -> WeightScheme:
    """
    Gets the weights for soil analysis
    """

    weights_for_response = {str(k): v for k, v in settings.GLOBAL_WEIGHTS.items()}

    response_value = WeightScheme.model_validate(weights_for_response)

    return response_value


@router.get("/", dependencies=[Depends(deps.get_jwt)])
def get_all_datasets_ids(
        db: Session = Depends(deps.get_db)
) -> list[str]:
    db_ids = crud_dataset.get_all_datasets(db)
    ids = [row.dataset_id for row in db_ids.all()]
    return ids


@router.post("/", dependencies=[Depends(deps.get_jwt)], response_model=Message)
def upload_dataset(
        dataset: list[DatasetScheme],
        db: Session = Depends(deps.get_db)
):
    try:
        for data in dataset:
            crud_dataset.add_dataset(db, data) # Can be faster!
    except:
        raise HTTPException(status_code=400, detail="Could not upload dataset")

    return Message(message="Successfully uploaded")


@router.get("/soil-types/", response_model=List[SoilTypeValuesScheme], dependencies=[Depends(deps.get_jwt)])
def get_soil_types(
        db: Session = Depends(deps.get_db)
):
    """
    Returns a list of all available soil types, including their id.
    Used to populate dropdowns in the frontend.
    """

    return db.query(SoilTypeValues).all()


@router.get("/soil-types/{soil_type_id}/", response_model=SoilTypeValuesScheme, dependencies=[Depends(deps.get_jwt)])
def get_soil_type(
        soil_type_id: uuid.UUID,
        db: Session = Depends(deps.get_db)
):
    """
    Returns a single soil type by id.
    """

    query_row = db.query(SoilTypeValues).filter(SoilTypeValues.id == soil_type_id).first()
    if query_row is None:
        raise HTTPException(status_code=404, detail=f"Soil type with id '{soil_type_id}' not found")

    return query_row


@router.post("/soil-types/", response_model=Message, dependencies=[Depends(deps.get_jwt)])
def create_soil_type(
        soil_type_in: SoilTypeCreate,
        db: Session = Depends(deps.get_db)
):
    """
    Adds a new soil type with its field capacity and wilting point values.
    Rejects the request if the soil type already exists.
    """

    exists = db.query(SoilTypeValues).filter(SoilTypeValues.soil_type == soil_type_in.soil_type).first()
    if exists:
        raise HTTPException(status_code=409, detail=f"Soil type '{soil_type_in.soil_type}' already exists")

    db_obj = SoilTypeValues(
        soil_type=soil_type_in.soil_type,
        field_capacity=soil_type_in.field_capacity,
        wilting_point=soil_type_in.wilting_point
    )
    db.add(db_obj)
    db.commit()

    return Message(message=f"Soil type '{soil_type_in.soil_type}' successfully added")


@router.put("/soil-types/{soil_type_id}/", response_model=Message, dependencies=[Depends(deps.get_jwt)])
def update_soil_type(
        soil_type_id: uuid.UUID,
        soil_type_in: SoilTypeUpdate,
        db: Session = Depends(deps.get_db)
):
    """
    Updates a soil type's name and/or field capacity/wilting point values.
    All fields are optional - only the provided ones are changed.
    """

    query_row = db.query(SoilTypeValues).filter(SoilTypeValues.id == soil_type_id).first()
    if query_row is None:
        raise HTTPException(status_code=404, detail=f"Soil type with id '{soil_type_id}' not found")

    update_data = soil_type_in.model_dump(exclude_unset=True)

    new_soil_type = update_data.pop("soil_type", None)
    if new_soil_type is not None and new_soil_type != query_row.soil_type:
        exists = db.query(SoilTypeValues).filter(SoilTypeValues.soil_type == new_soil_type).first()
        if exists:
            raise HTTPException(status_code=409, detail=f"Soil type '{new_soil_type}' already exists")
        query_row.soil_type = new_soil_type

    for key, value in update_data.items():
        setattr(query_row, key, value)

    db.commit()

    return Message(message=f"Soil type '{query_row.soil_type}' successfully updated")


@router.delete("/soil-types/{soil_type_id}/", response_model=Message, dependencies=[Depends(deps.get_jwt)])
def delete_soil_type(
        soil_type_id: uuid.UUID,
        db: Session = Depends(deps.get_db)
):
    """
    Deletes a soil type by id.
    """

    query_row = db.query(SoilTypeValues).filter(SoilTypeValues.id == soil_type_id).first()
    if query_row is None:
        raise HTTPException(status_code=404, detail=f"Soil type with id '{soil_type_id}' not found")

    deleted_name = query_row.soil_type
    db.delete(query_row)
    db.commit()

    return Message(message=f"Soil type '{deleted_name}' successfully deleted")


@router.get("/{dataset_id}/", dependencies=[Depends(deps.get_jwt)])
async def get_dataset(
        dataset_id: str,
        db: Session = Depends(deps.get_db),
        formatting: Literal["JSON", "JSON-LD"] = "JSON-LD"
):

    db_dataset = crud_dataset.get_datasets(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="No datasets with that id")

    if formatting == "JSON":
        return db_dataset

    return jsonld_get_dataset(db_dataset)


@router.delete("/{dataset_id}/", dependencies=[Depends(deps.get_jwt)], response_model=Message)
def remove_dataset(
        dataset_id: str,
        db: Session = Depends(deps.get_db)
):
    try:
        deleted = crud_dataset.delete_datasets(db, dataset_id)
    except:
        raise HTTPException(status_code=400, detail="Could not delete dataset")

    if deleted == 0:
        raise HTTPException(status_code=400, detail="No dataset with given id")

    return Message(message="Successfully deleted")


@router.get("/{dataset_id}/analysis/", dependencies=[Depends(deps.get_jwt)])
def analyse_soil_moisture(
        dataset_id: str,
        db: Session = Depends(deps.get_db),
        soil: Optional[uuid.UUID] = None,
        formatting: Literal["JSON", "JSON-LD"] = "JSON-LD"
):
    dataset: list[Dataset] = crud_dataset.get_datasets(db, dataset_id)
    dataset = [DatasetScheme(**data_part.__dict__) for data_part in dataset]

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    field_capacity = None
    wilting_point = None
    if soil:
        query_row = db.query(SoilTypeValues).filter(SoilTypeValues.id == soil).first()
        if query_row is None:
            raise HTTPException(status_code=404, detail="Soil type not found")

        field_capacity = query_row.field_capacity
        wilting_point = query_row.wilting_point


    result = calculate_soil_analysis_metrics(dataset, field_capacity, wilting_point)

    if formatting == "JSON":
        return result

    return jsonld_analyse_soil_moisture(result)


@router.get("/{dataset_id}/irrigation-datapoints/", dependencies=[Depends(deps.get_jwt)])
def get_irrigation_datapoints(
        dataset_id: str,
        db: Session = Depends(deps.get_db),
        soil: Optional[uuid.UUID] = None
):
    """
        Returns high dose irrigation datapoints for easier charts representation
    """
    dataset: list[Dataset] = crud_dataset.get_datasets(db, dataset_id)
    dataset = [DatasetScheme(**data_part.__dict__) for data_part in dataset]

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    field_capacity = None
    wilting_point = None
    if soil:
        query_row = db.query(SoilTypeValues).filter(SoilTypeValues.id == soil).first()
        if query_row is None:
            raise HTTPException(status_code=404, detail="Soil type not found")

        field_capacity = query_row.field_capacity
        wilting_point = query_row.wilting_point

    result = calculate_irrigation_datapoints(dataset, field_capacity, wilting_point)

    return result


@router.get("/soil-moisture/{parcel_id}/from/{from_date}/to/{to_date}")
def get_soil_moisture(
        parcel_id: str,
        from_date: datetime.date,
        to_date: datetime.date,
        access_token: str = Depends(get_jwt),
):
    """
        Returns requested soil moisture analysis based on FC farm parcel and date interval
    """
    pass
