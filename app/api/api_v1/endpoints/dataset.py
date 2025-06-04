from typing import List

from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from api import deps
from models import User, Dataset
from schemas import Dataset as DatasetScheme
from schemas import WeightScheme
from schemas import Message
from crud import dataset as crud_dataset

from utils import calculate_soil_analysis_metrics

from utils import jsonld_get_dataset, jsonld_analyse_soil_moisture

from core.config import settings


router = APIRouter()


@router.post("/weights/", response_model=Message)
async def set_weights(
        weight_scheme: WeightScheme,
        user: User = Depends(deps.get_current_user)
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

@router.get("/weights/", response_model=WeightScheme)
async def get_weights(
        user: User = Depends(deps.get_current_user)
) -> WeightScheme:
    """
    Gets the weights for soil analysis
    """

    weights_for_response = {str(k): v for k, v in settings.GLOBAL_WEIGHTS.items()}

    response_value = WeightScheme.model_validate(weights_for_response)

    return response_value


@router.get("/")
def get_all_datasets_ids(
        db: Session = Depends(deps.get_db),
        user: User = Depends(deps.get_current_user)
) -> list[str]:
    db_ids = crud_dataset.get_all_datasets(db)
    ids = [row.dataset_id for row in db_ids.all()]
    return ids


@router.post("/")
def upload_dataset(
        dataset: list[DatasetScheme],
        db: Session = Depends(deps.get_db),
        user: User = Depends(deps.get_current_user)
):
    try:
        for data in dataset:
            crud_dataset.add_dataset(db, data) # Can be faster!
    except:
        raise HTTPException(status_code=400, detail="Could not upload dataset")

    return {"status_code": 202, "detail": "Successfully uploaded"}


@router.get("/{dataset_id}/")
async def get_dataset(
        dataset_id: str,
        db: Session = Depends(deps.get_db),
        user: User = Depends(deps.get_current_user)
):

    db_dataset = crud_dataset.get_datasets(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="No datasets with that id")

    if settings.USING_FRONTEND:

        return db_dataset
    else:

        jsonld_db_dataset = jsonld_get_dataset(db_dataset)
        return jsonld_db_dataset


@router.delete("/{dataset_id}/")
def remove_dataset(
        dataset_id: str,
        db: Session = Depends(deps.get_db),
        user: User = Depends(deps.get_current_user)
):
    try:
        deleted = crud_dataset.delete_datasets(db, dataset_id)
    except:
        raise HTTPException(status_code=400, detail="Could not delete dataset")

    if deleted == 0:
        raise HTTPException(status_code=400, detail="No dataset with given id")
    return {"status_code":201, "detail": "Successfully deleted"}


@router.get("/{dataset_id}/analysis/")
def analyse_soil_moisture(
        dataset_id: str,
        db: Session = Depends(deps.get_db),
        user: User = Depends(deps.get_current_user)
):
    dataset: list[Dataset] = crud_dataset.get_datasets(db, dataset_id)
    dataset = [DatasetScheme(**data_part.__dict__) for data_part in dataset]

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = calculate_soil_analysis_metrics(dataset)
    if settings.USING_FRONTEND:
        return result
    else:
        jsonld_analysis = jsonld_analyse_soil_moisture(result)
        return jsonld_analysis
