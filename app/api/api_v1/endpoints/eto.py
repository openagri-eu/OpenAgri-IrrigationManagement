import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api import deps
import crud
from api.deps import get_jwt

from schemas import EToResponse
from utils import jsonld_eto_response


router = APIRouter()


@router.get("/get-calculations/{location_id}/from/{from_date}/to/{to_date}/", dependencies=[Depends(get_jwt)])
def get_calculations(
    location_id: int,
    from_date: datetime.date,
    to_date: datetime.date,
    db: Session = Depends(deps.get_db),
    formatting: Literal["JSON", "JSON-LD"] = "JSON"
):
    """
    Returns ETo calculations for the requested days
    """

    if from_date > to_date:
        raise HTTPException(
            status_code=400,
            detail="Error, from date can't be later than to date"
        )

    location_db = crud.location.get(db=db, id=location_id)

    if location_db is None:
        raise HTTPException(
            status_code=400,
            detail="Error, location with ID:{} does not exist.".format(location_id)
        )

    eto_response = EToResponse(
            calculations=crud.eto.get_calculations(
                db=db,
                from_date=from_date,
                to_date=to_date,
                location_id=location_id
            )
        )

    if formatting.lower() == "json":
        return eto_response
    else:
        jsonld_response = jsonld_eto_response(eto_response)
        return jsonld_response
