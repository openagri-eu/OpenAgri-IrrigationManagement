import datetime
import uuid

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api import deps
import crud
from api.deps import get_jwt

from schemas import EToResponse, Calculation, KcStage
from utils import jsonld_eto_response, fetch_parcel_by_id, fetch_parcel_lat_lon, fetch_farm_crop_by_id, resolve_kc_value, TimeUnit, fetch_weather_data, fetch_historical_eto_for_location

router = APIRouter()


def _resolve_kc_value(access_token: str, crop: Optional[uuid.UUID], stage: Optional[KcStage]) -> Optional[float]:
    if not crop or not stage:
        return None

    farm_crop = fetch_farm_crop_by_id(access_token=access_token, crop_id=str(crop))
    if farm_crop is None:
        raise HTTPException(404, f"No crop found in FarmCalendar with id {crop}")

    kc_value = resolve_kc_value(farm_crop, stage)
    if kc_value is None:
        raise HTTPException(404, f"No KC coefficient set for crop {crop}, stage {stage}")

    return kc_value


@router.get("/get-calculations/{location_id}/from/{from_date}/to/{to_date}/", dependencies=[Depends(get_jwt)])
def get_calculations(
    location_id: int,
    from_date: datetime.date,
    to_date: datetime.date,
    db: Session = Depends(deps.get_db),
    access_token: str = Depends(get_jwt),
    crop: Optional[uuid.UUID] = None,
    stage: Optional[KcStage] = None,
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

    kc_value = _resolve_kc_value(access_token=access_token, crop=crop, stage=stage)

    eto_response = EToResponse(
            calculations=crud.eto.get_calculations(
                db=db,
                from_date=from_date,
                to_date=to_date,
                location_id=location_id
            )
        )

    if kc_value is not None:
        calculations = eto_response.calculations

        for c in calculations:
            if c.value is not None:
                c.value = c.value * kc_value

    if formatting.lower() == "json":
        return eto_response
    else:
        jsonld_response = jsonld_eto_response(eto_response)
        return jsonld_response


@router.get("/calculate-gk/")
def calculate_eto_via_gk(
        parcel_id: str,
        from_date: datetime.date,
        to_date: datetime.date,
        access_token: str = Depends(get_jwt),
        db: Session = Depends(deps.get_db),
        crop: Optional[uuid.UUID] = None,
        stage: Optional[KcStage] = None,
        formatting: Literal["JSON", "JSON-LD"] = "JSON"
):
    """
    Returns requested ETo calculations based on FC farm parcel and date interval
    """

    if from_date > to_date:
        raise HTTPException(
            status_code=400,
            detail="from_date must be later than to_date, from_date: {} | to_date: {}".format(from_date, to_date)
        )

    parcel_fc = fetch_parcel_by_id(access_token=access_token, parcel_id=parcel_id)

    if not parcel_fc:
        raise HTTPException(
            status_code=400,
            detail="Parcel with ID:{} doesn't exist".format(parcel_id)
        )

    lat, lon = fetch_parcel_lat_lon(parcel_fc)

    weather_data = fetch_weather_data(
        latitude=lat, longitude=lon, access_token=access_token, start_date=from_date, end_date=to_date,
        variables=["et0_fao_evapotranspiration"]
    )

    if not weather_data:
        raise HTTPException(
            status_code=400,
            detail="Error during weather data fetch, none found"
        )

    kc_value = _resolve_kc_value(access_token=access_token, crop=crop, stage=stage)

    response_json = EToResponse(
        calculations=[
            Calculation(
                date=wd["date"],
                value=wd["values"]["et0_fao_evapotranspiration"]
            ) for wd in weather_data["data"]
        ]
    )

    if kc_value is not None:
        calculations = response_json.calculations

        for c in calculations:
            if c.value is not None:
                c.value = c.value * kc_value

    if formatting.lower() == "json":
        return response_json
    else:
        jsonld_response = jsonld_eto_response(response_json)
        return jsonld_response


@router.get("/calculate-coordinates/", dependencies=[Depends(get_jwt)])
def calculate_eto_by_coordinates(
        latitude: float,
        longitude: float,
        from_date: datetime.date,
        to_date: datetime.date,
        db: Session = Depends(deps.get_db),
        access_token: str = Depends(get_jwt),
        crop: Optional[uuid.UUID] = None,
        stage: Optional[KcStage] = None,
        formatting: Literal["JSON", "JSON-LD"] = "JSON"
):
    """
    Returns ETo calculations for specific coordinates on demand.
    """

    if from_date > to_date:
        raise HTTPException(
            status_code=400,
            detail=f"Error: from_date ({from_date}) cannot be later than to_date ({to_date})"
        )


    weather_data = fetch_weather_data(
        latitude=latitude,
        longitude=longitude,
        access_token=access_token,
        start_date=from_date,
        end_date=to_date,
        variables=["et0_fao_evapotranspiration"]
    )

    if not weather_data or "data" not in weather_data:
        raise HTTPException(
            status_code=404,
            detail="No weather data found for these coordinates/dates."
        )

    kc_value = _resolve_kc_value(access_token=access_token, crop=crop, stage=stage)

    calculations = []
    for wd in weather_data["data"]:
        val = wd["values"].get("et0_fao_evapotranspiration")

        if val is not None and kc_value is not None:
            val = val * kc_value

        calculations.append(Calculation(date=wd["date"], value=val))

    response_obj = EToResponse(calculations=calculations)

    if formatting.lower() == "json":
        return response_obj
    else:
        return jsonld_eto_response(response_obj)


@router.get("/fetch-and-store-eto/")
def fetch_and_store_eto(
    location_id: int,
    latitude: float,
    longitude: float,
    from_date: datetime.date,
    to_date: datetime.date,
    db: Session = Depends(deps.get_db),
    access_token: str = Depends(get_jwt),
    crop: Optional[uuid.UUID] = None,
    stage: Optional[KcStage] = None,
    formatting: Literal["JSON", "JSON-LD"] = "JSON"
):
    if from_date > to_date:
        raise HTTPException(
            status_code=400,
            detail=f"from_date must be later than to_date, from_date: {from_date} | to_date: {to_date}"
        )

    kc_value = _resolve_kc_value(access_token=access_token, crop=crop, stage=stage)

    response_json = fetch_historical_eto_for_location(
        location_id=location_id,
        latitude=latitude,
        longitude=longitude,
        from_date=from_date,
        to_date=to_date,
        db=db,
        kc_value=kc_value
    )

    if response_json is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch data from Open-Meteo or save to the database."
        )

    if formatting.lower() == "json":
        return response_json
    else:
        return jsonld_eto_response(response_json)