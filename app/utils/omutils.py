import datetime
from datetime import timezone

from api import deps

from typing import Optional

import crud

import openmeteo_requests
import requests_cache
from fastapi import Depends
from retry_requests import retry
from sqlalchemy.orm import Session

from schemas import EToResponse, Calculation, EtoCreate, Crop, KcStage
from models import CropKc


def fetch_historical_eto_for_location(
        location_id: int,
        latitude: float,
        longitude: float,
        from_date: datetime.date,
        to_date: datetime.date,
        crop: Optional[Crop] = None,
        stage: Optional[KcStage] = None,
        db: Session = Depends(deps.get_db),
) -> EToResponse:

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": from_date.strftime("%Y-%m-%d"),
        "end_date": to_date.strftime("%Y-%m-%d"),
        "daily": "et0_fao_evapotranspiration",
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    except Exception as e:
        openmeteo.session.close()
        return None

    daily = response.Daily()
    daily_et0_fao_evapotranspiration = daily.Variables(0).ValuesAsNumpy()

    start_time = daily.Time()
    interval = daily.Interval()

    eto_records = []
    calculations_list = []

    calculations = True

    kc_value = None
    if crop and stage:
        kc_row = db.query(CropKc).filter(CropKc.crop == crop.value).first()
        if kc_row is None:
            calculations = False

        if stage == KcStage.kc_init:
            kc_value = kc_row.kc_init
        elif stage == KcStage.kc_mid:
            kc_value = kc_row.kc_mid
        elif stage == KcStage.kc_end:
            kc_value = kc_row.kc_end

    # TODO: refactor if needed
    for i, eto_value in enumerate(daily_et0_fao_evapotranspiration):
        current_time = start_time + (i * interval)
        current_date = datetime.datetime.fromtimestamp(current_time, tz=timezone.utc).date()

        py_eto_val = float(eto_value)

        eto_data = EtoCreate(
            date=current_date,
            value=py_eto_val,
            location_id=location_id
        )
        eto_records.append(eto_data)

        display_val = 0.0
        if calculations:
            display_val = py_eto_val * kc_value if kc_value is not None else py_eto_val
        calculations_list.append(
            Calculation(date=current_date, value=display_val)
        )

    created_records = crud.eto.batch_create(db=db, obj_in=eto_records)

    if created_records is None:
        openmeteo.session.close()
        return None

    response_json = EToResponse(calculations=calculations_list)
    return response_json
