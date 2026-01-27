from unittest import TestCase
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
import os
import sys


REQUIRED_ENV_VARS = {
    "CORS_ORIGINS": '["*"]',
    "POSTGRES_USER": "mock_pg_user",
    "POSTGRES_PASSWORD": "mock_pg_pass",
    "POSTGRES_DB": "mock_db",
    "POSTGRES_HOST": "mock_host",
    "POSTGRES_PORT": "5432",
    "ACCESS_TOKEN_EXPIRATION_TIME": "3600",
    "REFRESH_TOKEN_EXPIRATION_TIME": "86400",
    "JWT_KEY": "mock_jwt_secret",
    "JWT_ALGORITHM": "HS256",
    "SERVICE_NAME": "irrigation_service",
    "SERVICE_PORT": "8000",
    "USING_GATEKEEPER": "True",
    "GATEKEEPER_USERNAME": "mock_user",
    "GATEKEEPER_PASSWORD": "mock_pass",
    "GATEKEEPER_BASE_URL": "http://mock.gatekeeper",
    "USING_FRONTEND": "False"
}


class TestIrrigationAPI(TestCase):
    CORRECT_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJJc3N1ZXIiOiJJc3N1ZXIifQ.HLkw6rgYSwcv0sE69OKiNQFvHoo-6VqlxC5nKuMmftg'
    WRONG_TOKEN = "ayJhbGciOiJIUzI1NiJ9.eyJJc3N1ZXIiOiJJc3N1ZXIifQ.HLkw6rgYSwcv0sE69OKiNQFvHoo-6VqlxC5nKuMmftg"

    BASE_URL = "/api/v1"

    @staticmethod
    def user_login(token: str = "token"):

        if token == TestIrrigationAPI.CORRECT_TOKEN:
            return {"user_id": 1, "email": "test@example.com"}
        else:
            raise HTTPException(status_code=401, detail='Not Auth!')


    def patch(self, obj, attr, value=None):
        if value is None:
            value = MagicMock()
        patcher = patch.object(obj, attr, value)
        self.addCleanup(patcher.stop)
        return patcher.start()


    def setUp(self):

        for k, v in REQUIRED_ENV_VARS.items():
            os.environ[k] = v

        current_file_path = os.path.abspath(__file__)
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        if app_directory not in sys.path:
            sys.path.append(app_directory)

        super().setUpClass()

        from main import app
        from api.deps import get_current_user

        app.dependency_overrides[get_current_user] = lambda: TestIrrigationAPI.user_login(
            TestIrrigationAPI.CORRECT_TOKEN)

        self.client = TestClient(app)


    def test_login_access_token_success(self):
        from api.api_v1.endpoints import login

        self.patch(login, "authenticate_user", MagicMock(return_value={"id": 1, "email": "test@test.com"}))
        self.patch(login, "create_access_token", MagicMock(return_value=self.CORRECT_TOKEN))

        payload = {
            "username": "test@test.com",
            "password": "password",
            "grant_type": "password"
        }

        response = self.client.post(f"{self.BASE_URL}/login/access-token/", data=payload)
        assert response.status_code == 200
        assert response.json()["access_token"] == self.CORRECT_TOKEN


    def test_get_eto_calculations(self):
        from api.api_v1.endpoints import eto

        mock_data = [{"date": "2023-01-01", "value": 5.5}]
        self.patch(eto, "get_calculations", MagicMock(return_value=mock_data))

        loc_id = 1
        f_date = "2023-01-01"
        t_date = "2023-01-05"

        response = self.client.get(
            f"{self.BASE_URL}/eto/get-calculations/{loc_id}/from/{f_date}/to/{t_date}/",
            headers={"Authorization": f"Bearer {self.CORRECT_TOKEN}"}
        )
        assert response.status_code == 200


    def test_calculate_eto_gk(self):
        from api.api_v1.endpoints import eto

        self.patch(eto, "calculate_via_gatekeeper", MagicMock(return_value={"result": "ok"}))

        params = {
            "parcel_id": "parcel_123",
            "from_date": "2023-01-01",
            "to_date": "2023-01-05"
        }

        response = self.client.get(
            f"{self.BASE_URL}/eto/calculate-gk/",
            headers={"Authorization": f"Bearer {self.CORRECT_TOKEN}"},
            params=params
        )
        assert response.status_code == 200


    def test_add_location_parcel_wkt(self):
        from api.api_v1.endpoints import location

        self.patch(location, "create_location", MagicMock(return_value={"id": 99, "message": "Created"}))

        payload = {
            "coordinates": "POLYGON ((40.2 21.2, 40.3 21.3, 40.5 25.2, 36.1 23.1, 40.2 21.2))"
        }

        response = self.client.post(
            f"{self.BASE_URL}/location/parcel-wkt/",
            headers={"Authorization": f"Bearer {self.CORRECT_TOKEN}"},
            json=payload
        )
        assert response.status_code == 200


    def test_get_all_locations(self):
        from api.api_v1.endpoints import location

        mock_locs = {"locations": [{"id": 1, "latitude": 10.0, "longitude": 20.0}]}
        self.patch(location, "get_locations_from_db", MagicMock(return_value=mock_locs))

        response = self.client.get(
            f"{self.BASE_URL}/location/",
            headers={"Authorization": f"Bearer {self.CORRECT_TOKEN}"}
        )
        assert response.status_code == 200


    def test_upload_dataset(self):
        from api.api_v1.endpoints import dataset

        self.patch(dataset, "save_dataset_data", MagicMock(return_value={"message": "Data saved"}))

        payload = [{
            "dataset_id": "ds_1",
            "date": "2023-01-01T00:00:00",
            "soil_moisture_10": 10.1,
            "soil_moisture_20": 20.2,
            "soil_moisture_30": 30.3,
            "soil_moisture_40": 40.4,
            "soil_moisture_50": 50.5,
            "soil_moisture_60": 60.6,
            "rain": 0,
            "temperature": 25,
            "humidity": 50
        }]

        response = self.client.post(
            f"{self.BASE_URL}/dataset/",
            headers={"Authorization": f"Bearer {self.CORRECT_TOKEN}"},
            json=payload
        )
        assert response.status_code == 200

""" For later
    def test_get_soil_moisture(self):
        from api.api_v1.endpoints import dataset

        mock_resp = {"analysis": "wet"}
        self.patch(dataset, "analyze_soil_moisture", MagicMock(return_value=mock_resp))

        p_id = "p_1"
        f_date = "2023-01-01"
        t_date = "2023-01-02"

        response = self.client.get(
            f"{self.BASE_URL}/dataset/soil-moisture/{p_id}/from/{f_date}/to/{t_date}",
            headers={"Authorization": f"Bearer {self.CORRECT_TOKEN}"}
        )
        assert response.status_code == 200
"""