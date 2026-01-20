from unittest import TestCase
from unittest.mock import patch, MagicMock

from fastapi import HTTPException
import os

REQUIRED_ENV_VARS = {
    "IRRIGATION_GATEKEEPER_USERNAME": "mock_user",
    "IRRIGATION_GATEKEEPER_PASSWORD": "mock_pass",
    "IRRIGATION_BACKEND_CORS_ORIGINS": '["*"]',
    "IRRIGATION_POSTGRES_USER": "mock_pg_user",
    "IRRIGATION_POSTGRES_PASSWORD": "mock_pg_pass",
    "IRRIGATION_POSTGRES_DB": "mock_db",
    "IRRIGATION_POSTGRES_HOST": "mock_host",
    "IRRIGATION_POSTGRES_PORT": "5432",
    "IRRIGATION_SERVICE_NAME": "mock_service",
    "IRRIGATION_SERVICE_PORT": "8005",
    "IRRIGATION_GATEKEEPER_BASE_URL": "http://mock.gatekeeper",
    "JWT_ACCESS_TOKEN_EXPIRATION_TIME": "3600",
    "JWT_SIGNING_KEY": "mock_jwt_secret",
    "IRRIGATION_USING_GATEKEEPER": "True"
}

class TestIrrigationAPI(TestCase):

    CORRECT_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJJc3N1ZXIiOiJJc3N1ZXIifQ.HLkw6rgYSwcv0sE69OKiNQFvHoo-6VqlxC5nKuMmftg'
    WRONG_TOKEN = "ayJhbGciOiJIUzI1NiJ9.eyJJc3N1ZXIiOiJJc3N1ZXIifQ.HLkw6rgYSwcv0sE69OKiNQFvHoo-6VqlxC5nKuMmftg"
    BASE_URL = "/api/v1/" # Changes according to endpoints i.e., eto, location, dataset

    @staticmethod
    def user_login(token):
        if token == TestIrrigationAPI.CORRECT_TOKEN:
            return ""
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

        super().setUpClass()
        from main import app
        from api.api_v1.endpoints import user
        from fastapi.testclient import TestClient
        from api.deps import get_current_user

        app.dependency_overrides[get_current_user] = TestIrrigationAPI.user_login
        self.patch(
            user,
            "decode_jwt_token",
            MagicMock(return_value={"user_id": "123"})
        )
        self.client = TestClient(app)


    def test_get_report_endpoint_not_auth(self):
        response = self.client.get(f"{TestIrrigationAPI.BASE_URL}/123/", headers={"X-Token": "OK"},
                                   params={"token": TestIrrigationAPI.WRONG_TOKEN})
        assert response.status_code == 401

    def test_get_report_endpoint_auth_in_progress(self):
        response = self.client.get(f"{TestIrrigationAPI.BASE_URL}/123/", headers={"X-Token": "OK"},
                                   params={"token": TestIrrigationAPI.CORRECT_TOKEN})
        assert response.status_code == 202