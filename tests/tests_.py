import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))


from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_get_all_datasets():
    response = client.get("/dataset")
    assert response.status_code == 200