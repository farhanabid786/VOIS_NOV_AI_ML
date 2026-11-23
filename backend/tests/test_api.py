from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_reports_loaded_xgboost():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model"] == "XGBClassifier"

def test_prediction_contract():
    payload = {
        "age": 48, "education": 2, "sex": "F", "cigsPerDay": 0,
        "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0,
        "diabetes": 0, "totChol": 220, "sysBP": 120, "diaBP": 80,
        "BMI": 26.5, "heartRate": 75, "glucose": 85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "XGBClassifier"
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_class_1"] <= 1
    assert 0 <= body["probability_class_0"] <= 1

def test_vercel_origin_is_allowed():
    response = client.options(
        "/predict",
        headers={
            "Origin": "https://vois-nov-ai-ml.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://vois-nov-ai-ml.vercel.app"
