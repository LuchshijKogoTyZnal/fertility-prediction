from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Загрузка модели
model = joblib.load('fertility_model.pkl')

# Определение структуры входных данных
class PatientData(BaseModel):
    season: float
    age: float
    child_diseases: int
    accident: int
    surgical_intervention: int
    high_fevers: int
    alcohol: float
    smoking: int
    hrs_sitting: float

@app.post("/predict")
def predict(data: PatientData):
    # Преобразование данных в DataFrame
    input_data = pd.DataFrame([dict(data)])
    # Предсказание
    prediction = model.predict(input_data)[0]
    return {"diagnosis": "O" if prediction == 1 else "N"}