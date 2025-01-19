from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from quantum_layer import quantum_layer
from hybrid_model import build_hybrid_model
import os

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji zmień na URL frontendu
    allow_methods=["*"],
    allow_headers=["*"],
)

# Załaduj model hybrydowy
model_path = "./models/hybrid_lstm_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={"quantum_layer": quantum_layer})
else:
    raise FileNotFoundError("Model LSTM nie został znaleziony. Uruchom `train_hybrid.py`.")

# Schemat zapytań
class PredictionRequest(BaseModel):
    ticker: str
    days_ahead: int

# Endpoint: Prognoza dla spółki
@app.post("/predict/")
async def predict_stock(data: PredictionRequest):
    try:
        ticker = data.ticker
        days_ahead = data.days_ahead

        # Wczytaj dane historyczne spółki
        csv_path = f"./data/{ticker}.csv"
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"Brak danych dla {ticker}.")

        stock_data = pd.read_csv(csv_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)

        # Przygotuj dane
        scaled_data = stock_data['Close'].values.reshape(-1, 1)
        predictions = predict_future_prices(model, scaled_data, days_ahead)
        return {"ticker": ticker, "predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Funkcja prognozowania
def predict_future_prices(model, data, days_ahead):
    sequence_length = 30
    last_sequence = data[-sequence_length:]
    predictions = []

    for _ in range(days_ahead):
        input_sequence = np.expand_dims(last_sequence, axis=0)
        predicted_price = model.predict(input_sequence)[0, 0]
        predictions.append(predicted_price)
        last_sequence = np.append(last_sequence[1:], [[predicted_price]], axis=0)

    return predictions

# Test endpoint
@app.get("/")
async def root():
    return {"message": "API działa poprawnie."}
