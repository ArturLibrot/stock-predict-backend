import pandas as pd
import numpy as np
from hybrid_model import build_hybrid_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Funkcja przygotowująca dane
def prepare_data(data, sequence_length=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X.reshape((X.shape[0], X.shape[1], 1)), y, scaler

# Wczytanie danych
data = pd.read_csv("./data/AAPL.csv")  # Przykładowy plik CSV
data = data[['Close']].values

# Przygotowanie danych
sequence_length = 30
X, y, scaler = prepare_data(data, sequence_length)

# Budowa modelu
input_shape = (X.shape[1], X.shape[2])
model = build_hybrid_model(input_shape)

# Trenowanie modelu
early_stopping = EarlyStopping(monitor="loss", patience=5)
model.fit(X, y, epochs=20, batch_size=32, callbacks=[early_stopping])

# Zapis modelu
model.save("./models/hybrid_lstm_model.h5")
print("Model został zapisany w ./models/hybrid_lstm_model.h5")
