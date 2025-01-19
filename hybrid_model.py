from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Lambda
from quantum_layer import quantum_layer

# Budowa hybrydowego modelu LSTM
def build_hybrid_model(input_shape):
    model = Sequential()

    # Warstwa LSTM
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))

    # Warstwa redukcji wymiarów
    model.add(Dense(4, activation="tanh"))

    # Warstwa kwantowa
    model.add(Lambda(quantum_layer, name="quantum_layer"))

    # Warstwa wyjściowa
    model.add(Dense(1))  # Prognoza ceny

    # Kompilacja modelu
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
