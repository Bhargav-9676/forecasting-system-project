
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense




def train_xgboost(X_train, y_train):

    print("Training XGBoost model...")

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    print("XGBoost training completed")

    return xgb_model



def train_arima(series):

    print("Training ARIMA model...")

    arima_model = ARIMA(
        series,
        order=(5, 1, 0)
    )

    arima_model = arima_model.fit()

    print("ARIMA training completed")

    return arima_model



def train_prophet(sales_data):

    print("Training Prophet model...")

    # Prophet requires special column names
    prophet_data = sales_data[
        ["Date", "Total"]
    ]

    prophet_data = prophet_data.rename(
        columns={
            "Date": "ds",
            "Total": "y"
        }
    )

    prophet_model = Prophet()

    prophet_model.fit(prophet_data)

    print("Prophet training completed")

    return prophet_model



def create_sequences(data, sequence_length=10):

    X = []
    y = []

    for i in range(len(data) - sequence_length):

        X.append(
            data[i:i + sequence_length]
        )

        y.append(
            data[i + sequence_length]
        )

    return np.array(X), np.array(y)



# ---------------------------------------------------
# LSTM MODEL
# ---------------------------------------------------

from sklearn.preprocessing import MinMaxScaler


def train_lstm(series):

    print("Training LSTM model...")

    # remove missing values
    series = series.dropna()

    # reshape data
    data = series.values.reshape(-1, 1)

    # scale data
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    # flatten scaled data
    scaled_data = scaled_data.flatten()

    # create sequences
    X, y = create_sequences(scaled_data)

    # reshape for LSTM
    X = X.reshape(
        (X.shape[0], X.shape[1], 1)
    )

    # build model
    lstm_model = Sequential()

    lstm_model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=(X.shape[1], 1)
        )
    )

    lstm_model.add(Dense(1))

    # compile model
    lstm_model.compile(
        optimizer="adam",
        loss="mse"
    )

    # train model
    lstm_model.fit(
        X,
        y,
        epochs=10,
        verbose=0
    )

    print("LSTM training completed")

    return lstm_model, X, y