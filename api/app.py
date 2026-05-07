from fastapi import FastAPI
import pandas as pd

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


# ---------------------------------------------------
# INITIALIZE FASTAPI
# ---------------------------------------------------

app = FastAPI()


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

print("Loading dataset...")

df = pd.read_excel(
    "data/Forecasting Case- Study.xlsx"
)

# convert date column
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

# sort dataset
df = df.sort_values(
    "Date"
)

print("Dataset loaded successfully")


# ---------------------------------------------------
# HOME ENDPOINT
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Sales Forecasting API is running successfully"
    }


# ---------------------------------------------------
# COMMON FORECAST FUNCTION
# ---------------------------------------------------

def generate_forecast(state):

    # filter state data
    state_data = df[
        df["State"] == state
    ].copy()

    if state_data.empty:

        return None

    # convert sales column
    state_data["Total"] = (
        state_data["Total"]
        .astype(float)
    )

    # ------------------------------------------------
    # PROPHET FORECAST
    # ------------------------------------------------

    prophet_df = state_data[
        ["Date", "Total"]
    ].copy()

    prophet_df.columns = [
        "ds",
        "y"
    ]

    prophet_model = Prophet()

    prophet_model.fit(
        prophet_df
    )

    future = prophet_model.make_future_dataframe(
        periods=8,
        freq="W"
    )

    prophet_forecast = prophet_model.predict(
        future
    )

    prophet_values = prophet_forecast[
        "yhat"
    ].tail(8).tolist()

    # ------------------------------------------------
    # ARIMA FORECAST
    # ------------------------------------------------

    arima_model = ARIMA(
        state_data["Total"],
        order=(5, 1, 0)
    )

    arima_model = arima_model.fit()

    arima_values = arima_model.forecast(
        steps=8
    ).tolist()

    # ------------------------------------------------
    # XGBOOST FORECAST
    # ------------------------------------------------

    state_data["lag1"] = (
        state_data["Total"].shift(1)
    )

    state_data = state_data.dropna()

    X = state_data[["lag1"]]

    y = state_data["Total"]

    xgb_model = XGBRegressor()

    xgb_model.fit(X, y)

    last_value = state_data[
        "Total"
    ].iloc[-1]

    xgb_values = []

    for _ in range(8):

        pred = xgb_model.predict(
            [[last_value]]
        )[0]

        pred = float(pred)

        xgb_values.append(pred)

        last_value = pred

    # ------------------------------------------------
    # FINAL FORECAST
    # ------------------------------------------------

    forecasts = []

    previous_value = state_data[
        "Total"
    ].iloc[-1]

    for i in range(8):

        prophet_pred = max(
            prophet_values[i],
            0
        )

        arima_pred = max(
            arima_values[i],
            0
        )

        xgb_pred = max(
            xgb_values[i],
            0
        )

        # ensemble average
        final_pred = (
            prophet_pred
            +
            arima_pred
            +
            xgb_pred
        ) / 3

        # trend analysis
        if final_pred > previous_value:

            trend = "Increase"

            reason = (
                "Historical trend and "
                "seasonality indicate growth"
            )

        elif final_pred < previous_value:

            trend = "Decrease"

            reason = (
                "Historical fluctuations and "
                "recent slowdown detected"
            )

        else:

            trend = "Stable"

            reason = (
                "Sales remain stable "
                "based on previous patterns"
            )

        forecasts.append({

            "week": i + 1,

            "prophet_prediction":
                f"{int(prophet_pred):,}",

            "arima_prediction":
                f"{int(arima_pred):,}",

            "xgboost_prediction":
                f"{int(xgb_pred):,}",

            "final_prediction":
                f"{int(final_pred):,}",

            "trend": trend,

            "reason": reason

        })

        previous_value = final_pred

    return forecasts


# ---------------------------------------------------
# SIMPLE FORECAST ENDPOINT
# ---------------------------------------------------

@app.get("/forecast/simple")
def simple_forecast(state: str):

    forecasts = generate_forecast(state)

    if forecasts is None:

        return {
            "error": "State not found"
        }

    simple_output = []

    for item in forecasts:

        simple_output.append({

            "week": item["week"],

            "sales": item["final_prediction"]

        })

    return {

        "state": state,

        "forecast": simple_output

    }


# ---------------------------------------------------
# DETAILED FORECAST ENDPOINT
# ---------------------------------------------------

@app.get("/forecast/detailed")
def detailed_forecast(state: str):

    forecasts = generate_forecast(state)

    if forecasts is None:

        return {
            "error": "State not found"
        }

    return {

        "state": state,

        "forecast": forecasts

    }