"""
FORECASTING MODULE
"""

import joblib
import numpy as np


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_model(path):

    model = joblib.load(path)

    return model


# ---------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------

def prepare_features(current_row):

    features = [[

        current_row["lag1"],

        current_row["lag7"],

        current_row["lag30"],

        current_row["rolling_mean7"],

        current_row["rolling_std7"],

        current_row["day_of_week"],

        current_row["month"]

    ]]

    return np.array(features)


# ---------------------------------------------------
# FORECAST FUNCTION
# ---------------------------------------------------

def forecast_next_weeks(
    model,
    last_row,
    weeks=8
):

    forecasts = []

    current_row = last_row.copy()

    previous_prediction = current_row["lag1"]

    for week in range(weeks):

        # prepare features
        features = prepare_features(
            current_row
        )

        # model prediction
        prediction = model.predict(
            features
        )[0]

        # ----------------------------------------
        # ADD SMALL TREND CHANGE
        # ----------------------------------------

        trend_factor = 1 + (0.015 * week)

        prediction = prediction * trend_factor

        # smooth prediction
        prediction = (
            prediction * 0.7
            +
            previous_prediction * 0.3
        )

        # avoid negative values
        prediction = max(
            prediction,
            0
        )

        prediction = float(prediction)

        # save prediction
        forecasts.append({

            "week": week + 1,

            "sales": f"{int(prediction):,}"

        })

        # ----------------------------------------
        # UPDATE FEATURES
        # ----------------------------------------

        current_row["lag30"] = current_row["lag7"]

        current_row["lag7"] = current_row["lag1"]

        current_row["lag1"] = prediction

        current_row["rolling_mean7"] = (
            current_row["rolling_mean7"] * 0.6
            +
            prediction * 0.4
        )

        current_row["rolling_std7"] = abs(
            prediction -
            current_row["rolling_mean7"]
        )

        # update date features
        current_row["day_of_week"] = (
            current_row["day_of_week"] + 1
        ) % 7

        current_row["month"] = (
            current_row["month"] % 12
        ) + 1

        previous_prediction = prediction

    return forecasts