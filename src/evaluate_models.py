
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


def evaluate_predictions(actual, predicted):

    'MAE - Mean Absolute Error'
    mae = mean_absolute_error(
        actual,
        predicted
    )

    'MSE - Mean Squared Error'
    mse = mean_squared_error(
        actual,
        predicted
    )

    'RMSE - Root Mean Squared Error'
    rmse = np.sqrt(mse)

    'MAPE - Mean Absolute Percentage Error'
    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    'R2 SCORE - Coefficient of Determination'
    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2_SCORE": round(r2, 2)
    }



def evaluate_xgboost(model, X_test, y_test):

    predictions = model.predict(X_test)

    results = evaluate_predictions(
        y_test,
        predictions
    )

    return results


def evaluate_arima(model, y_test):

    predictions = model.forecast(
        steps=len(y_test)
    )

    results = evaluate_predictions(
        y_test,
        predictions
    )

    return results




def evaluate_prophet(model, test_data):

    
    future = test_data[["Date"]].rename(
        columns={
            "Date": "ds"
        }
    )


    forecast = model.predict(future)

    predicted = forecast["yhat"]

    actual = test_data["Total"]

    results = evaluate_predictions(
        actual,
        predicted
    )

    return results


# ---------------------------------------------------
# LSTM EVALUATION
# ---------------------------------------------------

def evaluate_lstm(model, X_test, y_test):

    try:

        # generate predictions
        predictions = model.predict(X_test)

        # flatten predictions
        predictions = predictions.flatten()

        # convert to numpy arrays
        predictions = np.array(predictions)
        y_test = np.array(y_test)

        # remove NaN values
        valid_indices = (
            ~np.isnan(predictions)
            &
            ~np.isnan(y_test)
        )

        predictions = predictions[valid_indices]
        y_test = y_test[valid_indices]

        # check empty arrays
        if len(predictions) == 0:

            print("LSTM evaluation skipped due to invalid predictions")

            return {
                "MAE": float("inf"),
                "MSE": float("inf"),
                "RMSE": float("inf"),
                "MAPE": float("inf"),
                "R2_SCORE": -1
            }

        # evaluate metrics
        results = evaluate_predictions(
            y_test,
            predictions
        )

        return results

    except Exception as error:

        print("LSTM Evaluation Error:", error)

        return {
            "MAE": float("inf"),
            "MSE": float("inf"),
            "RMSE": float("inf"),
            "MAPE": float("inf"),
            "R2_SCORE": -1
        }