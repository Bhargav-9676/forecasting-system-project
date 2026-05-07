

import os
import joblib


from src.data_preprocessing import (
    load_data,
    clean_data,
    integrate_data,
    transform_data,
    reduce_data,
    handle_outliers,
    validate_data
)


from src.feature_engineering import create_features


from src.train_models import (
    train_xgboost,
    train_arima,
    train_prophet,
    train_lstm
)


from src.evaluate_models import (
    evaluate_xgboost,
    evaluate_arima,
    evaluate_lstm,
    evaluate_prophet
)


print("\nLoading dataset...\n")

sales_data = load_data(
    "data/Forecasting Case- Study.xlsx"
)


sales_data = clean_data(sales_data)
sales_data = integrate_data(sales_data)
sales_data = transform_data(sales_data)
sales_data = reduce_data(sales_data)
sales_data = handle_outliers(sales_data)
validate_data(sales_data)



print("\nCreating time-series features...\n")

sales_data = create_features(sales_data)

print("Feature engineering completed")


print("\nSplitting dataset...\n")

train_data = sales_data[
    sales_data["Date"] < "2023-01-01"
]

test_data = sales_data[
    sales_data["Date"] >= "2023-01-01"
]



forecast_features = [
    "lag1",
    "lag7",
    "lag30",
    "rolling_mean7",
    "rolling_std7",
    "day_of_week",
    "month"
]


X_train = train_data[forecast_features]
y_train = train_data["Total"]

X_test = test_data[forecast_features]
y_test = test_data["Total"]

print("Train-test split completed")



xgb_model = train_xgboost(
    X_train,
    y_train
)

xgb_results = evaluate_xgboost(
    xgb_model,
    X_test,
    y_test
)

print("\nXGBoost Evaluation Metrics:")
print(xgb_results)


arima_model = train_arima(
    sales_data["Total"]
)

arima_results = evaluate_arima(
    arima_model,
    y_test
)

print("\nARIMA Evaluation Metrics:")
print(arima_results)


prophet_model = train_prophet(
    sales_data
)

prophet_results = evaluate_prophet(
    prophet_model,
    test_data
)

print("\nProphet Evaluation Metrics:")
print(prophet_results)

print("\nProphet model trained successfully")


# ---------------------------------------------------
# STEP 13 - TRAIN LSTM
# ---------------------------------------------------

lstm_model, X_lstm, y_lstm = train_lstm(
    sales_data["Total"]
)

# evaluate lstm
lstm_results = evaluate_lstm(
    lstm_model,
    X_lstm,
    y_lstm
)

print("\nLSTM Evaluation Metrics:")
print(lstm_results)



# ---------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------

print("\nComparing model performance...\n")

all_results = {
    "XGBoost": xgb_results,
    "ARIMA": arima_results,
    "Prophet": prophet_results,
    "LSTM": lstm_results
}


for model_name, metrics in all_results.items():

    print(f"\n{model_name} Metrics:")

    for metric_name, value in metrics.items():

        print(f"{metric_name}: {value}")


best_model_name = min(
    all_results,
    key=lambda model: (
        all_results[model]["RMSE"] +
        all_results[model]["MAE"] +
        all_results[model]["MAPE"] -
        all_results[model]["R2_SCORE"]
    )
)

print("\nBest Model Selected:", best_model_name)



# SELECT BEST MODEL OBJECT
if best_model_name == "XGBoost":

    best_model = xgb_model

elif best_model_name == "ARIMA":

    best_model = arima_model

elif best_model_name == "Prophet":

    best_model = prophet_model

else:

    best_model = lstm_model


# ---------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------

print("\nSaving best model...\n")

# create models folder
os.makedirs(
    "models",
    exist_ok=True
)

# save trained model
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

# save model name separately
with open(
    "models/model_name.txt",
    "w"
) as file:

    file.write(best_model_name)

print(f"{best_model_name} saved successfully")


# ---------------------------------------------------
# PIPELINE COMPLETED
# ---------------------------------------------------

print("\nForecasting pipeline completed successfully\n")