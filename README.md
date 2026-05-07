# Sales Forecasting System using Machine Learning and Time-Series Models

## Project Overview

This project is an AI-powered Sales Forecasting System developed using Machine Learning and Time-Series Forecasting techniques.

The system predicts future sales for different states using historical sales data.

## Technologies Used

- Python
- Pandas
- NumPy
- FastAPI
- Prophet
- ARIMA
- XGBoost
- TensorFlow/Keras
- Scikit-learn

## Project Structure

```bash
forecasting-system-project/
│
├── data/
├── models/
├── src/
├── api/
├── train_pipeline.py
├── requirements.txt
└── README.md
```

## Features

- Data preprocessing
- Feature engineering
- Prophet forecasting
- ARIMA forecasting
- XGBoost forecasting
- Ensemble forecasting
- FastAPI deployment
- Swagger UI integration

## API Endpoints

### Home Endpoint

```bash
GET /
```

### Simple Forecast Endpoint

```bash
GET /forecast/simple?state=Texas
```

### Detailed Forecast Endpoint

```bash
GET /forecast/detailed?state=Texas
```

## Run Project

### Create Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python train_pipeline.py
```

### Run API

```bash
uvicorn api.app:app --reload
```

### Open Swagger

```bash
http://127.0.0.1:8000/docs
```

## Conclusion

This project combines Machine Learning and Time-Series Forecasting models to generate realistic future sales predictions using historical sales data.
