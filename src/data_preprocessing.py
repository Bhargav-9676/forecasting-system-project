import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def load_data(path):

    print("Loading dataset...")

    sales_data = pd.read_excel(path)

    sales_data["Date"] = pd.to_datetime(
        sales_data["Date"]
    )

    sales_data = sales_data.sort_values("Date")

    print("Dataset loaded successfully")

    return sales_data


def clean_data(sales_data):

    print("Cleaning dataset...")

    
    sales_data = sales_data.drop_duplicates()

    sales_data["Total"] = sales_data["Total"].ffill()

    
    sales_data = sales_data.dropna(
        subset=["State"]
    )

    print("Data cleaning completed")

    return sales_data




def integrate_data(sales_data):

    print("Data integration completed")

    # currently single dataset
    return sales_data




def transform_data(sales_data):

    print("Transforming dataset...")

   
    encoder = LabelEncoder()

    sales_data["State_Encoded"] = encoder.fit_transform(
        sales_data["State"]
    )

    
    scaler = MinMaxScaler()

    sales_data["Total_Scaled"] = scaler.fit_transform(
        sales_data[["Total"]]
    )

    print("Transformation completed")

    return sales_data



def reduce_data(sales_data):

    print("Reducing unnecessary columns...")

    selected_columns = [
        "State",
        "Date",
        "Total",
        "Category",
        "State_Encoded",
        "Total_Scaled"
    ]

    sales_data = sales_data[selected_columns]

    return sales_data




def handle_outliers(sales_data):

    print("Handling outliers...")

    Q1 = sales_data["Total"].quantile(0.25)
    Q3 = sales_data["Total"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    sales_data["Total"] = np.where(
        sales_data["Total"] > upper_limit,
        upper_limit,
        sales_data["Total"]
    )

    sales_data["Total"] = np.where(
        sales_data["Total"] < lower_limit,
        lower_limit,
        sales_data["Total"]
    )

    print("Outliers handled")

    return sales_data



def validate_data(sales_data):

    print("Validating dataset...")

    print("Dataset Shape:",sales_data.shape)

    print("Missing Values:", sales_data.isnull().sum())

    print("Validation completed sucessfully")