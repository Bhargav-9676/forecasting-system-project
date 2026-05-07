

import pandas as pd


def create_features(df):

   
    df["lag1"] = df.groupby("State")["Total"].shift(1)
    df["lag7"] = df.groupby("State")["Total"].shift(7)
    df["lag30"] = df.groupby("State")["Total"].shift(30)

    
    df["rolling_mean7"] = df.groupby("State")["Total"].shift(1).rolling(7).mean()
    df["rolling_std7"] = df.groupby("State")["Total"].shift(1).rolling(7).std()


    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month


    df = df.dropna()

    return df