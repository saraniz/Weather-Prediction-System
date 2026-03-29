import pandas as pd
import os

#load the data
RAW_PATH = "../data/raw/weather_raw.csv"
PROCESSED_PATH = "../data/raw/weather_clean.csv"

def load_data():
    #check whether the file is available on the given path
    # not keyword is logical operator
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Raw data not found")

    df = pd.read_csv(RAW_PATH)
    print("Loaded data:", df.shape) 
    return df

#clean dataset
def clean_data(df):
    print("Cleaning data...")

    df = df.copy()
    # Remove duplicates
    df = df.drop_duplicates()

    # .loc is used for label based indexing. df.loc[rows, columns]
    # : → means all rows "timestamp" → select that column. So this means: “Select ALL rows of the timestamp column”
    # then replace with converting timestamp into datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.loc[:, "rain_probability"] = df["rain_probability"].fillna(0)

    return df

def create_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # IMPORTANT: sort for time series
    df = df.sort_values(["city", "timestamp"])

    # time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

   # lag features
    df["temp_lag_1"] = df.groupby("city")["temperature"].shift(1)
    df["temp_lag_2"] = df.groupby("city")["temperature"].shift(2)
    df["temp_lag_3"] = df.groupby("city")["temperature"].shift(3)

    df["humidity_lag_1"] = df.groupby("city")["humidity"].shift(1)
    df["pressure_lag_1"] = df.groupby("city")["pressure"].shift(1)

    return df

def finalize_data(df):
    print("Finalizing data...")

    # Drop rows with NaN after lag/rolling
    df = df.dropna()

    return df


def save_data(df):
    df.to_csv(PROCESSED_PATH, index=False)
    print("Processed data saved:", df.shape)

def main():
    print("ETL Pipeline loading ....")

    df = load_data()
    df = clean_data(df)
    df = create_features(df)
    df = finalize_data(df)

    print(df.head())

    save_data(df)

if __name__ == "__main__":
    main()