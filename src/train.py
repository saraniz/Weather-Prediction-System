import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

MODEL_PATH = "../models"
DATA_PATH = '../data/raw/weather_clean.csv'

# targets we want to predict
TARGETS = ["temperature", "humidity", "pressure", "rain_probability"]

# IMPORTANT: must match Streamlit input
FEATURES = [
    "city", "hour", "day", "month",
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "humidity_lag_1",
    "pressure_lag_1"
]

def get_best_mae(target):
    path = os.path.join(MODEL_PATH, f"best_mae_{target}.txt")

    if not os.path.exists(path):
        return float("inf")

    with open(path, "r") as f:
        return float(f.read())


def save_best_mae(target, score):
    path = os.path.join(MODEL_PATH, f"best_mae_{target}.txt")

    os.makedirs(MODEL_PATH, exist_ok=True)

    with open(path, "w") as f:
        f.write(str(score))


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Data loaded:", df.shape)
    return df

#split the data
def split_data(df, target):
    X = df[FEATURES]
    y = df[target]

    # convert categorical feature (city)
    X = pd.get_dummies(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# gt model
def get_models():
    lr = LinearRegression()

    rf = RandomForestRegressor(
        n_estimators=100, # use 100 trees
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBRegressor(
        n_estimators=200, # use 200
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8, # Uses 80% of data for each tree (reduces overfitting).
        colsample_bytree=0.8, #Uses 80% of features per tree.
        random_state=42,
        n_jobs=-1
    )

    return lr, rf, xgb

# train models
def train_models(lr, rf, xgb, X_train, y_train):
    print("Training Linear Regression...")
    lr.fit(X_train, y_train)

    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb.fit(X_train, y_train)

    return lr, rf, xgb

#evaluate models
def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    print(name)
    print("MAE:", round(mae, 4))
    print("-" * 30)

    return mae

#save models
def save_model(model, target_name):
    os.makedirs(MODEL_PATH, exist_ok=True)

    file_path = os.path.join(MODEL_PATH, f"model_{target_name}.pkl")
    joblib.dump(model, file_path)

    print("Saved:", file_path)


def main():
    df = load_data()

    for target in TARGETS:

        print("\n============================")
        print("Training:", target)
        print("============================")

        X_train, X_test, y_train, y_test = split_data(df, target)

        lr, rf, xgb = get_models()
        lr, rf, xgb = train_models(lr, rf, xgb, X_train, y_train)

        # evaluate models
        lr_mae = evaluate(lr, X_test, y_test, "Linear Regression")
        rf_mae = evaluate(rf, X_test, y_test, "Random Forest")
        xgb_mae = evaluate(xgb, X_test, y_test, "XGBoost")

        scores = {
            "lr": lr_mae,
            "rf": rf_mae,
            "xgb": xgb_mae
        }

        best_model_name = min(scores, key=scores.get)

        best_model = {
            "lr": lr,
            "rf": rf,
            "xgb": xgb
        }[best_model_name]

        best_mae = scores[best_model_name]

        print("Best model:", best_model_name)

        old_best_mae = get_best_mae(target)

        print("Old best MAE:", old_best_mae)
        print("New best MAE:", best_mae)

        if best_mae < old_best_mae:
            print("New best model found. Saving...")

            save_model(best_model, target)
            save_best_mae(target, best_mae)
        else:
            print("Model not better. Not saving.")


if __name__ == "__main__":
    main()