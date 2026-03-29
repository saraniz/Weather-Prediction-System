import pandas as pd
import joblib

def test_data_loading():
    df = pd.read_csv("data/weather.csv")
    assert df.shape[0] > 0

def test_model_file_exists():
    import os
    assert os.path.exists("models/model.pkl")