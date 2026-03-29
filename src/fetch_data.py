import requests
import pandas as pd
import os
from datetime import datetime

# ==============================
# CONFIG
# ==============================

API_KEY = "18d72659c9873b0a6465c49d14d91e90"

CITIES = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612},
    "Gampaha": {"lat": 7.0873, "lon": 79.9992},
    "Kalutara": {"lat": 6.5854, "lon": 79.9607},
    "Kandy": {"lat": 7.2906, "lon": 80.6337},
    "Matale": {"lat": 7.4675, "lon": 80.6234},
    "Nuwara Eliya": {"lat": 6.9497, "lon": 80.7891},
    "Galle": {"lat": 6.0535, "lon": 80.2210},
    "Matara": {"lat": 5.9549, "lon": 80.5550},
    "Hambantota": {"lat": 6.1246, "lon": 81.1185},
    "Jaffna": {"lat": 9.6615, "lon": 80.0255},
    "Kilinochchi": {"lat": 9.3803, "lon": 80.3760},
    "Mannar": {"lat": 8.9810, "lon": 79.9042},
    "Vavuniya": {"lat": 8.7514, "lon": 80.4971},
    "Mullaitivu": {"lat": 9.2671, "lon": 80.8142},
    "Batticaloa": {"lat": 7.7170, "lon": 81.7000},
    "Ampara": {"lat": 7.2975, "lon": 81.6820},
    "Trincomalee": {"lat": 8.5874, "lon": 81.2152},
    "Kurunegala": {"lat": 7.4863, "lon": 80.3647},
    "Puttalam": {"lat": 8.0362, "lon": 79.8283},
    "Anuradhapura": {"lat": 8.3114, "lon": 80.4037},
    "Polonnaruwa": {"lat": 7.9403, "lon": 81.0188},
    "Badulla": {"lat": 6.9934, "lon": 81.0550},
    "Monaragala": {"lat": 6.8728, "lon": 81.3507},
    "Ratnapura": {"lat": 6.7056, "lon": 80.3847},
    "Kegalle": {"lat": 7.2513, "lon": 80.3464}
}

RAW_PATH = "../data/raw/weather_raw.csv"

os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

# get current weather
def fetch_current(city, lat, lon):

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    )

    # create request for api and get response
    response = requests.get(url)
    # api return data as json and .json conver it to python dictionary
    data = response.json()

    # 200 mean success
    if response.status_code != 200:
        print(f"{city} CURRENT ERROR:", response.text)
        return None

    return {
        "city": city,
        "type": "current",
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "timestamp": datetime.utcnow()
    }


# forecast
def fetch_forecast(city, lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print(f"{city} FORECAST ERROR:", response.text)
        return []

    # create empty list to store forecast data
    records = []

    # "list" is a key in the JSON response from the OpenWeather API.
    for item in data.get("list", []):
        records.append({
            "city": city,
            "type": "forecast",
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "wind_speed": item["wind"]["speed"],
            "rain_probability": item.get("pop"),
            "timestamp": datetime.fromtimestamp(item["dt"])
        })

    return records


def fetch_all_data():
    all_data = []

    for city, coords in CITIES.items():
        print(f"Fetching {city}")

        # call fetch current data function to get data for relevant cities
        current = fetch_current(city, coords["lat"], coords["lon"])
        if current:
            all_data.append(current)

        # then get the forecast data for that city
        forecast = fetch_forecast(city, coords["lat"], coords["lon"])
        # append() → adds one item. extend() → adds all items from another list
        all_data.extend(forecast)

    return pd.DataFrame(all_data)

# save the data
def save_data(df):
    if df.empty:
        print("No data to save")
        return

    if os.path.exists(RAW_PATH):
        df.to_csv(RAW_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_PATH, index=False)

    print(f"Saved data: {df.shape}")

# __name__ == "__main__" → entry point of the program
if __name__ == "__main__":
    df = fetch_all_data()
    save_data(df)