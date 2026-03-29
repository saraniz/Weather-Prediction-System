# Weather Prediction System

Live App: https://weather-prediction--system.streamlit.app/

## Project Overview

This project is a machine learning powered weather forecasting dashboard built with Streamlit.
It predicts key weather parameters for Sri Lankan cities:

- Temperature
- Humidity
- Pressure
- Rain Probability

The app allows users to select a city, choose a target parameter (or view all), and generate a multi-day forecast.

## Why I Built This

I built this as part of my MLOps learning journey.
My goal was not only to train models, but to build a complete pipeline from data collection to deployment.

Instead of relying only on fixed, old datasets, I learned to use real-world API data and create a repeatable workflow that can continue to improve over time.

## Live Demo

Use the deployed app here:

https://weather-prediction--system.streamlit.app/

## My MLOps Journey (Step by Step)

### 1) Data Collection from Real APIs

- I used OpenWeather API to fetch both current weather and forecast data.
- I collected data for multiple cities.
- I stored raw data in CSV format so it can be appended over time.

File used: src/fetch_data.py

### 2) ETL and Feature Engineering

- Cleaned duplicates and missing values.
- Converted timestamps and sorted by city and time.
- Created time-based features:
	- hour
	- day
	- month
- Created lag features:
	- temp_lag_1, temp_lag_2, temp_lag_3
	- humidity_lag_1
	- pressure_lag_1

File used: src/etl.py

### 3) Model Training and Selection

For each target variable, I trained multiple models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Then I compared models using MAE and saved only the best model if performance improved.
This made the training process version-aware at a basic level.

File used: src/train.py

### 4) Model Artifact Management

- Best models are saved in the models folder.
- Best MAE values are tracked in text files for each target.

This helped me avoid replacing a good model with a worse one.

### 5) Serving Predictions with Streamlit

- Loaded trained models with joblib.
- Used latest known city row as the seed input.
- Generated future predictions day by day.
- Displayed KPIs, tables, and trend charts.

File used: app.py

### 6) Deployment and Production Lessons

- Deployed on Streamlit Community Cloud.
- Learned to handle environment and dependency compatibility issues.
- Updated dependency strategy to support cloud Python versions reliably.

### 7) CI/CD Workflow (GitHub + Streamlit Cloud)

- Used Git and GitHub as the source of truth for code changes.
- Built CI/CD for a real-world API-driven ML system, not a one-time static dataset.
- Used GitHub Actions to run model retraining regularly so models stay updated with incoming weather data.
- Added automated test execution in the pipeline before model updates are pushed.
- Followed a push-to-deploy workflow with Streamlit Community Cloud.
- When CI commits updated artifacts or code changes are pushed to main, Streamlit automatically rebuilds and deploys the app.
- Used build and deployment logs to debug runtime/dependency issues and improve reliability.

Workflow file: .github/workflows/retrain.yml

### CI/CD Retraining Flow (API-Driven)

The retraining pipeline is designed for live API data, not fixed static datasets.

1. New weather data is collected from API-based sources.
2. CI/CD runs validation and tests.
3. Models are retrained to reflect newer data patterns.
4. Updated model artifacts are committed to the repository.
5. Streamlit Cloud automatically redeploys from the updated main branch.

## What I Learned

### 1) Real-world data is messy

Working with API data taught me much more than static datasets.
I had to deal with missing values, timestamp handling, and evolving data quality.

### 2) Building the pipeline matters as much as model accuracy

Model training is only one part.
Data ingestion, ETL, reproducibility, artifact management, and deployment are equally important.

### 3) Version and environment management are critical in MLOps

A model can work locally and still fail in deployment due to Python/package mismatch.
I learned to manage runtime and dependency compatibility for cloud deployment.

### 4) Continuous improvement mindset

Using live data means this system can be retrained and improved over time, unlike one-time academic projects.

### 5) CI/CD is part of MLOps, not optional

Automated cloud rebuilds from repository updates made deployment repeatable and faster.
I learned that monitoring build logs and managing dependencies are key CI/CD skills in real ML projects.
I also learned that for API-driven projects, CI/CD should include testing plus scheduled retraining, not only app redeployment.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Joblib
- Requests

## Project Structure

```
Weather Prediction/
|-- app.py
|-- requirements.txt
|-- runtime.txt
|-- .python-version
|-- data/
|   `-- raw/
|       |-- weather_raw.csv
|       `-- weather_clean.csv
|-- models/
|   |-- model_temperature.pkl
|   |-- model_humidity.pkl
|   |-- model_pressure.pkl
|   |-- model_rain_probability.pkl
|   |-- best_mae_temperature.txt
|   |-- best_mae_humidity.txt
|   |-- best_mae_pressure.txt
|   `-- best_mae_rain_probability.txt
`-- src/
		|-- fetch_data.py
		|-- etl.py
		`-- train.py
```

## How to Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Future Improvements

- Add model versioning with MLflow or DVC.
- Add automated retraining pipeline (scheduled jobs).
- Add experiment tracking dashboard.
- Add CI/CD checks for data and model quality.
- Move from CSV to a database or data lake for larger-scale data.

## Author Note

This project represents my transition from model-centric learning to full MLOps thinking.
It is not only about predicting weather, but about learning how to build reliable ML systems end to end.
