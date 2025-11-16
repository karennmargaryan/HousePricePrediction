from pathlib import Path

import pandas as pd
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

ROOT_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'

model = joblib.load(ARTIFACTS_DIR / "random_forest_model.joblib")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

with open(ARTIFACTS_DIR / "model_columns.json", "r") as f:
    model_columns = json.load(f)

numerical_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_renovated',
    'sale_month', 'house_age'
]

categorical_features = ['city']


def preprocess_new_data(data):
    """
    Applies the full preprocessing pipeline to new,
    incoming data for prediction.
    """
    df = pd.DataFrame(data, index=[0])

    df['date'] = pd.to_datetime(df['date'])
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    df['house_age'] = df['sale_year'] - df['yr_built']

    df[numerical_features] = scaler.transform(df[numerical_features])

    df_cat = pd.get_dummies(df[categorical_features], drop_first=True)

    df_processed = pd.concat([df[numerical_features], df_cat], axis=1)
    df_aligned = df_processed.reindex(columns=model_columns, fill_value=0)

    return df_aligned


@app.route("/predict", methods=['POST'])
def predict():
    """
    Receives new data via POST request, preprocesses it,
    makes a prediction, and returns the result.
    """
    try:
        new_data = request.get_json()

        processed_data = preprocess_new_data(new_data)

        log_prediction = model.predict(processed_data)

        final_prediction = np.exp(log_prediction)

        return jsonify({
            "status": "success",
            "prediction_dollars": final_prediction[0]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)