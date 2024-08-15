# prediction_pipeline.py

import pandas as pd
from heart_disease_prediction.components import estimator


class PredictionPipeline:
    def __init__(self, model_path, preprocessing_path):
        self.model_path = model_path
        self.preprocessing_path = preprocessing_path

    def predict(self, user_input: dict) -> str:
        # Convert user input into DataFrame
        user_df = pd.DataFrame([user_input])

        # Convert appropriate columns to numeric
        numeric_columns = [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
        ]
        for column in numeric_columns:
            user_df[column] = pd.to_numeric(user_df[column])

        # Predict using the estimator
        prediction = estimator.predict(
            self.model_path, self.preprocessing_path, user_df
        )
        return "Stroke" if prediction[0] == 1 else "No Stroke"
