import pandas as pd
from heart_disease_prediction.components import estimator


class StrokePredictionPipeline:
    def __init__(self, model_file, preprocessing_file):
        self.model_file = model_file
        self.preprocessing_file = preprocessing_file

    def make_prediction(self, input_data: dict) -> str:
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure relevant columns are numeric
        numeric_fields = [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
        ]
        for field in numeric_fields:
            input_df[field] = pd.to_numeric(input_df[field])

        # Generate prediction using the estimator
        prediction_result = predict(
            self.model_file, self.preprocessing_file, input_df
        )
        return "Stroke" if prediction_result[0] == 1 else "No Stroke"
