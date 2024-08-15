# estimator.py

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def load_and_evaluate(model_path, preprocessing_path, test_data_path):
    # Load the model and preprocessing pipeline
    model = joblib.load(model_path)
    preprocessing_pipeline = joblib.load(preprocessing_path)

    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Extract true labels before transformation
    y_test = test_data["stroke"].values  # Adjust based on your dataset

    # Apply preprocessing to the test data
    transformed_data = preprocessing_pipeline.transform(test_data)

    # Make predictions
    predictions = model.predict(transformed_data)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")


def predict(model_path, preprocessing_path, user_data):
    # Load the model and preprocessing pipeline
    model = joblib.load(model_path)
    preprocessing_pipeline = joblib.load(preprocessing_path)

    # Ensure the user data has all required columns with correct data types
    required_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    # Reorder and fill missing columns if necessary
    for col in required_columns:
        if col not in user_data.columns:
            user_data[col] = 0

    # Apply preprocessing to the user data
    transformed_data = preprocessing_pipeline.transform(user_data)

    # Make predictions
    predictions = model.predict(transformed_data)
    return predictions
